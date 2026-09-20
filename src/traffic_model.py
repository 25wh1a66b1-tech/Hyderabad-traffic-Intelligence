"""Dataset-backed BPR traffic metrics and vehicle-aware route recommendations."""
from __future__ import annotations

from itertools import islice
from typing import Dict, Iterable, Tuple

import networkx as nx
import pandas as pd

from src.data_loader import load_dataset_frames


VEHICLE_LATENCY = {
    "Passenger Vehicle": 1.0,
    "Light Commercial EV": 0.95,
    "Standard Bus": 1.2,
    "Heavy Logistics Truck": 1.45,
}


def _current_segment_snapshot() -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, traffic, incidents = load_dataset_frames()
    latest_time = traffic["timestamp"].max()
    recent = traffic[traffic["timestamp"] == latest_time].copy()
    if recent.empty:
        recent = traffic.tail(1).copy()
    summary = (
        recent.groupby("segment_id", as_index=False)
        .agg(avg_speed_kmh=("speed_kmh", "mean"), avg_flow_vph=("flow_vph", "mean"), avg_travel_time_min=("travel_time_min", "mean"))
        .set_index("segment_id")
    )
    active_incidents = incidents[incidents["end_time"] >= latest_time].copy()
    return summary, active_incidents


def _edge_bundle_cost(edge_bundle: dict) -> float:
    if not edge_bundle:
        return 1.0
    values = edge_bundle.values() if isinstance(edge_bundle, dict) else [edge_bundle]
    return min((float(edge.get("travel_time", 1.0)) for edge in values if edge), default=1.0)


def get_edge_cost(_u: str, _v: str, data: dict) -> float:
    return _edge_bundle_cost(data)


def assign_edge_metrics(graph: nx.MultiDiGraph, demand_factor: float = 1.0, capacity_reduction: float = 0.0) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """Attach BPR volume, capacity, speed, travel-time, and V/C metrics to every edge."""
    snapshot, _ = _current_segment_snapshot()
    metrics: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    reduction = min(max(float(capacity_reduction), 0.0), 0.95)
    demand = max(float(demand_factor), 0.0)

    for u, v, key, data in graph.edges(keys=True, data=True):
        segment_id = str(data.get("segment_id", key))
        current = snapshot.loc[segment_id] if segment_id in snapshot.index else None
        free_speed = max(float(data.get("free_flow_speed_kmh") or 35.0), 5.0)
        length_km = max(float(data.get("length_km") or 1.0), 0.01)
        base_capacity = max(float(data.get("capacity_vph") or 1000.0), 1.0)
        capacity = base_capacity * (1.0 - reduction)
        observed_flow = float(current["avg_flow_vph"]) if current is not None else base_capacity * 0.35
        volume = max(observed_flow * demand, 1.0)
        ratio = volume / capacity
        free_flow_time = length_km / free_speed * 60.0
        travel_time = free_flow_time * (1.0 + 0.15 * ratio**4)
        speed = max(length_km / max(travel_time / 60.0, 1e-6), 1.0)
        values = {"volume": float(volume), "capacity": float(capacity), "vc_ratio": float(ratio), "congestion": float(ratio), "travel_time": float(travel_time), "speed": float(speed), "free_flow_speed": float(free_speed)}
        graph[u][v][key].update(values)
        metrics[(u, v, key)] = values
    return metrics


def synthetic_incident_edges(graph: nx.MultiDiGraph, metrics: Dict[Tuple[str, str, str], Dict[str, float]], severity: float = 0.7):
    """Apply active incident multipliers to matching dataset segments."""
    incident_edges = []
    try:
        _, incidents = _current_segment_snapshot()
        for _, incident in incidents.iterrows():
            segment_id = str(incident["segment_id"])
            multiplier = 1.0 + max(float(severity), 0.0) + float(incident.get("severity", 0.0))
            for u, v, key, data in graph.edges(keys=True, data=True):
                edge_key = (u, v, key)
                if str(data.get("segment_id")) != segment_id or edge_key not in metrics:
                    continue
                values = metrics[edge_key]
                values["volume"] *= multiplier
                values["vc_ratio"] = values["volume"] / max(values["capacity"], 1.0)
                values["congestion"] = values["vc_ratio"]
                values["travel_time"] *= 1.0 + 0.55 * multiplier
                values["speed"] = max(values["speed"] / multiplier, 1.0)
                graph[u][v][key].update(values)
                incident_edges.append(edge_key)
    except (KeyError, TypeError, ValueError, OSError):
        return incident_edges
    return incident_edges


def _best_edge(graph: nx.MultiDiGraph, source: str, target: str) -> dict | None:
    bundle = graph.get_edge_data(source, target, default={})
    if not bundle:
        return None
    return min(bundle.values(), key=lambda edge: float(edge.get("travel_time", 1.0)))


def compute_route_metrics(graph: nx.MultiDiGraph, path: Iterable[str], vehicle_profile: str = "Passenger Vehicle") -> dict:
    nodes = list(path)
    latency = VEHICLE_LATENCY.get(vehicle_profile, 1.0)
    total_time = 0.0
    congestion_total = 0.0
    speed_total = 0.0
    edge_count = 0
    for source, target in zip(nodes, nodes[1:]):
        edge = _best_edge(graph, source, target)
        if edge is None:
            continue
        total_time += float(edge.get("travel_time", 1.0)) * latency
        congestion_total += float(edge.get("vc_ratio", edge.get("congestion", 0.0)))
        speed_total += float(edge.get("speed", 25.0)) / latency
        edge_count += 1
    return {"path": nodes, "route_time": total_time, "avg_congestion": congestion_total / max(edge_count, 1), "avg_speed": speed_total / max(edge_count, 1), "vehicle_profile": vehicle_profile}


def candidate_routes(graph: nx.MultiDiGraph, origin: str, destination: str, k: int = 3) -> list[list[str]]:
    if origin == destination:
        return [[origin]]
    try:
        routing_graph = nx.DiGraph()
        for source, target, bundle in graph.edges(data=True):
            cost = _edge_bundle_cost(graph.get_edge_data(source, target, default=bundle))
            if not routing_graph.has_edge(source, target) or cost < routing_graph[source][target]["travel_time"]:
                routing_graph.add_edge(source, target, travel_time=cost)
        generator = nx.shortest_simple_paths(routing_graph, origin, destination, weight="travel_time")
        paths = list(islice(generator, max(int(k), 1)))
        return paths or [nx.shortest_path(routing_graph, origin, destination, weight="travel_time")]
    except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError, TypeError, ValueError):
        try:
            routing_graph = nx.DiGraph()
            for source, target in graph.edges():
                cost = _edge_bundle_cost(graph.get_edge_data(source, target, default={}))
                routing_graph.add_edge(source, target, travel_time=cost)
            return [nx.shortest_path(routing_graph, origin, destination, weight="travel_time")]
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            return []


def route_recommendation(graph: nx.MultiDiGraph, origin: str, destination: str, vehicle_profile: str = "Passenger Vehicle") -> dict:
    """Return the primary route plus up to two valid alternatives."""
    routes = candidate_routes(graph, str(origin), str(destination), k=3)
    scored = [compute_route_metrics(graph, path, vehicle_profile) for path in routes if path and path[-1] == str(destination)]
    if not scored:
        fallback = compute_route_metrics(graph, [str(origin), str(destination)], vehicle_profile)
        return {"best": fallback, "alternatives": []}
    scored.sort(key=lambda result: result["route_time"])
    return {"best": scored[0], "alternatives": scored[1:3]}
