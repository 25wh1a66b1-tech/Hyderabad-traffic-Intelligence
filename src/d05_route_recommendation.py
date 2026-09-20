from __future__ import annotations

from itertools import islice
from typing import Dict, Iterable

import networkx as nx


def _edge_bundle_cost(edge_bundle):
    """Select the fastest parallel segment in a MultiDiGraph edge bundle."""
    if not edge_bundle:
        return 1.0
    values = edge_bundle.values() if isinstance(edge_bundle, dict) else [edge_bundle]
    costs = [float(edge.get("travel_time", 1.0)) for edge in values if edge]
    return min(costs, default=1.0)


def get_edge_cost(u, v, data: Dict):
    return _edge_bundle_cost(data)


def candidate_routes(graph: nx.MultiDiGraph, origin: str, destination: str, k: int = 3):
    """Find up to k diverse shortest routes using simple-path enumeration."""
    if origin == destination:
        return [[origin]]
    try:
        paths = list(islice(nx.shortest_simple_paths(graph, origin, destination, weight=get_edge_cost), k))
        if not paths:
            return [nx.shortest_path(graph, origin, destination, weight=get_edge_cost)]
        return paths[:k]
    except Exception:
        try:
            return [nx.shortest_path(graph, origin, destination, weight=get_edge_cost)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [[origin, destination]]


def compute_route_metrics(graph: nx.MultiDiGraph, path: Iterable[str]):
    path = list(path)
    if len(path) < 2:
        return {"path": path, "route_time": 0.0, "avg_congestion": 0.0, "avg_speed": 0.0}

    total_time = 0.0
    total_congestion = 0.0
    total_speed = 0.0
    count = 0

    for i in range(len(path) - 1):
        neighbors = graph.get_edge_data(path[i], path[i + 1], default={})
        if not neighbors:
            continue
        edge_data = min(neighbors.values(), key=lambda edge: float(edge.get("travel_time", 1.0)))
        total_time += float(edge_data.get("travel_time", 1.0))
        total_congestion += float(edge_data.get("congestion", 0.0))
        total_speed += float(edge_data.get("speed", 25.0))
        count += 1

    return {
        "path": path,
        "route_time": float(total_time),
        "avg_congestion": float(total_congestion / max(count, 1)),
        "avg_speed": float(total_speed / max(count, 1)),
    }


def route_recommendation(graph: nx.MultiDiGraph, origin: str, destination: str):
    routes = candidate_routes(graph, origin, destination, k=3)
    scored = [compute_route_metrics(graph, route) for route in routes]
    scored.sort(key=lambda r: r["route_time"])
    best = scored[0]
    return {"best": best, "alternatives": scored[1:3]}
