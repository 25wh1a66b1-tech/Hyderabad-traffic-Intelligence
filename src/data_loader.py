from __future__ import annotations

import math
import os
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import pandas as pd


DEFAULT_DATASET_PATH = Path(r"C:\Users\kattu\Downloads\NEURAX_SMART_CITIES_TRAINING_V2.zip")


def _find_dataset_zip() -> Path | None:
    candidates = [
        DEFAULT_DATASET_PATH,
        Path.cwd() / "NEURAX_SMART_CITIES_TRAINING_V2.zip",
        Path.cwd().parent / "NEURAX_SMART_CITIES_TRAINING_V2.zip",
    ]
    for path in candidates:
        if path and path.exists():
            return path
    return None


def load_dataset_frames():
    zip_path = _find_dataset_zip()
    if zip_path is None:
        raise FileNotFoundError(
            "Smart Cities dataset ZIP not found. Please place NEURAX_SMART_CITIES_TRAINING_V2.zip in Downloads or the project folder."
        )

    with zipfile.ZipFile(zip_path) as archive:
        nodes = pd.read_csv(archive.open("nodes.csv"))
        network = pd.read_csv(archive.open("network.csv"))
        traffic = pd.read_csv(archive.open("traffic_train.csv"), parse_dates=["timestamp"])
        incidents = pd.read_csv(archive.open("incidents_train.csv"), parse_dates=["start_time", "end_time"])

    return nodes, network, traffic, incidents


def build_city_graph() -> nx.MultiDiGraph:
    """Build the road network directly from the provided smart-city dataset."""
    nodes, network, _, _ = load_dataset_frames()

    graph = nx.MultiDiGraph()
    node_map = {}

    for _, row in nodes.iterrows():
        node_id = str(row["node_id"])
        lon = float(row["lon"])
        lat = float(row["lat"])
        node_map[node_id] = {"x": lon, "y": lat, "lon": lon, "lat": lat}
        graph.add_node(node_id, **node_map[node_id])

    for _, row in network.iterrows():
        source = str(row["source_node"])
        target = str(row["target_node"])
        segment_id = str(row["segment_id"])
        attrs = {key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()}
        attrs["segment_id"] = segment_id
        graph.add_edge(source, target, key=segment_id, **attrs)

    return graph


def select_route_nodes(graph: nx.MultiDiGraph) -> Tuple[str, str]:
    """Choose a corridor from the real smart-city network."""
    if graph.number_of_nodes() == 0:
        raise ValueError("No graph nodes found in the smart-city dataset.")

    nodes = list(graph.nodes())
    sample = nodes[: min(80, len(nodes))]
    if len(sample) < 2:
        return nodes[0], nodes[0]

    origin = sample[0]
    destination = sample[-1]
    max_distance = -1.0
    max_pair = (origin, destination)

    for n1 in sample:
        for n2 in sample:
            if n1 == n2:
                continue
            if not nx.has_path(graph, n1, n2):
                continue
            x1, y1 = graph.nodes[n1]["x"], graph.nodes[n1]["y"]
            x2, y2 = graph.nodes[n2]["x"], graph.nodes[n2]["y"]
            dist = math.dist((x1, y1), (x2, y2))
            if dist > max_distance:
                max_distance = dist
                max_pair = (n1, n2)

    return max_pair


def get_network_summary(graph: nx.MultiDiGraph) -> Dict[str, float]:
    """Summarize the dataset-backed graph state for the dashboard."""
    if graph.number_of_nodes() == 0:
        return {"nodes": 0.0, "edges": 0.0, "avg_speed": 0.0}

    speeds = []
    for _, _, data in graph.edges(data=True):
        raw_speed = data.get("free_flow_speed_kmh")
        if raw_speed is not None:
            try:
                speeds.append(float(raw_speed))
            except (TypeError, ValueError):
                pass

    avg_speed = sum(speeds) / len(speeds) if speeds else 35.0
    return {
        "nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "avg_speed": float(avg_speed),
    }
