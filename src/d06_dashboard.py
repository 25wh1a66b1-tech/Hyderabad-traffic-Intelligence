from __future__ import annotations

import ast

import pandas as pd
import pydeck as pdk


def congestion_color(value: float):
    if pd.isna(value):
        value = 0.0
    if value < 0.4:
        return [16, 255, 160]
    if value < 0.8:
        return [255, 190, 35]
    return [255, 55, 82]


def _geometry_path(raw_geometry, start_lon: float, start_lat: float, end_lon: float, end_lat: float):
    """Parse stored road geometry, falling back to the dataset node endpoints."""
    has_geometry = raw_geometry is not None
    if isinstance(raw_geometry, float) and pd.isna(raw_geometry):
        has_geometry = False
    if has_geometry:
        try:
            geometry = ast.literal_eval(raw_geometry) if isinstance(raw_geometry, str) else raw_geometry
            if isinstance(geometry, (list, tuple)) and len(geometry) >= 2:
                return [[float(point[0]), float(point[1])] for point in geometry]
        except (ValueError, SyntaxError, TypeError, IndexError):
            pass
    return [[start_lon, start_lat], [end_lon, end_lat]]


def build_segment_map_layer(network_df: pd.DataFrame, nodes_df: pd.DataFrame, latest_segment_stats: pd.DataFrame) -> pdk.Layer:
    node_lookup = nodes_df.set_index("node_id")
    merged = network_df.copy()
    merged["source_node"] = merged["source_node"].astype(str)
    merged["target_node"] = merged["target_node"].astype(str)
    merged["start_lat"] = merged["source_node"].map(node_lookup["lat"])
    merged["start_lon"] = merged["source_node"].map(node_lookup["lon"])
    merged["end_lat"] = merged["target_node"].map(node_lookup["lat"])
    merged["end_lon"] = merged["target_node"].map(node_lookup["lon"])

    stats = latest_segment_stats.copy()
    stats["segment_id"] = stats["segment_id"].astype(str)
    stat_columns = ["segment_id", "congestion_index", "avg_speed_kmh"]
    merged = merged.merge(stats[stat_columns], on="segment_id", how="left")
    merged["congestion_index"] = merged["congestion_index"].fillna(0.3)

    path_data = []
    for _, row in merged.iterrows():
        path_data.append({
            "path": _geometry_path(
                row.get("geometry"),
                float(row["start_lon"]),
                float(row["start_lat"]),
                float(row["end_lon"]),
                float(row["end_lat"]),
            ),
            "color": congestion_color(float(row["congestion_index"])),
            "congestion_index": float(row["congestion_index"]),
            "avg_speed_kmh": float(row["avg_speed_kmh"]) if pd.notna(row["avg_speed_kmh"]) else None,
            "segment_id": str(row["segment_id"]),
            "road_class": str(row.get("road_class", "road")),
        })

    return pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        get_color="color",
        get_width=3.5,
        width_scale=10,
        width_min_pixels=1.5,
        width_max_pixels=7,
        pickable=True,
        opacity=0.9,
    )


def build_route_layers(route_points: list[list[float]]) -> list[pdk.Layer]:
    """Render a continuous route with a broad glow and a crisp center line."""
    if len(route_points) < 2:
        return []

    route_data = [{"path": route_points}]
    return [
        pdk.Layer(
            "PathLayer",
            data=route_data,
            get_path="path",
            get_color=[59, 130, 246, 90],
            get_width=28,
            width_scale=1,
            width_min_pixels=10,
            width_max_pixels=24,
            opacity=0.55,
        ),
        pdk.Layer(
            "PathLayer",
            data=route_data,
            get_path="path",
            get_color=[59, 130, 246, 255],
            get_width=8,
            width_scale=1,
            width_min_pixels=4,
            width_max_pixels=10,
            pickable=True,
            opacity=0.98,
        ),
    ]


def build_incident_pin_layer(incidents_df: pd.DataFrame, network_df: pd.DataFrame, nodes_df: pd.DataFrame, target_timestamp: pd.Timestamp) -> list[pdk.Layer]:
    empty_layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=[],
            get_position="[0, 0]",
            get_radius=0,
            get_fill_color=[0, 0, 0, 0],
            opacity=0,
        ),
    ]
    if incidents_df.empty:
        return empty_layers

    node_lookup = nodes_df.set_index("node_id")
    active = incidents_df[(incidents_df["start_time"] <= target_timestamp) & (incidents_df["end_time"] >= target_timestamp)].copy()
    if active.empty:
        return empty_layers

    focused = network_df[["segment_id", "source_node", "target_node"]].copy()
    focused["source_node"] = focused["source_node"].astype(str)
    focused["target_node"] = focused["target_node"].astype(str)
    focused["start_lat"] = focused["source_node"].map(node_lookup["lat"])
    focused["start_lon"] = focused["source_node"].map(node_lookup["lon"])
    focused["end_lat"] = focused["target_node"].map(node_lookup["lat"])
    focused["end_lon"] = focused["target_node"].map(node_lookup["lon"])

    incident_points = []
    for _, row in active.iterrows():
        segment = focused[focused["segment_id"].astype(str) == str(row["segment_id"])].iloc[0]
        center_lon = (float(segment["start_lon"]) + float(segment["end_lon"])) / 2.0
        center_lat = (float(segment["start_lat"]) + float(segment["end_lat"])) / 2.0
        incident_points.append({
            "lon": center_lon,
            "lat": center_lat,
            "severity": float(row["severity"]),
            "incident_type": str(row["incident_type"]),
            "marker": "▲",
            "color": [255, 55, 82, 220],
        })

    return [
        pdk.Layer(
            "ScatterplotLayer",
            data=incident_points,
            get_position="[lon, lat]",
            get_radius=260,
            get_fill_color=[255, 35, 70, 80],
            stroked=False,
            opacity=0.42,
        ),
        pdk.Layer(
            "TextLayer",
            data=incident_points,
            get_position="[lon, lat]",
            get_text="marker",
            get_color=[255, 55, 82, 255],
            get_size=25,
            get_alignment_baseline="center",
            get_text_anchor="middle",
            pickable=True,
        ),
    ]


def build_trend_series(traffic_df: pd.DataFrame, target_timestamp: pd.Timestamp, window_hours: int = 3) -> pd.DataFrame:
    df = traffic_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    window_start = target_timestamp - pd.Timedelta(hours=window_hours)
    series = df[(df["timestamp"] >= window_start) & (df["timestamp"] <= target_timestamp)].copy()
    if series.empty:
        return pd.DataFrame({"time": [target_timestamp], "congestion_index": [0.0]})
    series["bucket"] = series["timestamp"].dt.floor("15min")
    agg = series.groupby("bucket", as_index=False)["congestion_index"].mean()
    agg = agg.rename(columns={"bucket": "time", "congestion_index": "congestion_index"})
    return agg.sort_values("time").reset_index(drop=True)


def build_risk_series(traffic_df: pd.DataFrame, target_timestamp: pd.Timestamp, window_hours: int = 3) -> pd.DataFrame:
    df = traffic_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    window_start = target_timestamp - pd.Timedelta(hours=window_hours)
    series = df[(df["timestamp"] >= window_start) & (df["timestamp"] <= target_timestamp)].copy()
    if series.empty:
        return pd.DataFrame({"time": [target_timestamp], "incident_risk": [0.0]})
    series["bucket"] = series["timestamp"].dt.floor("15min")
    agg = series.groupby("bucket", as_index=False)["delay_min"].mean()
    agg["incident_risk"] = (agg["delay_min"] / agg["delay_min"].max() if agg["delay_min"].max() else 0.0)
    agg = agg.rename(columns={"bucket": "time"})
    return agg[["time", "incident_risk"]].sort_values("time").reset_index(drop=True)


def deck_view_state(latitude: float = 17.385, longitude: float = 78.4867, zoom: int = 11):
    return pdk.ViewState(latitude=latitude, longitude=longitude, zoom=zoom, pitch=52, bearing=-8)
