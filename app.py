from __future__ import annotations

import pandas as pd
import networkx as nx
import streamlit as st
import pydeck as pdk

from src.data_loader import build_city_graph, get_network_summary, load_dataset_frames, select_route_nodes
from src.traffic_model import assign_edge_metrics, route_recommendation, synthetic_incident_edges
from src.d03_detect_incidents import detect_anomalies, active_incidents_for_timestamp
from src.d04_forecast_traffic import SpatialTrafficTrainer, anomaly_tracker, segment_forecast_window
from src.auth import ROLE_AUTHORITIES, ROLE_FLEET, ROLE_RESEARCH, ROLE_SECURITY, current_user, login_form, logout, has_role, ensure_traffic_database, save_road_status_snapshot
from src.d06_dashboard import (
    build_segment_map_layer,
    build_incident_pin_layer,
    build_route_layers,
    build_trend_series,
    build_risk_series,
    deck_view_state,
)


st.set_page_config(page_title="Hyderabad Traffic Intelligence", page_icon="🚦", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 15%, rgba(16, 185, 129, 0.16), transparent 20%),
            radial-gradient(circle at 85% 85%, rgba(59, 130, 246, 0.2), transparent 26%),
            linear-gradient(135deg, #03070d 0%, #071522 42%, #09111c 100%);
        color: #edf6ff;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
    }

    .hero-shell {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1.25rem;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.1rem;
        border-radius: 28px;
        border: 1px solid rgba(161, 175, 194, 0.18);
        background: linear-gradient(135deg, rgba(5, 14, 24, 0.82), rgba(14, 26, 37, 0.88));
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.25);
    }

    .eyebrow {
        display: inline-block;
        background: linear-gradient(90deg, rgba(45, 212, 191, 0.16), rgba(59, 130, 246, 0.18));
        border: 1px solid rgba(45, 212, 191, 0.3);
        color: #acf3eb;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.48rem 0.8rem;
        border-radius: 999px;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        line-height: 1.02 !important;
        letter-spacing: -0.04em;
        color: #f8fbff !important;
        margin: 0 !important;
    }

    .hero-sub {
        margin-top: 0.75rem;
        color: rgba(211, 227, 243, 0.8);
        font-size: 0.98rem;
    }

    .hero-side {
        min-width: 220px;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        align-self: flex-start;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #bafde0;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 700;
    }

    .status-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 12px rgba(52, 211, 153, 0.8);
    }

    .mini-stat {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 18px;
        padding: 0.8rem 1rem;
    }

    .mini-label {
        color: #9ab1c6;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.65rem;
    }

    .mini-value {
        margin-top: 0.35rem;
        font-size: 1.6rem;
        font-weight: 800;
        color: #c7f9ff;
    }

    .glass-card {
        background: linear-gradient(180deg, rgba(17, 27, 39, 0.85), rgba(15, 22, 31, 0.88));
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        box-shadow: 0 18px 36px rgba(2, 6, 23, 0.35);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1.15rem 1.2rem;
    }

    h2, h3 {
        color: #ebf6ff !important;
        margin-top: 0 !important;
    }

    div[data-testid="stMetricContainer"] {
        background: linear-gradient(180deg, rgba(16, 30, 42, 0.9), rgba(12, 18, 26, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        box-shadow: 0 12px 24px rgba(5, 10, 20, 0.28);
        padding: 0.8rem 0.9rem;
    }

    div[data-testid="stMetricValue"] {
        color: #99ebff !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #bfd3e8 !important;
        font-size: 0.74rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    .sidebar-content {
        background: rgba(8, 15, 23, 0.92);
    }

    [data-testid="stSidebar"] {
        background: rgba(8, 15, 23, 0.9);
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }

    [data-testid="stSidebarNav"] {
        background: rgba(8, 15, 23, 0.92);
    }

    .stDataFrame {
        background: rgba(12, 21, 29, 0.7);
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.16);
    }

    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(90deg, #14b8a6, #3b82f6);
        color: white;
        border: none;
        font-weight: 700;
        box-shadow: 0 12px 26px rgba(59, 130, 246, 0.3);
    }

    .route-item {
        padding: 0.8rem 0.9rem;
        border-radius: 12px;
        background: rgba(15, 28, 38, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.12);
        margin-bottom: 0.65rem;
    }

    .route-badge {
        display: inline-block;
        padding: 0.28rem 0.5rem;
        border-radius: 999px;
        background: rgba(96, 165, 250, 0.14);
        border: 1px solid rgba(96, 165, 250, 0.24);
        color: #bfe3ff;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .map-legend {
        display: flex;
        gap: 1rem;
        align-items: center;
        flex-wrap: wrap;
        color: #a7bed2;
        font-size: 0.72rem;
        margin: -0.3rem 0 0.85rem;
    }

    .legend-item { display: inline-flex; align-items: center; gap: 0.35rem; }
    .legend-dot { width: 8px; height: 8px; border-radius: 50%; box-shadow: 0 0 10px currentColor; }

    .role-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }

    .role-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.25);
        color: #d9effd;
        padding: 0.45rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .quick-tip {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
        color: #dfeaf8;
        font-size: 0.82rem;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


user = login_form()
if user is None:
    st.stop()


def _route_segment_ids(graph: nx.MultiDiGraph, route_path: list[str]) -> set[str]:
    if not route_path or len(route_path) < 2:
        return set()
    route_nodes = set(str(node) for node in route_path)
    segment_ids = set()
    for source, target, key, data in graph.edges(keys=True, data=True):
        if str(source) in route_nodes and str(target) in route_nodes:
            segment_id = str(data.get("segment_id", key))
            if segment_id:
                segment_ids.add(segment_id)
    return segment_ids


def _status_color(value: float) -> list[int]:
    if value >= 0.8:
        return [255, 55, 82]
    if value >= 0.45:
        return [255, 190, 35]
    return [16, 255, 160]


@st.cache_data(show_spinner=False)
def load_data():
    try:
        graph = build_city_graph()
        nodes_df, network_df, traffic_df, incident_df = load_dataset_frames()
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        st.error(f"Dataset initialization failed: {exc}")
        st.stop()
    summary = get_network_summary(graph)
    origin, destination = select_route_nodes(graph)
    metrics = assign_edge_metrics(graph, demand_factor=1.1)
    incident_edges = synthetic_incident_edges(graph, metrics, severity=0.8)
    route_data = route_recommendation(graph, origin, destination)
    return graph, nodes_df, network_df, traffic_df, incident_df, summary, origin, destination, metrics, incident_edges, route_data


@st.cache_resource(show_spinner=False)
def load_spatial_trainer():
    trainer = SpatialTrafficTrainer()
    if trainer.model_path.exists():
        trainer.load_saved_model()
        trainer.train_historical_baseline()
        return trainer
    trainer.train_historical_baseline()
    trainer.train_ml_model()
    return trainer


def write_traffic_snapshot_once(metrics):
    if not st.session_state.get("traffic_snapshot_written"):
        ensure_traffic_database()
        save_road_status_snapshot(metrics)
        st.session_state["traffic_snapshot_written"] = True


graph, nodes_df, network_df, traffic_df, incidents_df, summary, origin, destination, metrics, incident_edges, route_data = load_data()
spatial_trainer = load_spatial_trainer()
write_traffic_snapshot_once(metrics)


def apply_role_visibility(user, graph_obj, network_frame, traffic_frame, incidents_frame, selected_origin, selected_destination):
    if has_role(user, ROLE_AUTHORITIES):
        return network_frame.copy(), traffic_frame.copy(), incidents_frame.copy(), "Full citywide operational view"
    if has_role(user, ROLE_RESEARCH):
        return network_frame.copy(), traffic_frame.copy(), incidents_frame.copy(), "Full analytical dataset view"
    if has_role(user, ROLE_SECURITY):
        return network_frame.copy(), traffic_frame.copy(), incidents_frame.copy(), "Complete road network status view"

    route_path = route_recommendation(graph_obj, str(selected_origin), str(selected_destination), "Passenger Vehicle")["best"]["path"]
    route_segment_ids = _route_segment_ids(graph_obj, route_path)
    if not route_segment_ids:
        route_segment_ids = set(network_frame["segment_id"].astype(str).head(15))
    visible_segments = sorted(route_segment_ids)
    visible_network = network_frame[network_frame["segment_id"].astype(str).isin(visible_segments)].copy()
    visible_traffic = traffic_frame[traffic_frame["segment_id"].astype(str).isin(visible_segments)].copy()
    visible_incidents = incidents_frame[incidents_frame["segment_id"].astype(str).isin(visible_segments)].copy()
    return visible_network, visible_traffic, visible_incidents, f"Corridor-focused route view ({len(visible_segments)} segments)"


time_options = sorted(traffic_df["timestamp"].drop_duplicates().tolist())
playback_options = time_options[::12] or time_options
selected_ts = st.select_slider("Playback timestamp", options=playback_options, value=playback_options[min(12, len(playback_options) - 1)])

visible_network_df, visible_traffic_df, visible_incidents_df, visible_scope_label = apply_role_visibility(
    user,
    graph,
    network_df,
    traffic_df,
    incidents_df,
    origin,
    destination,
)

alert_df = active_incidents_for_timestamp(visible_incidents_df, selected_ts)
window_df = visible_traffic_df[(visible_traffic_df["timestamp"] >= selected_ts - pd.Timedelta(hours=3)) & (visible_traffic_df["timestamp"] <= selected_ts)].copy()
segment_snapshot = (
    window_df.groupby("segment_id", as_index=False)
    .agg(
        avg_speed_kmh=("speed_kmh", "mean"),
        avg_flow_vph=("flow_vph", "mean"),
        congestion_index=("congestion_index", "mean"),
        delay_min=("delay_min", "mean"),
        travel_time_min=("travel_time_min", "mean"),
    )
)
segment_snapshot["segment_id"] = segment_snapshot["segment_id"].astype(str)

anomalies = detect_anomalies(visible_traffic_df, target_timestamp=selected_ts)
anomaly_count = len(anomalies)

route_data = route_recommendation(graph, origin, destination)
best = route_data["best"]
alt_routes = route_data["alternatives"]

role_headers = {
    ROLE_AUTHORITIES: (
        "Hyderabad Traffic Management Centre (TMC)",
        "Citywide traffic operations & incident control",
        "TMC operators, planners, and emergency dispatchers see the full operational network including live congestion, incident alerts, and capacity-management controls.",
    ),
    ROLE_FLEET: (
        "Commercial Fleet & Transit Operations",
        "Predictive routing for freight, transit, and ride-hailing",
        "Logistics, TSRTC, and ride-hailing teams get corridor-level routing guidance, ETA forecasts, and disruption-aware alternative paths for efficient dispatch.",
    ),
    ROLE_RESEARCH: (
        "Transportation Research & Development",
        "Spatial analytics and model diagnostics",
        "Data scientists, ML researchers, and technical reviewers can inspect residual anomalies, forecast validation, and raw telemetry for model evaluation.",
    ),
    ROLE_SECURITY: (
        "Traffic Monitoring & Security",
        "Live road-status monitoring & congestion watch",
        "Security teams monitor the complete road map and immediately identify congested corridors without route-level origin and destination disclosure.",
    ),
}
role_quick_actions = {
    ROLE_AUTHORITIES: ["Live congestion view", "Incident response", "Capacity planning"],
    ROLE_FLEET: ["Predictive ETA", "Route alternatives", "Disruption avoidance"],
    ROLE_RESEARCH: ["Residual analysis", "Forecast validation", "Telemetry audit"],
    ROLE_SECURITY: ["Congestion map", "Road watch", "Security visibility"],
}
role_eyebrow, role_title, role_subtitle = role_headers.get(
    user.role,
    ("Hyderabad mobility intelligence", "Traffic flow & incident intelligence", "Live urban mobility monitoring with predictive congestion and route guidance."),
)

visible_network_df, visible_traffic_df, visible_incidents_df, visible_scope_label = apply_role_visibility(
    user,
    graph,
    network_df,
    traffic_df,
    incidents_df,
    origin,
    destination,
)

quick_action_tags = "".join(
    f'<span class="role-tag">{tag}</span>' for tag in role_quick_actions.get(user.role, ["Traffic overview", "Live routing", "Smart dispatch"])
)

st.markdown(
    """
    <div class="hero-shell">
      <div>
        <div class="eyebrow">{role_eyebrow}</div>
        <h1 class="hero-title">{role_title}</h1>
        <div class="hero-sub">{role_subtitle}</div>
        <div class="hero-sub" style="margin-top:0.5rem; color:#a8f1d8; font-size:0.76rem; letter-spacing:0.06em; text-transform:uppercase;">{role_scope}</div>
        <div class="role-tags">{quick_actions}</div>
      </div>
      <div class="hero-side">
        <div class="status-pill"><span class="status-dot"></span>Live network</div>
        <div class="mini-stat">
          <div class="mini-label">Selected time</div>
          <div class="mini-value">{selected_ts}</div>
        </div>
      </div>
    </div>
    """.format(
        role_eyebrow=role_eyebrow,
        role_title=role_title,
        role_subtitle=role_subtitle,
        role_scope=visible_scope_label,
        quick_actions=quick_action_tags,
        selected_ts=selected_ts.strftime("%Y-%m-%d %H:%M"),
    ),
    unsafe_allow_html=True,
)

with st.sidebar:
    st.caption(f"Signed in as **{user.username}**")
    st.caption(user.role)
    if st.button("Sign out", width="stretch"):
        logout()
    st.markdown("---")
    st.header("Scenario Controls")
    if has_role(user, ROLE_AUTHORITIES):
        demand_factor = st.slider("Network demand", 0.6, 1.6, 1.1, 0.1)
        incident_severity = st.slider("Global incident severity", 0.3, 1.5, 0.8, 0.1)
        residual_threshold = st.slider("Active bottleneck residual", 1.5, 4.0, 2.5, 0.1)
        capacity_factor = st.slider("Network capacity", 0.6, 1.4, 1.0, 0.05)
    else:
        demand_factor, incident_severity, residual_threshold, capacity_factor = 1.1, 0.8, 2.5, 1.0
    if has_role(user, ROLE_FLEET):
        fleet_origin = st.selectbox("Current origin node", list(graph.nodes()), index=0)
        fleet_destination = st.selectbox("Destination node", list(graph.nodes()), index=min(1, len(graph.nodes()) - 1))
        vehicle_class = st.selectbox("Vehicle class", ["Passenger Vehicle", "Light Commercial EV", "Standard Bus", "Heavy Logistics Truck"])
    else:
        fleet_origin, fleet_destination, vehicle_class = origin, destination, "Passenger Vehicle"
    st.markdown("---")
    st.subheader("Data visibility")
    if has_role(user, ROLE_AUTHORITIES):
        st.markdown("- Full city graph and all segment metrics")
        st.markdown("- Live incident and congestion control panels")
        st.markdown("- Citywide capacity and emergency routing view")
    elif has_role(user, ROLE_FLEET):
        st.markdown("- Corridor-level route and ETA focus")
        st.markdown("- Forecast-based dispatch and congestion avoidance")
        st.markdown("- Only operationally relevant network slices")
    elif has_role(user, ROLE_RESEARCH):
        st.markdown("- Raw telemetry preview and residual diagnostics")
        st.markdown("- Forecast validation and model evaluation views")
        st.markdown("- Full analytical dataset access for research")
    elif has_role(user, ROLE_SECURITY):
        st.markdown("- Complete road map with congested and free corridors")
        st.markdown("- No route origin/destination details shown")
        st.markdown("- Focus on network-wide congestion and security monitoring")
    st.markdown("---")
    st.subheader("System Snapshot")
    st.markdown("- Live network: NEURAX smart-city dataset")
    st.markdown("- Model: segment-level congestion + incident logic")
    st.markdown("- Detection: residual anomaly scoring")
    st.markdown("- Routing: k-shortest path recommendations")

active_origin = fleet_origin if has_role(user, ROLE_FLEET) else origin
active_destination = fleet_destination if has_role(user, ROLE_FLEET) else destination
if has_role(user, ROLE_AUTHORITIES):
    try:
        metrics = assign_edge_metrics(graph, demand_factor=demand_factor, capacity_reduction=1.0 - capacity_factor)
        incident_edges = synthetic_incident_edges(graph, metrics, severity=incident_severity)
    except (KeyError, TypeError, ValueError):
        metrics, incident_edges = {}, []
else:
    metrics = assign_edge_metrics(graph, demand_factor=1.0, capacity_reduction=0.1)
    incident_edges = synthetic_incident_edges(graph, metrics, severity=0.7)
try:
    route_data = route_recommendation(graph, active_origin, active_destination, vehicle_class)
except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
    route_data = {"best": {"path": [active_origin], "route_time": 0.0, "avg_congestion": 0.0, "avg_speed": 0.0}, "alternatives": []}
best = route_data["best"]
alt_routes = route_data["alternatives"]

if has_role(user, ROLE_FLEET):
    route_segments = _route_segment_ids(graph, best["path"])
    fleet_metric_rows = []
    for (u, v, key), values in metrics.items():
        segment_id = str(graph[u][v][key].get("segment_id", key))
        if segment_id in route_segments:
            fleet_metric_rows.append({
                "segment_id": segment_id,
                "congestion_index": values.get("vc_ratio", values.get("congestion", 0.0)),
                "avg_speed_kmh": values.get("speed", 0.0),
            })
    visible_metric_df = pd.DataFrame(fleet_metric_rows)
    if visible_metric_df.empty:
        visible_metric_df = pd.DataFrame([{"segment_id": "route", "congestion_index": best["avg_congestion"], "avg_speed_kmh": best["avg_speed"]}])
elif has_role(user, ROLE_SECURITY):
    visible_metric_df = pd.DataFrame([
        {"segment_id": key[2], "congestion_index": values.get("vc_ratio", values.get("congestion", 0.0)), "avg_speed_kmh": values.get("speed", 0.0)}
        for key, values in metrics.items()
    ])
else:
    visible_metric_df = pd.DataFrame([
        {"segment_id": key[2], "congestion_index": values.get("vc_ratio", values.get("congestion", 0.0)), "avg_speed_kmh": values.get("speed", 0.0)}
        for key, values in metrics.items()
    ])

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Road nodes", f"{int(summary['nodes'])}")
    with col2:
        st.metric("Road edges", f"{int(summary['edges'])}")
    with col3:
        st.metric("Avg. speed", f"{summary['avg_speed']:.1f} km/h")
    with col4:
        gridlock_factor = sum(values.get("vc_ratio", 0.0) >= 0.8 for values in metrics.values()) / max(len(metrics), 1) * 100.0
        st.metric("Gridlock factor", f"{gridlock_factor:.1f}%")

if has_role(user, ROLE_SECURITY):
    st.markdown('<div class="glass-card"><h3>Road status overview</h3>', unsafe_allow_html=True)
    security_rows = []
    for key, values in metrics.items():
        congestion = values.get("vc_ratio", values.get("congestion", 0.0))
        if congestion >= 0.8:
            status = "Congested"
        elif congestion >= 0.45:
            status = "Moderate"
        else:
            status = "Free"
        security_rows.append({
            "Segment ID": key[2],
            "Status": status,
            "Congestion": round(float(congestion), 2),
            "Speed km/h": round(float(values.get("speed", 0.0)), 1),
        })
    st.dataframe(pd.DataFrame(security_rows).sort_values("Congestion", ascending=False).head(25), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    col_a, col_b = st.columns([1.25, 1.5])
    with col_a:
        st.markdown('<div class="glass-card"><h3>Selected corridor</h3>', unsafe_allow_html=True)
        st.markdown(f"<div class='route-item'><div class='route-badge'>Origin</div><div style='margin-top:0.5rem; font-weight:600; color:#eaf7ff;'>{active_origin}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='route-item'><div class='route-badge'>Destination</div><div style='margin-top:0.5rem; font-weight:600; color:#eaf7ff;'>{active_destination}</div></div>", unsafe_allow_html=True)
        st.write(f"Best route time: **{best['route_time']:.2f}**")
        st.write(f"Average congestion: **{best['avg_congestion']:.2f}**")
        st.write(f"Average speed: **{best['avg_speed']:.1f} km/h**")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-card"><h3>Alternate routes</h3>', unsafe_allow_html=True)
        route_options = []
        for idx, item in enumerate(alt_routes[:3], 1):
            route = item.get("route", item.get("path", []))
            metrics_val = item.get("metrics", item)
            route_options.append(route)
            st.markdown(
                f"""
                <div class="route-item">
                  <div class="route-badge">Route {idx}</div>
                  <div style="margin-top:0.5rem; color:#eaf7ff; font-weight:600;">{len(route)} nodes • {metrics_val['route_time']:.2f} travel time • {metrics_val['avg_speed']:.1f} km/h</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

if has_role(user, ROLE_SECURITY):
    route_layers = []
    highlighted_path = []
else:
    selected_route = st.selectbox("Highlight route", options=["Best"] + [f"Alt {i}" for i in range(1, min(len(alt_routes), 3) + 1)], index=0)

    if selected_route == "Best":
        highlighted_path = best["path"]
    else:
        alt_index = int(selected_route.split()[-1]) - 1
        highlighted_path = alt_routes[alt_index].get("route", alt_routes[alt_index].get("path", [])) if alt_index < len(alt_routes) else best["path"]

    node_lookup = nodes_df.set_index("node_id")
    route_points = []
    for node in highlighted_path:
        if str(node) in node_lookup.index:
            row = node_lookup.loc[str(node)]
            route_points.append([float(row["lon"]), float(row["lat"])])

    route_layers = build_route_layers(route_points)

map_stats = visible_metric_df if not visible_metric_df.empty else pd.DataFrame([{"segment_id": "", "congestion_index": 0.0, "avg_speed_kmh": 0.0}])
road_layer = build_segment_map_layer(visible_network_df, nodes_df, map_stats if not map_stats.empty else segment_snapshot)
incident_layer = build_incident_pin_layer(visible_incidents_df, visible_network_df, nodes_df, selected_ts)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Network Overview & Routed Corridors")
st.markdown(
    '<div class="map-legend">'
    '<span class="legend-item"><span class="legend-dot" style="color:#10ffa0;background:#10ffa0"></span>Free flow</span>'
    '<span class="legend-item"><span class="legend-dot" style="color:#ffbe23;background:#ffbe23"></span>Moderate delay</span>'
    '<span class="legend-item"><span class="legend-dot" style="color:#ff3752;background:#ff3752"></span>Gridlock / incident</span>'
    '<span class="legend-item"><span class="legend-dot" style="color:#26d3ff;background:#26d3ff"></span>Recommended route</span>'
    '</div>',
    unsafe_allow_html=True,
)
view = deck_view_state(latitude=17.385, longitude=78.4867, zoom=11.2)
map_deck = pdk.Deck(
    map_provider="carto",
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    initial_view_state=view,
    layers=[road_layer, *incident_layer, *route_layers],
    tooltip={"html": "<b>Segment:</b> {segment_id}<br><b>Congestion:</b> {congestion_index}", "style": {"color": "white"}},
)
st.pydeck_chart(map_deck, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

trend_df = build_trend_series(visible_traffic_df, selected_ts, window_hours=3)
risk_df = build_risk_series(visible_traffic_df, selected_ts, window_hours=3)

left, right = st.columns(2)
with left:
    st.markdown('<div class="glass-card"><h3>Congestion trend</h3>', unsafe_allow_html=True)
    st.line_chart(trend_df.set_index("time")["congestion_index"], color="#5eead4")
    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="glass-card"><h3>Incident risk</h3>', unsafe_allow_html=True)
    st.line_chart(risk_df.set_index("time")["incident_risk"], color="#60a5fa")
    st.markdown('</div>', unsafe_allow_html=True)

forecast_ids = list(visible_network_df["segment_id"].astype(str).unique()[:8])
forecast_df = segment_forecast_window(visible_traffic_df, selected_ts, forecast_ids, horizons_minutes=(15, 30, 45, 60))

if has_role(user, ROLE_AUTHORITIES):
    st.markdown('<div class="glass-card"><h3>TMC Command Panel</h3><p style="color:#9ab1c6">City-wide congestion, capacity, and incident impact controls.</p>', unsafe_allow_html=True)
    authority_rows = [
        {
            "Segment ID": key[2],
            "Volume vph": round(values.get("volume", 0.0), 1),
            "Capacity vph": round(values.get("capacity", 0.0), 1),
            "Congestion": round(values.get("congestion", 0.0), 2),
        }
        for key, values in metrics.items()
    ]
    if authority_rows:
        st.dataframe(pd.DataFrame(authority_rows).sort_values("Congestion", ascending=False).head(25), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><h3>Authority controls</h3>', unsafe_allow_html=True)
    authority_kpi_left, authority_kpi_mid, authority_kpi_right = st.columns(3)
    authority_kpi_left.metric("City gridlock factor", f"{gridlock_factor:.1f}%")
    authority_kpi_mid.metric("Mean modeled speed", f"{sum(values.get('speed', 0.0) for values in metrics.values()) / max(len(metrics), 1):.1f} km/h")
    authority_kpi_right.metric("Active alerts", len(incident_edges))
    st.write(f"Capacity multiplier: **{capacity_factor:.2f}**")
    st.write(f"Residual alert threshold: **{residual_threshold:.1f} sigma**")
    st.write(f"Network demand scenario: **{demand_factor:.2f}x**")
    st.markdown('</div>', unsafe_allow_html=True)

if has_role(user, ROLE_SECURITY):
    st.markdown('<div class="glass-card"><h3>Security monitoring overview</h3>', unsafe_allow_html=True)
    congested_roads = []
    for key, values in metrics.items():
        congestion = float(values.get("vc_ratio", values.get("congestion", 0.0)))
        if congestion >= 0.45:
            congested_roads.append({
                "Segment ID": key[2],
                "Road status": "Congested" if congestion >= 0.8 else "Moderate",
                "Congestion index": round(congestion, 2),
                "Speed": round(float(values.get("speed", 0.0)), 1),
            })
    if congested_roads:
        st.dataframe(pd.DataFrame(congested_roads).sort_values("Congestion index", ascending=False).head(20), width="stretch")
    else:
        st.info("No congested roads detected in the current network snapshot.")
    st.markdown('</div>', unsafe_allow_html=True)

if has_role(user, ROLE_FLEET):
    st.markdown('<div class="glass-card"><h3>Fleet ETA matrix</h3>', unsafe_allow_html=True)
    st.write(f"Vehicle class: **{vehicle_class}**")
    st.dataframe(pd.DataFrame([{
        "Origin": active_origin,
        "Destination": active_destination,
        "Vehicle": vehicle_class,
        "ETA minutes": round(best["route_time"], 2),
        "Route speed km/h": round(best["avg_speed"], 1),
    }]), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

if has_role(user, ROLE_RESEARCH):
    residuals = anomaly_tracker(visible_traffic_df, selected_ts, z_threshold=2.5)
    trained_forecasts = spatial_trainer.forecast_table(forecast_ids, selected_ts)
    st.markdown('<div class="glass-card"><h3>Research and model diagnostics</h3>', unsafe_allow_html=True)
    model_col, chart_col = st.columns([1, 1.6])
    with model_col:
        learning_rate = st.number_input("Learning rate baseline", min_value=0.0001, max_value=0.1, value=0.0012, step=0.0001, format="%.4f")
        epoch_count = st.slider("Target epoch iterations", 10, 500, 150)
        dilation_channels = st.selectbox("Spatial dilation channels", [16, 32, 64])
        st.caption(f"Configured baseline: lr={learning_rate:.4f}, epochs={epoch_count}, channels={dilation_channels}")
    with chart_col:
        st.markdown("**Forecast validation profile**")
        if forecast_df.empty:
            st.info("Forecast validation data is unavailable for this timestamp.")
        else:
            validation_profile = forecast_df.groupby("horizon_minutes", as_index=True)["predicted_speed_kmh"].mean()
            st.line_chart(validation_profile, color="#5eead4")
    r1, r2, r3 = st.columns(3)
    r1.metric("Residual anomalies", len(residuals))
    r2.metric("Model state", "Trained" if spatial_trainer.is_trained else "Fallback")
    r3.metric("Telemetry rows", f"{len(traffic_df):,}")
    st.caption(f"Historical lookup entries: {len(spatial_trainer.historical_lookup):,} | Training rows: {spatial_trainer.training_rows:,}")
    if not trained_forecasts.empty:
        st.dataframe(trained_forecasts.head(32), width="stretch")
    if not residuals.empty:
        st.dataframe(residuals.head(25), width="stretch")
    st.dataframe(visible_traffic_df.head(25), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card"><h3>Traffic Summary</h3>', unsafe_allow_html=True)
summary_df = [
    {"Metric": "Total Nodes", "Value": int(summary['nodes'])},
    {"Metric": "Total Edges", "Value": int(summary['edges'])},
    {"Metric": "Average Speed", "Value": round(summary['avg_speed'], 1)},
    {"Metric": "Incident Count", "Value": len(alert_df)},
    {"Metric": "Anomaly Count", "Value": anomaly_count},
    {"Metric": "Best Route Time", "Value": round(best['route_time'], 2)},
    {"Metric": "Avg Congestion", "Value": round(best['avg_congestion'], 2)},
]
st.dataframe(summary_df, width="stretch")
if not forecast_df.empty:
    st.dataframe(forecast_df.head(12), width="stretch")

if has_role(user, ROLE_AUTHORITIES):
    st.markdown(
        """
        <div class="quick-tip">
        <strong>Operational tip:</strong> Watch for the junctions with sustained congestion spikes. Use the corridor controls to raise incident severity and compare how the network absorbs pressure before dispatching field response teams.
        </div>
        """,
        unsafe_allow_html=True,
    )
elif has_role(user, ROLE_FLEET):
    st.markdown(
        """
        <div class="quick-tip">
        <strong>Fleet tip:</strong> Prefer the route with lower forecasted delay and higher projected average speed for dispatch windows. This helps avoid spillback hotspots and reduces deadhead time.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="quick-tip">
        <strong>Research tip:</strong> Compare the historical residual footprint against the latest forecast profile to inspect whether the anomaly engine is responding to real bottleneck events or purely seasonal drift.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Prototype status: powered by the NEURAX smart-cities training data and segment-level traffic analysis.")
