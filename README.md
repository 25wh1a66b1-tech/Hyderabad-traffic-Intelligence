# Hyderabad Traffic Flow & Incident Intelligence

A prototype for modeling urban traffic conditions in Hyderabad, with a focus on congestion analysis, incident detection, short-horizon forecasting, and route recommendation.

## Problem

Hyderabad faces recurring congestion on arterial roads, flyovers, and junction-heavy corridors. We need a system that can:

- detect abnormal traffic conditions,
- estimate where incidents are likely occurring,
- forecast congestion over the next 15–60 minutes,
- recommend better routes before congestion spreads,
- present results in a clear dashboard.

## Approach

We build a Hyderabad-like road network from OpenStreetMap using OSMnx, model traffic on a directed graph, and simulate congestion using a BPR-style delay model adapted for mixed urban traffic conditions.

The pipeline includes:

1. Road network creation and cleanup
2. Demand and capacity modeling
3. Congestion simulation with spillback behavior
4. Incident detection using residual and spatial anomaly logic
5. Forecasting with a graph-based model such as Graph WaveNet
6. Predictive route recommendation using k-shortest paths
7. Visualization in a Streamlit + PyDeck dashboard

## Key Design Choices

- OSMnx for road graph construction
- BPR-based travel-time estimation adjusted for mixed-traffic urban conditions
- Flyovers and service roads treated as separate graph edges
- Junction consolidation to avoid artificial short links
- Residual-based anomaly detection inspired by Standard Normal Deviate and California-style logic
- Predictive routing based on forecasted travel time rather than current conditions

## Why this works

This approach keeps the project explainable, fast to build, and realistic enough for a hackathon/demo setting. It avoids unnecessary complexity while still using established transportation-engineering and ML concepts.

## Expected Output

The system will generate:

- congestion scores by road segment,
- incident alerts and affected zones,
- short-term traffic forecasts,
- alternative route suggestions,
- a live map-based dashboard.

## Core Stack

- Python
- OSMnx
- NetworkX
- Pandas / NumPy
- Streamlit
- PyDeck
- Graph-based forecasting model (Graph WaveNet or compact GNN baseline)

## Project Structure

```text
hyderabad-traffic-intelligence/
├── README.md
├── requirements.txt
├── src/
│   ├── 01_build_graph.py
│   ├── 02_generate_data.py
│   ├── 03_detect_incidents.py
│   ├── 04_forecast_traffic.py
│   ├── 05_route_recommendation.py
│   └── 06_dashboard.py
├── data/
├── notebooks/
├── models/
└── tests/
```

## Summary

This project is a practical Hyderabad-focused traffic intelligence prototype that combines network modeling, congestion simulation, anomaly detection, forecasting, and route recommendation into a single explainable system.
