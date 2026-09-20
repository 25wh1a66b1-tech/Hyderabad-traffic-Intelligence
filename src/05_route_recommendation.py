"""Compatibility entry point for the production routing engine."""
from importlib import import_module

_engine = import_module("src.traffic_model")
get_edge_cost = _engine.get_edge_cost
candidate_routes = _engine.candidate_routes
compute_route_metrics = _engine.compute_route_metrics
route_recommendation = _engine.route_recommendation
