"""Compatibility entry point for the production dashboard layer engine."""
from importlib import import_module

_engine = import_module("src.d06_dashboard")
congestion_color = _engine.congestion_color
build_segment_map_layer = _engine.build_segment_map_layer
build_incident_pin_layer = _engine.build_incident_pin_layer
build_route_layers = _engine.build_route_layers
build_trend_series = _engine.build_trend_series
build_risk_series = _engine.build_risk_series
deck_view_state = _engine.deck_view_state
