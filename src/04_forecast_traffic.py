from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd


def _segment_history(traffic_df: pd.DataFrame, segment_id: str) -> pd.DataFrame:
    seg = traffic_df[traffic_df["segment_id"] == segment_id].sort_values("timestamp").copy()
    return seg


def rolling_forecast(
    traffic_df: pd.DataFrame,
    segment_id: str,
    target_timestamp: pd.Timestamp,
    horizons_minutes: Iterable[int] = (15, 30, 60),
) -> Dict[int, float]:
    """A lightweight rolling baseline forecast based on the most recent change in speed."""
    seg = _segment_history(traffic_df, segment_id)
    seg = seg[seg["timestamp"] <= pd.Timestamp(target_timestamp)].tail(5)
    if seg.empty:
        return {h: 0.0 for h in horizons_minutes}

    if len(seg) >= 2:
        slope = (seg["speed_kmh"].iloc[-1] - seg["speed_kmh"].iloc[-2]) / max((seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[-2]).total_seconds() / 60.0, 1.0)
    else:
        slope = 0.0

    current = float(seg["speed_kmh"].iloc[-1])
    result: Dict[int, float] = {}
    for horizon in horizons_minutes:
        predicted = max(current + slope * horizon, 0.0)
        result[int(horizon)] = round(float(predicted), 2)
    return result


def segment_forecast_window(
    traffic_df: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    segment_ids: List[str],
    horizons_minutes: Iterable[int] = (15, 30, 60),
) -> pd.DataFrame:
    rows = []
    for segment_id in segment_ids:
        forecast = rolling_forecast(traffic_df, segment_id, target_timestamp, horizons_minutes)
        for horizon, value in forecast.items():
            rows.append({
                "segment_id": segment_id,
                "horizon_minutes": horizon,
                "predicted_speed_kmh": value,
            })
    return pd.DataFrame(rows)
    """Compatibility entry point for the importable forecast engine."""
    from importlib import import_module

    _engine = import_module("src.d04_forecast_traffic")
    SpatialTrafficTrainer = _engine.SpatialTrafficTrainer
    historical_average_forecast = _engine.historical_average_forecast
    rolling_forecast = _engine.rolling_forecast
    segment_forecast_window = _engine.segment_forecast_window
    anomaly_tracker = _engine.anomaly_tracker
