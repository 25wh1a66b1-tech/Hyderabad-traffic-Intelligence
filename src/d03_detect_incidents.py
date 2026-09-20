from __future__ import annotations

from typing import Optional

import pandas as pd


def detect_anomalies(
    traffic_df: pd.DataFrame,
    target_timestamp: Optional[pd.Timestamp] = None,
    zscore_threshold: float = 2.5,
) -> pd.DataFrame:
    """Flag road segments whose current speed is unusually low versus the historical distribution for the same hour-of-week."""
    if traffic_df.empty:
        return pd.DataFrame(columns=["segment_id", "timestamp", "speed_kmh", "mean_speed_kmh", "std_speed_kmh", "z_score"])

    df = traffic_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if target_timestamp is None:
        target_timestamp = df["timestamp"].max()
    target_timestamp = pd.Timestamp(target_timestamp)

    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_day"] = df["timestamp"].dt.hour

    base = df[
        (df["timestamp"] <= target_timestamp)
        & (df["day_of_week"] == target_timestamp.dayofweek)
        & (df["hour_of_day"] == target_timestamp.hour)
    ].copy()
    if base.empty:
        base = df[df["segment_id"].isin(df["segment_id"].unique()[:5])].copy()

    hist = base.groupby(["segment_id"], as_index=False)["speed_kmh"].agg(["mean", "std"]).reset_index()
    hist = hist.rename(columns={"mean": "mean_speed_kmh", "std": "std_speed_kmh"})

    current = df[df["timestamp"] == target_timestamp].copy()
    if current.empty:
        current = df[df["timestamp"] == df["timestamp"].max()].copy()

    current = current[["segment_id", "timestamp", "speed_kmh"]].copy()
    merged = current.merge(hist, on="segment_id", how="left")
    merged["std_speed_kmh"] = merged["std_speed_kmh"].fillna(1.0)
    merged["z_score"] = (merged["mean_speed_kmh"] - merged["speed_kmh"]) / merged["std_speed_kmh"].replace(0, 1.0)
    merged["anomaly"] = merged["z_score"] >= zscore_threshold
    return merged[merged["anomaly"]].sort_values("z_score", ascending=False)


def active_incidents_for_timestamp(incidents_df: pd.DataFrame, target_timestamp: pd.Timestamp) -> pd.DataFrame:
    """Return incidents that are active at the selected timestamp."""
    if incidents_df.empty:
        return incidents_df.copy()

    df = incidents_df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    return df[(df["start_time"] <= target_timestamp) & (df["end_time"] >= target_timestamp)].copy()
