from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_loader import load_dataset_frames


class SpatialTrafficTrainer:
    """Traffic speed model trained from cleaned historical data and reusable for unseen test inputs."""

    def __init__(self):
        self.historical_lookup: dict[tuple[str, int, int], float] = {}
        self.segment_standard_deviations: dict[str, float] = {}
        self.is_trained = False
        self.training_rows = 0
        self.dataset_summary: dict[str, float | int | str] = {}
        self.model = None
        self.training_metrics: dict[str, float] = {}
        self.model_path = Path(__file__).resolve().parent.parent / "models" / "traffic_speed_model.joblib"

    @staticmethod
    def _prepare_training_frame(traffic_df: pd.DataFrame) -> pd.DataFrame:
        if traffic_df is None or traffic_df.empty:
            return pd.DataFrame()

        frame = traffic_df.copy()
        frame = frame.loc[:, [col for col in [
            "segment_id", "timestamp", "speed_kmh", "avg_speed_kmh", "flow_vph", "capacity_vph",
            "occupancy_pct", "congestion_index", "weather_condition", "direction", "road_name",
            "route_status", "incident_flag", "day_of_week", "hour_slot"
        ] if col in frame.columns]]
        if "segment_id" not in frame.columns or "timestamp" not in frame.columns:
            raise ValueError("Training dataset must include segment_id and timestamp columns.")

        if "speed_kmh" not in frame.columns:
            if "avg_speed_kmh" in frame.columns:
                frame = frame.rename(columns={"avg_speed_kmh": "speed_kmh"})
            elif "congestion_index" in frame.columns:
                frame["speed_kmh"] = (1.0 - pd.to_numeric(frame["congestion_index"], errors="coerce")) * 80.0
            else:
                raise ValueError("Training dataset must include speed_kmh or a usable substitute.")

        frame["segment_id"] = frame["segment_id"].astype(str).str.strip()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["speed_kmh"] = pd.to_numeric(frame["speed_kmh"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "segment_id", "speed_kmh"])

        if frame.empty:
            return frame

        frame = frame.sort_values("timestamp")
        frame = frame.drop_duplicates(subset=["segment_id", "timestamp"], keep="last")
        if "day_of_week" not in frame.columns:
            frame["day_of_week"] = frame["timestamp"].dt.dayofweek
        if "hour_slot" in frame.columns:
            frame["hour"] = pd.to_numeric(frame["hour_slot"], errors="coerce")
        else:
            frame["hour"] = frame["timestamp"].dt.hour
        if "weather_condition" not in frame.columns:
            frame["weather_condition"] = "Clear"
        if "direction" not in frame.columns:
            frame["direction"] = "Inbound"
        if "route_status" not in frame.columns:
            frame["route_status"] = "Moderate"
        if "incident_flag" not in frame.columns:
            frame["incident_flag"] = 0
        if "flow_vph" in frame.columns:
            frame["flow_vph"] = pd.to_numeric(frame["flow_vph"], errors="coerce").fillna(0)
        if "capacity_vph" in frame.columns:
            frame["capacity_vph"] = pd.to_numeric(frame["capacity_vph"], errors="coerce").fillna(0)
        if "occupancy_pct" in frame.columns:
            frame["occupancy_pct"] = pd.to_numeric(frame["occupancy_pct"], errors="coerce").fillna(0)
        if "congestion_index" in frame.columns:
            frame["congestion_index"] = pd.to_numeric(frame["congestion_index"], errors="coerce").fillna(0.0)
        frame["incident_flag"] = pd.to_numeric(frame["incident_flag"], errors="coerce").fillna(0).astype(int)
        return frame

    def train_historical_baseline(self, traffic_df: pd.DataFrame | None = None) -> bool:
        try:
            if traffic_df is None:
                _, _, traffic_df, _ = load_dataset_frames()
            frame = self._prepare_training_frame(traffic_df)
            if frame.empty:
                self.is_trained = False
                self.training_rows = 0
                return False

            grouped = frame.groupby(["segment_id", "day_of_week", "hour"])["speed_kmh"].agg(["mean", "std"])
            self.historical_lookup = {
                (str(segment_id), int(day), int(hour)): float(values["mean"])
                for (segment_id, day, hour), values in grouped.iterrows()
            }
            standard_deviations = frame.groupby("segment_id")["speed_kmh"].std().fillna(4.0)
            self.segment_standard_deviations = {str(key): max(float(value), 0.1) for key, value in standard_deviations.items()}
            self.training_rows = int(len(frame))
            self.dataset_summary = {
                "segments": int(frame["segment_id"].nunique()),
                "time_slots": int(frame.shape[0]),
                "average_speed": round(float(frame["speed_kmh"].mean()), 2),
            }
            self.is_trained = bool(self.historical_lookup)
            return self.is_trained
        except (KeyError, TypeError, ValueError, OSError):
            self.is_trained = False
            self.training_rows = 0
            return False

    def train_ml_model(self, traffic_df: pd.DataFrame | None = None) -> bool:
        try:
            if traffic_df is None:
                _, _, traffic_df, _ = load_dataset_frames()
            frame = self._prepare_training_frame(traffic_df)
            if frame.empty:
                self.model = None
                return False

            used_columns = [
                "segment_id", "hour", "day_of_week", "weather_condition",
                "direction", "flow_vph", "capacity_vph", "occupancy_pct",
                "congestion_index", "incident_flag", "route_status"
            ]
            X = frame[[col for col in used_columns if col in frame.columns]].copy()
            y = frame["speed_kmh"].astype(float).copy()

            numeric_features = [
                col for col in ["hour", "day_of_week", "flow_vph", "capacity_vph", "occupancy_pct", "congestion_index", "incident_flag"]
                if col in X.columns
            ]
            categorical_features = [
                col for col in ["segment_id", "weather_condition", "direction", "route_status"]
                if col in X.columns
            ]

            X[numeric_features] = X[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            for col in categorical_features:
                X[col] = X[col].fillna("unknown").astype(str)

            if X.empty or y.empty:
                self.model = None
                return False

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            preprocessor = ColumnTransformer([
                ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), categorical_features),
            ])
            regressor = RandomForestRegressor(
                n_estimators=500,
                max_features="sqrt",
                random_state=42,
                min_samples_leaf=2,
                n_jobs=-1,
            )
            pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)
            self.model = pipeline
            self.training_metrics = {
                "mae_kmh": float(mean_absolute_error(y_test, preds)),
                "r2_score": float(r2_score(y_test, preds)),
            }
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, self.model_path)
            self.is_trained = True
            return True
        except (KeyError, TypeError, ValueError, OSError):
            self.model = None
            self.is_trained = False
            return False

    def load_saved_model(self) -> bool:
        """Load a previously trained artifact from disk without redoing expensive training."""
        if not self.model_path.exists():
            return False
        try:
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            self.training_metrics = {"mae_kmh": 0.0, "r2_score": 0.0}
            return True
        except Exception:
            self.model = None
            self.is_trained = False
            return False

    def retrain_on_dataframe(self, traffic_df: pd.DataFrame) -> bool:
        """Retrain both the historical baseline and the ML regressor on deduplicated data."""
        ok = self.train_historical_baseline(traffic_df)
        ml_ok = self.train_ml_model(traffic_df)
        return ok or ml_ok

    def forecast_horizon(self, segment_id: str, current_time: pd.Timestamp, horizon_minutes: int) -> float:
        future_time = pd.Timestamp(current_time) + pd.Timedelta(minutes=int(horizon_minutes))
        key = (str(segment_id), future_time.dayofweek, future_time.hour)
        if self.model is not None:
            try:
                feature_row = {
                    "segment_id": str(segment_id),
                    "hour": int(future_time.hour),
                    "day_of_week": int(future_time.dayofweek),
                    "weather_condition": "Clear",
                    "direction": "Inbound",
                    "flow_vph": 900,
                    "capacity_vph": 1500,
                    "occupancy_pct": 60,
                    "congestion_index": 0.6,
                    "incident_flag": 0,
                    "route_status": "Moderate",
                }
                prediction = float(self.model.predict(pd.DataFrame([feature_row]))[0])
                return round(max(prediction, 0.0), 2)
            except Exception:
                pass
        if not self.is_trained:
            return 35.0
        return round(float(self.historical_lookup.get(key, 35.0)), 2)

    def forecast_table(self, segment_ids: Iterable[str], current_time: pd.Timestamp, horizons: Iterable[int] = (15, 30, 45, 60)) -> pd.DataFrame:
        rows = []
        for segment_id in segment_ids:
            for horizon in horizons:
                rows.append({"segment_id": str(segment_id), "horizon_minutes": int(horizon), "predicted_speed_kmh": self.forecast_horizon(str(segment_id), current_time, int(horizon))})
        return pd.DataFrame(rows)


def _segment_history(traffic_df: pd.DataFrame, segment_id: str) -> pd.DataFrame:
    seg = traffic_df[traffic_df["segment_id"].astype(str) == str(segment_id)].sort_values("timestamp").copy()
    seg["timestamp"] = pd.to_datetime(seg["timestamp"], errors="coerce")
    return seg


def historical_average_forecast(
    traffic_df: pd.DataFrame,
    segment_id: str,
    target_timestamp: pd.Timestamp,
    horizons_minutes: Iterable[int] = (15, 30, 45, 60),
) -> Dict[int, float]:
    """Forecast speed from the segment's historical hour-of-week profile."""
    frame = _segment_history(traffic_df, segment_id)
    if frame.empty:
        return {int(horizon): 0.0 for horizon in horizons_minutes}
    target = pd.Timestamp(target_timestamp)
    frame["hour_of_week"] = frame["timestamp"].dt.dayofweek * 24 + frame["timestamp"].dt.hour
    target_hour = target.dayofweek * 24 + target.hour
    historical = frame[frame["hour_of_week"] == target_hour]["speed_kmh"].dropna()
    baseline = float(historical.mean()) if not historical.empty else float(frame["speed_kmh"].tail(24).mean())
    return {int(horizon): round(max(baseline, 0.0), 2) for horizon in horizons_minutes}


def rolling_forecast(
    traffic_df: pd.DataFrame,
    segment_id: str,
    target_timestamp: pd.Timestamp,
    horizons_minutes: Iterable[int] = (15, 30, 45, 60),
) -> Dict[int, float]:
    """Blend an autoregressive rolling baseline with the historical hour-of-week mean."""
    seg = _segment_history(traffic_df, segment_id)
    seg = seg[seg["timestamp"] <= pd.Timestamp(target_timestamp)].tail(5)
    if seg.empty:
        return {int(h): 0.0 for h in horizons_minutes}

    if len(seg) >= 2:
        slope = (seg["speed_kmh"].iloc[-1] - seg["speed_kmh"].iloc[-2]) / max((seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[-2]).total_seconds() / 60.0, 1.0)
    else:
        slope = 0.0

    current = float(seg["speed_kmh"].iloc[-1])
    historical = historical_average_forecast(traffic_df, segment_id, target_timestamp, horizons_minutes)
    result: Dict[int, float] = {}
    for horizon in horizons_minutes:
        rolling = max(current + slope * horizon, 0.0)
        predicted = 0.65 * rolling + 0.35 * historical[int(horizon)]
        result[int(horizon)] = round(float(predicted), 2)
    return result


def anomaly_tracker(
    traffic_df: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    z_threshold: float = 2.5,
) -> pd.DataFrame:
    """Flag speeds at least z_threshold standard deviations below segment history."""
    frame = traffic_df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["segment_id"] = frame["segment_id"].astype(str)
    current = frame[frame["timestamp"] == pd.Timestamp(target_timestamp)].copy()
    if current.empty:
        return pd.DataFrame(columns=["segment_id", "speed_kmh", "historical_mean", "historical_std", "residual_z", "is_anomaly"])
    profile = frame.groupby("segment_id")["speed_kmh"].agg(historical_mean="mean", historical_std="std").reset_index()
    merged = current.merge(profile, on="segment_id", how="left")
    merged["historical_std"] = merged["historical_std"].fillna(0.0).replace(0.0, 1e-6)
    merged["residual_z"] = (merged["speed_kmh"] - merged["historical_mean"]) / merged["historical_std"]
    merged["is_anomaly"] = merged["residual_z"] <= -abs(float(z_threshold))
    return merged[merged["is_anomaly"]].sort_values("residual_z").reset_index(drop=True)


def segment_forecast_window(
    traffic_df: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    segment_ids: List[str],
    horizons_minutes: Iterable[int] = (15, 30, 45, 60),
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
