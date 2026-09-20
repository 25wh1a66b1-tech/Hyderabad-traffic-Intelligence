from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.d04_forecast_traffic import SpatialTrafficTrainer


def load_training_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Training file is empty: {path}")
    return frame


def main() -> None:
    csv_path = Path(__file__).resolve().parent / "data" / "traffic_training_dataset.csv"
    frame = load_training_csv(csv_path)

    trainer = SpatialTrafficTrainer()
    successful = trainer.retrain_on_dataframe(frame)

    if not successful:
        raise RuntimeError("Model training failed. Check the dataset columns and format.")

    print("MODEL_TRAINING_SUCCESS")
    print(f"Rows used for training: {trainer.training_rows}")
    print(f"Segments learned: {trainer.dataset_summary.get('segments', 0)}")
    print(f"Average speed baseline: {trainer.dataset_summary.get('average_speed', 0)} km/h")

    sample = list(trainer.historical_lookup.items())[:5]
    print("SAMPLE_LEARNED_PATTERNS")
    for key, value in sample:
        print(key, round(value, 2))


if __name__ == "__main__":
    main()
