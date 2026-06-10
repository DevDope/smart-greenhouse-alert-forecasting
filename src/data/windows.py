"""Build supervised frames aligned by observation end timestamp."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_supervised_frame(
    feature_frame: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
    task_config: dict[str, Any],
    data_config: dict[str, Any],
    feature_metadata: dict[str, int],
) -> pd.DataFrame:
    timestamp_column = data_config["timestamp_column"]
    frequency = pd.Timedelta(data_config["frequency"])
    window_size_steps = max(task_config.get("window_size_steps", 1), feature_metadata["window_size_steps"])

    merged = feature_frame.merge(labels, on=timestamp_column, how="inner")
    merged["observation_end"] = pd.to_datetime(merged[timestamp_column])
    merged["observation_start"] = merged["observation_end"] - ((window_size_steps - 1) * frequency)

    required_columns = [*feature_columns, task_config["label_column"], "label_end"]
    supervised = merged.dropna(subset=required_columns).reset_index(drop=True)
    return supervised
