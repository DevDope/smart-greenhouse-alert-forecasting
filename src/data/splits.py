"""Chronological dataset splitting with embargo."""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_embargo_steps(task_config: dict[str, Any], feature_metadata: dict[str, int], data_config: dict[str, Any]) -> int:
    frequency = pd.Timedelta(data_config["frequency"])
    horizon_steps = int(pd.Timedelta(minutes=task_config["horizon_minutes"]) / frequency)
    return max(
        task_config.get("window_size_steps", 1),
        horizon_steps,
        feature_metadata.get("max_rolling_lookback_steps", 1),
        feature_metadata.get("window_size_steps", 1),
    )


def apply_chronological_split(
    supervised: pd.DataFrame,
    split_config: dict[str, Any],
    embargo_steps: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if supervised.empty:
        raise ValueError("Supervised frame is empty after feature and label alignment.")

    split = supervised.copy().reset_index(drop=True)
    n_rows = len(split)
    usable_rows = n_rows - (2 * embargo_steps)
    if usable_rows < 9:
        raise ValueError(
            f"Not enough rows ({n_rows}) after embargo ({embargo_steps}) to create train/val/test splits."
        )

    train_ratio = float(split_config["train_ratio"])
    val_ratio = float(split_config["val_ratio"])
    test_ratio = float(split_config["test_ratio"])
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("Split ratios must add up to 1.0.")

    train_n = max(1, int(usable_rows * train_ratio))
    val_n = max(1, int(usable_rows * val_ratio))
    test_n = usable_rows - train_n - val_n
    if test_n < 1:
        raise ValueError("Test split has no rows after embargo.")

    train_end = train_n - 1
    val_start = train_end + embargo_steps + 1
    val_end = val_start + val_n - 1
    test_start = val_end + embargo_steps + 1

    split["split"] = "dropped"
    split.loc[:train_end, "split"] = "train"
    split.loc[val_start:val_end, "split"] = "val"
    split.loc[test_start:, "split"] = "test"

    metadata = {
        "embargo_steps": embargo_steps,
        "n_rows": n_rows,
        "train_rows": int((split["split"] == "train").sum()),
        "val_rows": int((split["split"] == "val").sum()),
        "test_rows": int((split["split"] == "test").sum()),
    }
    return split[split["split"] != "dropped"].reset_index(drop=True), metadata
