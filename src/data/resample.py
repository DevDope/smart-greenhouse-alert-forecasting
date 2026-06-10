"""Resampling utilities."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd


def resample_data(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if not config.get("resample", {}).get("enabled", True):
        return df.copy()

    timestamp_column = config["timestamp_column"]
    frequency = config["frequency"]
    aggregation = config["resample"]["aggregation"]
    present_columns = [column for column in df.columns if column != timestamp_column]
    optional_columns = set(config["schema"].get("optional_columns", []))
    missing_optional = [column for column in aggregation if column not in df.columns and column in optional_columns]
    if missing_optional:
        warnings.warn(
            "Optional canonical columns absent during resample; skipping aggregation for "
            f"{missing_optional}."
        )

    aggregation_map: dict[str, str] = {}
    for column in present_columns:
        if column in aggregation:
            if aggregation[column] == "sum":
                aggregation_map[column] = lambda series: series.sum(min_count=1)
            else:
                aggregation_map[column] = aggregation[column]
        elif pd.api.types.is_numeric_dtype(df[column]):
            aggregation_map[column] = "mean"

    if not aggregation_map:
        raise ValueError("No columns available for resampling after canonical adaptation.")

    resampled = (
        df.set_index(timestamp_column)
        .sort_index()
        .resample(frequency)
        .agg(aggregation_map)
        .reset_index()
    )
    return resampled
