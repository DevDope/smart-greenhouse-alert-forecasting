"""Cleaning utilities for canonical greenhouse data."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd


def clean_data(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    timestamp_column = config["timestamp_column"]
    cleaning = config.get("cleaning", {})
    bounds = cleaning.get("bounds", {})
    numeric_features = [column for column in config.get("numeric_features", []) if column in df.columns]
    required_columns = set(config["schema"]["required_columns"])

    cleaned = df.copy()
    cleaned[timestamp_column] = pd.to_datetime(cleaned[timestamp_column], utc=False)
    invalid_timestamps = int(cleaned[timestamp_column].isna().sum())
    if invalid_timestamps:
        warnings.warn(f"Dropping {invalid_timestamps} rows with invalid timestamps during cleaning.")
        cleaned = cleaned.loc[cleaned[timestamp_column].notna()].copy()
    cleaned = cleaned.sort_values(timestamp_column)

    duplicate_strategy = cleaning.get("duplicate_strategy", "last")
    cleaned = cleaned.drop_duplicates(subset=[timestamp_column], keep=duplicate_strategy)

    for column in numeric_features:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        if column in bounds:
            lower, upper = bounds[column]
            cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)

    strategy = cleaning.get("missing_numeric", "interpolate_ffill_bfill")
    if strategy == "interpolate_ffill_bfill":
        cleaned[numeric_features] = cleaned[numeric_features].interpolate(limit_direction="both")
        cleaned[numeric_features] = cleaned[numeric_features].ffill().bfill()
    elif strategy == "ffill_bfill":
        cleaned[numeric_features] = cleaned[numeric_features].ffill().bfill()
    else:
        raise ValueError(f"Unsupported missing value strategy: {strategy}")

    for column in required_columns:
        if column != timestamp_column and column in cleaned.columns and cleaned[column].notna().sum() == 0:
            raise ValueError(f"Required canonical column '{column}' became empty after cleaning.")

    cleaned = cleaned.reset_index(drop=True)
    return cleaned
