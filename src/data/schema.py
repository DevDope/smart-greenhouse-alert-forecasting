"""Schema validation for canonical greenhouse data."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd


def _schema_columns(raw_columns: list[str] | dict[str, Any]) -> list[str]:
    if isinstance(raw_columns, dict):
        return list(raw_columns.keys())
    return list(raw_columns)


def validate_schema(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    timestamp_column = config["timestamp_column"]
    required_columns = _schema_columns(config["schema"]["required_columns"])
    optional_columns = _schema_columns(config["schema"].get("optional_columns", []))
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    validated = df.copy()
    validated[timestamp_column] = pd.to_datetime(validated[timestamp_column], utc=False)
    if validated[timestamp_column].isna().any():
        raise ValueError("Canonical dataset contains invalid timestamps after adaptation.")

    for column, dtype_name in config["schema"].get("dtypes", {}).items():
        if column not in validated.columns:
            if column in optional_columns:
                warnings.warn(f"Optional canonical column '{column}' is absent from the dataset.")
                continue
            raise ValueError(f"Configured dtype for missing required column '{column}'.")
        if dtype_name == "float":
            validated[column] = pd.to_numeric(validated[column], errors="raise")
        if column in required_columns and validated[column].notna().sum() == 0:
            raise ValueError(f"Required canonical column '{column}' has no usable values.")
    return validated.sort_values(timestamp_column).reset_index(drop=True)
