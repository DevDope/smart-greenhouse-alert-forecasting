"""Feature engineering utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, data_config: dict[str, Any]) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    feature_cfg = data_config.get("feature_engineering", {})
    numeric_features = data_config.get("numeric_features", [])
    timestamp_column = data_config["timestamp_column"]
    available_features = [
        column for column in numeric_features if column in df.columns and df[column].notna().sum() > 0
    ]
    unavailable_features = [column for column in numeric_features if column not in available_features]
    missing_required_features = [
        column for column in feature_cfg.get("required_feature_columns", []) if column not in available_features
    ]
    if missing_required_features:
        raise ValueError(
            "Cannot build features because required feature columns are unavailable: "
            f"{missing_required_features}."
        )

    feature_map: dict[str, pd.Series] = {timestamp_column: pd.to_datetime(df[timestamp_column])}
    if feature_cfg.get("include_raw_features", True):
        for column in available_features:
            feature_map[column] = df[column]

    lags = feature_cfg.get("lags", [])
    rolling_windows = feature_cfg.get("rolling_windows", [])
    rolling_statistics = feature_cfg.get("rolling_statistics", [])
    deltas = feature_cfg.get("deltas", [])

    for column in available_features:
        series = df[column]
        for lag in lags:
            feature_map[f"{column}__lag_{lag}"] = series.shift(lag)
        for window in rolling_windows:
            rolled = series.rolling(window=window, min_periods=window)
            for statistic in rolling_statistics:
                if statistic == "mean":
                    feature_map[f"{column}__roll_mean_{window}"] = rolled.mean()
                elif statistic == "std":
                    feature_map[f"{column}__roll_std_{window}"] = rolled.std()
                elif statistic == "min":
                    feature_map[f"{column}__roll_min_{window}"] = rolled.min()
                elif statistic == "max":
                    feature_map[f"{column}__roll_max_{window}"] = rolled.max()
                else:
                    raise ValueError(f"Unsupported rolling statistic: {statistic}")
        for delta in deltas:
            feature_map[f"{column}__delta_{delta}"] = series - series.shift(delta)

    if feature_cfg.get("include_calendar", True):
        timestamps = pd.to_datetime(feature_map[timestamp_column])
        hour_angle = 2 * np.pi * timestamps.dt.hour / 24.0
        dow_angle = 2 * np.pi * timestamps.dt.dayofweek / 7.0
        feature_map["hour_sin"] = pd.Series(np.sin(hour_angle), index=timestamps.index)
        feature_map["hour_cos"] = pd.Series(np.cos(hour_angle), index=timestamps.index)
        feature_map["dow_sin"] = pd.Series(np.sin(dow_angle), index=timestamps.index)
        feature_map["dow_cos"] = pd.Series(np.cos(dow_angle), index=timestamps.index)

    features = pd.DataFrame(feature_map)
    feature_columns = [column for column in features.columns if column != timestamp_column]
    lookback_steps = max(
        [
            feature_cfg.get("window_size_steps", 1),
            max(lags, default=0) + 1,
            max(rolling_windows, default=1),
            max(deltas, default=0) + 1,
        ]
    )
    metadata = {
        "max_lag_steps": max(lags, default=0),
        "max_rolling_lookback_steps": max(rolling_windows, default=1),
        "window_size_steps": lookback_steps,
        "available_numeric_features": available_features,
        "unavailable_numeric_features": unavailable_features,
    }
    return features, feature_columns, metadata
