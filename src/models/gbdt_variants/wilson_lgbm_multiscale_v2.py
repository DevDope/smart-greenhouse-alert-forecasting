"""Wilson-inspired hierarchical multiscale LightGBM wrapper."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .common import FeatureAwareBinaryGBDT
from .common import LGBMClassifier
from .common import _available_signals
from .common import _as_datetime
from .common import _clean_feature_frame
from .common import _patch_matrix
from .common import _rolling_quantile
from .common import _slope


def _patch_slope_frame(patch: pd.DataFrame) -> pd.Series:
    if patch.shape[1] <= 1:
        return pd.Series(np.nan, index=patch.index)
    return (patch.iloc[:, 0] - patch.iloc[:, -1]) / float(patch.shape[1] - 1)


def _wilson_multiscale_v2_engineering(source: pd.DataFrame, signals: Sequence[str]) -> pd.DataFrame:
    feature_map: dict[str, pd.Series] = {
        "timestamp": _as_datetime(source["timestamp"]),
    }
    scale_windows = (3, 6, 12, 24, 48)
    patch_layouts = ((4, 12), (8, 6), (12, 4))

    for column in signals:
        series = pd.to_numeric(source[column], errors="coerce")
        short_mean = series.rolling(window=6, min_periods=3).mean()
        long_mean = series.rolling(window=48, min_periods=12).mean()
        short_std = series.rolling(window=6, min_periods=3).std()
        long_std = series.rolling(window=48, min_periods=12).std()
        slope_6 = _slope(series, 6)
        slope_24 = _slope(series, 24)

        feature_map[f"wilson_v2__{column}__trend_strength"] = (short_mean - long_mean) / (long_std.abs() + 1e-6)
        feature_map[f"wilson_v2__{column}__volatility_by_scale"] = short_std / (long_std.abs() + 1e-6)
        feature_map[f"wilson_v2__{column}__trend_delta_6_24"] = slope_6 - slope_24
        feature_map[f"wilson_v2__{column}__frequency_gap_6_48"] = short_mean - long_mean

        for window in scale_windows:
            min_periods = max(2, window // 2)
            rolled = series.rolling(window=window, min_periods=min_periods)
            roll_mean = rolled.mean()
            feature_map[f"wilson_v2__{column}__roll_mean_{window}"] = roll_mean
            feature_map[f"wilson_v2__{column}__roll_std_{window}"] = rolled.std()
            feature_map[f"wilson_v2__{column}__roll_min_{window}"] = rolled.min()
            feature_map[f"wilson_v2__{column}__roll_max_{window}"] = rolled.max()
            feature_map[f"wilson_v2__{column}__roll_q25_{window}"] = _rolling_quantile(series, window, 0.25)
            feature_map[f"wilson_v2__{column}__roll_q75_{window}"] = _rolling_quantile(series, window, 0.75)
            feature_map[f"wilson_v2__{column}__slope_{window}"] = _slope(series, window)
            feature_map[f"wilson_v2__{column}__delta_{window}"] = series - series.shift(window)
            feature_map[f"wilson_v2__{column}__residual_{window}"] = series - roll_mean

        for patch_count, patch_size in patch_layouts:
            patch_mean_names: list[str] = []
            patch_std_names: list[str] = []
            patch_slope_names: list[str] = []
            for patch_index in range(patch_count):
                patch = _patch_matrix(series, start=patch_index * patch_size, patch_size=patch_size)
                label = patch_index + 1
                mean_name = f"wilson_v2__{column}__patch{patch_count}x{patch_size}_mean_p{label}"
                std_name = f"wilson_v2__{column}__patch{patch_count}x{patch_size}_std_p{label}"
                slope_name = f"wilson_v2__{column}__patch{patch_count}x{patch_size}_slope_p{label}"
                feature_map[mean_name] = patch.mean(axis=1)
                feature_map[std_name] = patch.std(axis=1)
                feature_map[slope_name] = _patch_slope_frame(patch)
                patch_mean_names.append(mean_name)
                patch_std_names.append(std_name)
                patch_slope_names.append(slope_name)

            feature_map[f"wilson_v2__{column}__patch{patch_count}x{patch_size}_mean_swing"] = (
                feature_map[patch_mean_names[0]] - feature_map[patch_mean_names[-1]]
            )
            feature_map[f"wilson_v2__{column}__patch{patch_count}x{patch_size}_vol_swing"] = (
                feature_map[patch_std_names[0]] - feature_map[patch_std_names[-1]]
            )
            feature_map[f"wilson_v2__{column}__patch{patch_count}x{patch_size}_slope_swing"] = (
                feature_map[patch_slope_names[0]] - feature_map[patch_slope_names[-1]]
            )

    return _clean_feature_frame(pd.DataFrame(feature_map))


class WilsonLGBMMultiscaleV2Classifier(FeatureAwareBinaryGBDT):
    model_name = "wilson_lgbm_multiscale_v2"

    def _build_engineered_source(self, source_frame: pd.DataFrame) -> pd.DataFrame:
        signals = _available_signals(source_frame, max_count=min(self.max_source_signals, 4))
        return _wilson_multiscale_v2_engineering(source_frame, signals)

    def _build_estimator(self, feature_names: Sequence[str]) -> Any:
        if LGBMClassifier is None:  # pragma: no cover - environment without lightgbm
            raise ImportError("LightGBM is required for wilson_lgbm_multiscale_v2.")
        params = dict(self.estimator_params)
        params.setdefault("objective", "binary")
        params.setdefault("n_estimators", 220)
        params.setdefault("learning_rate", 0.04)
        params.setdefault("num_leaves", 31)
        params.setdefault("min_child_samples", 24)
        params.setdefault("subsample", 0.85)
        params.setdefault("colsample_bytree", 0.78)
        params.setdefault("random_state", self.random_state)
        params.setdefault("verbosity", -1)
        params.setdefault("n_jobs", 1)
        return LGBMClassifier(**params)


__all__ = ["WilsonLGBMMultiscaleV2Classifier"]
