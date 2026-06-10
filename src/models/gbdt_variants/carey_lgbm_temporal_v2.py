"""Carey-inspired richer temporal LightGBM wrapper."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .common import FeatureAwareBinaryGBDT
from .common import LGBMClassifier
from .common import _available_signals
from .common import _as_datetime
from .common import _clean_feature_frame
from .common import _rolling_quantile
from .common import _slope


def _time_since_last_event(timestamps: pd.Series, events: pd.Series) -> pd.Series:
    event_times = timestamps.where(events.fillna(0).astype(float) > 0.0)
    last_event = event_times.ffill()
    minutes = (timestamps - last_event).dt.total_seconds() / 60.0
    return minutes.where(last_event.notna())


def _carey_temporal_v2_engineering(source: pd.DataFrame, signals: Sequence[str]) -> pd.DataFrame:
    timestamps = _as_datetime(source["timestamp"])
    elapsed_minutes = ((timestamps - timestamps.iloc[0]).dt.total_seconds() / 60.0).fillna(0.0)
    hour = timestamps.dt.hour + (timestamps.dt.minute / 60.0)
    dow = timestamps.dt.dayofweek + (hour / 24.0)
    hour_angle = (2.0 * np.pi * hour) / 24.0
    week_angle = (2.0 * np.pi * dow) / 7.0
    normalized_time = elapsed_minutes / max(float(elapsed_minutes.max()), 1.0)

    feature_map: dict[str, pd.Series] = {
        "timestamp": timestamps,
        "carey_v2__time_linear": normalized_time,
        "carey_v2__time_linear_sq": normalized_time**2,
        "carey_v2__week_sin": np.sin(week_angle),
        "carey_v2__week_cos": np.cos(week_angle),
    }

    for harmonic in (1, 2, 3, 4):
        feature_map[f"carey_v2__hour_sin_h{harmonic}"] = np.sin(harmonic * hour_angle)
        feature_map[f"carey_v2__hour_cos_h{harmonic}"] = np.cos(harmonic * hour_angle)
        feature_map[f"carey_v2__time2vec_sin_{harmonic}"] = np.sin((harmonic + 0.5) * normalized_time * 2.0 * np.pi)
        feature_map[f"carey_v2__time2vec_cos_{harmonic}"] = np.cos((harmonic + 0.5) * normalized_time * 2.0 * np.pi)

    event_indicator = (
        pd.to_numeric(source["irrigation_event_observed"], errors="coerce")
        if "irrigation_event_observed" in source.columns
        else pd.Series(0.0, index=source.index)
    ).fillna(0.0)
    feature_map["carey_v2__time_since_last_event"] = _time_since_last_event(timestamps, event_indicator)
    feature_map["carey_v2__event_density_recent"] = event_indicator.rolling(window=24, min_periods=3).mean()

    recent_change_terms: list[pd.Series] = []
    contrast_terms: list[pd.Series] = []
    for column in signals:
        series = pd.to_numeric(source[column], errors="coerce")
        mu_12 = series.rolling(window=12, min_periods=6).mean()
        sd_12 = series.rolling(window=12, min_periods=6).std().replace(0.0, np.nan)
        mu_48 = series.rolling(window=48, min_periods=12).mean()
        sd_48 = series.rolling(window=48, min_periods=12).std().replace(0.0, np.nan)
        fast_mean = series.rolling(window=6, min_periods=3).mean()
        slow_mean = series.rolling(window=24, min_periods=6).mean()
        diff_1 = series.diff()
        diff_3 = series.diff(3)
        abs_diff = diff_1.abs()
        local_energy_12 = abs_diff.pow(2).rolling(window=12, min_periods=4).mean()
        local_energy_24 = abs_diff.pow(2).rolling(window=24, min_periods=6).mean()
        scale_contrast = fast_mean - slow_mean

        feature_map[f"carey_v2__{column}__mu_12"] = mu_12
        feature_map[f"carey_v2__{column}__sd_12"] = sd_12
        feature_map[f"carey_v2__{column}__mu_48"] = mu_48
        feature_map[f"carey_v2__{column}__sd_48"] = sd_48
        feature_map[f"carey_v2__{column}__norm_local_12"] = (series - mu_12) / sd_12
        feature_map[f"carey_v2__{column}__norm_local_48"] = (series - mu_48) / sd_48
        feature_map[f"carey_v2__{column}__residual_fast"] = series - fast_mean
        feature_map[f"carey_v2__{column}__residual_slow"] = series - slow_mean
        feature_map[f"carey_v2__{column}__spectral_residual"] = fast_mean - slow_mean
        feature_map[f"carey_v2__{column}__wavelet_like_energy_12"] = local_energy_12
        feature_map[f"carey_v2__{column}__wavelet_like_energy_24"] = local_energy_24
        feature_map[f"carey_v2__{column}__scale_contrast"] = scale_contrast
        feature_map[f"carey_v2__{column}__delta_1"] = diff_1
        feature_map[f"carey_v2__{column}__delta_3"] = diff_3
        feature_map[f"carey_v2__{column}__slope_6"] = _slope(series, 6)
        feature_map[f"carey_v2__{column}__slope_24"] = _slope(series, 24)
        feature_map[f"carey_v2__{column}__roll_q25_12"] = _rolling_quantile(series, 12, 0.25)
        feature_map[f"carey_v2__{column}__roll_q75_12"] = _rolling_quantile(series, 12, 0.75)
        feature_map[f"carey_v2__{column}__roll_q25_48"] = _rolling_quantile(series, 48, 0.25)
        feature_map[f"carey_v2__{column}__roll_q75_48"] = _rolling_quantile(series, 48, 0.75)

        for lag in (1, 2, 3, 6, 12, 24, 48):
            lagged = series.shift(lag)
            feature_map[f"carey_v2__{column}__lag_{lag}"] = lagged
            feature_map[f"carey_v2__{column}__lag_norm48_{lag}"] = (lagged - mu_48) / sd_48

        recent_change_terms.append(abs_diff.rolling(window=12, min_periods=4).mean())
        contrast_terms.append(scale_contrast.abs())

    if recent_change_terms:
        recent_change_frame = pd.concat(recent_change_terms, axis=1)
        feature_map["carey_v2__recent_change_intensity"] = recent_change_frame.mean(axis=1)
    else:
        feature_map["carey_v2__recent_change_intensity"] = pd.Series(np.nan, index=source.index)

    if contrast_terms:
        contrast_frame = pd.concat(contrast_terms, axis=1)
        feature_map["carey_v2__short_long_contrast"] = contrast_frame.mean(axis=1)
    else:
        feature_map["carey_v2__short_long_contrast"] = pd.Series(np.nan, index=source.index)

    return _clean_feature_frame(pd.DataFrame(feature_map))


class CareyLGBMTemporalV2Classifier(FeatureAwareBinaryGBDT):
    model_name = "carey_lgbm_temporal_v2"

    def _build_engineered_source(self, source_frame: pd.DataFrame) -> pd.DataFrame:
        signals = _available_signals(source_frame, max_count=self.max_source_signals)
        return _carey_temporal_v2_engineering(source_frame, signals)

    def _build_estimator(self, feature_names: Sequence[str]) -> Any:
        if LGBMClassifier is None:  # pragma: no cover - environment without lightgbm
            raise ImportError("LightGBM is required for carey_lgbm_temporal_v2.")
        params = dict(self.estimator_params)
        params.setdefault("objective", "binary")
        params.setdefault("n_estimators", 220)
        params.setdefault("learning_rate", 0.04)
        params.setdefault("num_leaves", 31)
        params.setdefault("min_child_samples", 26)
        params.setdefault("subsample", 0.9)
        params.setdefault("colsample_bytree", 0.82)
        params.setdefault("random_state", self.random_state)
        params.setdefault("verbosity", -1)
        params.setdefault("n_jobs", 1)
        return LGBMClassifier(**params)


__all__ = ["CareyLGBMTemporalV2Classifier"]
