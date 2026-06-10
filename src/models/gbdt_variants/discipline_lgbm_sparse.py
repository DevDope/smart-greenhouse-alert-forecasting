"""Austere LightGBM variant for the active benchmark line."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .common import (
    FeatureAwareBinaryGBDT,
    LGBMClassifier,
    _as_datetime,
    _clean_feature_frame,
    _fripp_constraints,
)


DISCIPLINE_PRIORITY_COLUMNS = (
    "soil_moisture",
    "air_temperature",
    "relative_humidity",
    "radiation_proxy",
    "vpd",
    "irrigation_lph",
    "irrigation_event_observed",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "discipline__soil_moisture__delta_6",
    "discipline__soil_moisture__delta_24",
    "discipline__soil_moisture__vs_median_24",
    "discipline__soil_moisture__drydown_12",
    "discipline__air_temperature__delta_6",
    "discipline__radiation_proxy__mean_12",
    "discipline__vpd__mean_12",
    "discipline__irrigation_lph__sum_12",
)


def _discipline_engineering(source: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"timestamp": _as_datetime(source["timestamp"])})
    if "soil_moisture" in source.columns:
        soil = pd.to_numeric(source["soil_moisture"], errors="coerce")
        median_24 = soil.rolling(window=24, min_periods=8).median()
        frame["discipline__soil_moisture__delta_6"] = soil - soil.shift(6)
        frame["discipline__soil_moisture__delta_24"] = soil - soil.shift(24)
        frame["discipline__soil_moisture__vs_median_24"] = soil - median_24
        frame["discipline__soil_moisture__drydown_12"] = (soil.shift(12) - soil).clip(lower=0.0)
    if "air_temperature" in source.columns:
        temperature = pd.to_numeric(source["air_temperature"], errors="coerce")
        frame["discipline__air_temperature__delta_6"] = temperature - temperature.shift(6)
    if "radiation_proxy" in source.columns:
        radiation = pd.to_numeric(source["radiation_proxy"], errors="coerce")
        frame["discipline__radiation_proxy__mean_12"] = radiation.rolling(
            window=12,
            min_periods=6,
        ).mean()
    if "vpd" in source.columns:
        vpd = pd.to_numeric(source["vpd"], errors="coerce")
        frame["discipline__vpd__mean_12"] = vpd.rolling(window=12, min_periods=6).mean()
    if "irrigation_lph" in source.columns:
        irrigation = pd.to_numeric(source["irrigation_lph"], errors="coerce")
        frame["discipline__irrigation_lph__sum_12"] = irrigation.rolling(
            window=12,
            min_periods=3,
        ).sum()
    return _clean_feature_frame(frame)


class DisciplineLGBMSparseClassifier(FeatureAwareBinaryGBDT):
    """Regalia-GBDT Discipline architecture.

    This is the official austere/constrained architecture, not the historical
    Fripp compatibility wrapper. It keeps the shared GBDT fitting interface but
    owns its temporal encoder, sparse feature gate, and constrained head.
    """

    model_name = "discipline_lgbm_sparse"

    def _build_engineered_source(self, source_frame: pd.DataFrame) -> pd.DataFrame:
        return _discipline_engineering(source_frame)

    def _select_feature_columns(self, combined: pd.DataFrame) -> list[str]:
        selected = [column for column in DISCIPLINE_PRIORITY_COLUMNS if column in combined.columns]
        return selected or combined.columns.tolist()

    def _build_estimator(self, feature_names: Sequence[str]) -> Any:
        if LGBMClassifier is None:  # pragma: no cover - environment without lightgbm
            raise ImportError("LightGBM is required for discipline_lgbm_sparse.")
        params = dict(self.estimator_params)
        monotone, groups = _fripp_constraints(feature_names)
        params.setdefault("objective", "binary")
        params.setdefault("n_estimators", 140)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("num_leaves", 15)
        params.setdefault("max_depth", 4)
        params.setdefault("min_child_samples", 40)
        params.setdefault("subsample", 0.75)
        params.setdefault("colsample_bytree", 0.65)
        params.setdefault("reg_alpha", 1.0)
        params.setdefault("reg_lambda", 3.0)
        params.setdefault("random_state", self.random_state)
        params.setdefault("verbosity", -1)
        params.setdefault("n_jobs", 1)
        params.setdefault("monotone_constraints", monotone)
        if groups:
            params.setdefault("interaction_constraints", groups)
        return LGBMClassifier(**params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
    ) -> "DisciplineLGBMSparseClassifier":
        super().fit(X, y, sample_weight=sample_weight)
        self.training_metadata["architecture"] = "Regalia-GBDT/Discipline"
        self.training_metadata["architecture_blocks"] = [
            "temporal_agronomic_encoder",
            "sparse_prioritized_feature_gate",
            "constrained_calibrated_gbdt_head",
        ]
        self.training_metadata["is_legacy_wrapper"] = False
        self.training_metadata["variant_impl"] = "discipline_sparse_architecture"
        self.training_metadata["feature_policy"] = "austere_priority_columns"
        return self


__all__ = ["DisciplineLGBMSparseClassifier"]
