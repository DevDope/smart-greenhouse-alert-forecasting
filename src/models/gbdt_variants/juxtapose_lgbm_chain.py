"""Regalia-GBDT Juxtapose architecture."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ...evaluation.metrics_classification import compute_classification_metrics
from ...training.calibration import select_optimal_threshold
from ..base import BaseModel
from .common import (
    BinaryProbabilityCalibrator,
    LGBMClassifier,
    TASK_ORDER,
    _base_feature_union,
    _clean_feature_frame,
    _feature_source_from_X,
    _multi_task_targets,
    _positive_probability_frame,
    _reference_probabilities,
    _zappa_engineering,
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    clean_denominator = denominator.where(denominator.abs() > 1e-6)
    return numerator / clean_denominator


def _juxtapose_engineering(source: pd.DataFrame) -> pd.DataFrame:
    frame = _zappa_engineering(source)
    if "soil_moisture" in source.columns:
        soil = pd.to_numeric(source["soil_moisture"], errors="coerce")
        fast = soil.rolling(window=6, min_periods=3).mean()
        slow = soil.rolling(window=24, min_periods=8).mean()
        frame["juxtapose__soil_moisture__fast_minus_slow"] = fast - slow
        frame["juxtapose__soil_moisture__fast_over_slow"] = _safe_ratio(fast, slow)
    if {"air_temperature", "relative_humidity"} <= set(source.columns):
        temperature = pd.to_numeric(source["air_temperature"], errors="coerce")
        humidity = pd.to_numeric(source["relative_humidity"], errors="coerce")
        warm = temperature.rolling(window=12, min_periods=6).mean()
        dry = (100.0 - humidity).rolling(window=12, min_periods=6).mean()
        frame["juxtapose__warm_dry_product_12"] = warm * dry
    if {"radiation_proxy", "soil_moisture"} <= set(source.columns):
        radiation = pd.to_numeric(source["radiation_proxy"], errors="coerce")
        soil = pd.to_numeric(source["soil_moisture"], errors="coerce")
        light_12 = radiation.rolling(window=12, min_periods=6).mean()
        soil_12 = soil.rolling(window=12, min_periods=6).mean()
        frame["juxtapose__light_soil_gap_12"] = light_12 - soil_12
    if "irrigation_lph" in source.columns:
        irrigation = pd.to_numeric(source["irrigation_lph"], errors="coerce")
        recent = irrigation.rolling(window=6, min_periods=2).sum()
        baseline = irrigation.rolling(window=24, min_periods=8).sum()
        frame["juxtapose__irrigation_recent_share"] = _safe_ratio(recent, baseline)
    return _clean_feature_frame(frame)


class JuxtaposeLGBMChainClassifier(BaseModel):
    """Contrastive multitask GBDT architecture with a cross-task bridge.

    The historical Zappa compatibility wrapper remains available separately.
    Juxtapose owns its contrastive temporal encoder, first-stage task relation
    heads, cross-task bridge features, and calibrated event heads.
    """

    model_name = "juxtapose_lgbm_chain"

    def __init__(
        self,
        *,
        calibration_method: str = "sigmoid",
        calibration_fraction: float = 0.2,
        threshold_metric: str = "f1",
        random_state: int = 42,
        task_order: Sequence[str] = TASK_ORDER,
        bridge_n_estimators: int = 64,
        **estimator_params: Any,
    ) -> None:
        self.calibration_method = calibration_method
        self.calibration_fraction = calibration_fraction
        self.threshold_metric = threshold_metric
        self.random_state = random_state
        self.task_order = list(task_order)
        self.bridge_n_estimators = bridge_n_estimators
        self.estimator_params = estimator_params

        self.active_task_: str | None = None
        self.task_columns_: list[str] = []
        self.feature_columns_: list[str] = []
        self.bridge_columns_: list[str] = []
        self.engineered_source_: pd.DataFrame | None = None
        self.imputer_: SimpleImputer | None = None
        self.primary_estimators_: dict[str, Any] = {}
        self.bridge_estimators_: dict[str, Any] = {}
        self.calibrators_: dict[str, BinaryProbabilityCalibrator] = {}
        self.thresholds_: dict[str, float] = {}
        self.training_metadata: dict[str, Any] = {}

    def _prepare_features(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit or self.engineered_source_ is None:
            source = _feature_source_from_X(X)
            self.engineered_source_ = _juxtapose_engineering(source)
            combined = _base_feature_union(X, engineered=self.engineered_source_)
            self.feature_columns_ = [column for column in combined.columns if combined[column].notna().any()]
            return combined[self.feature_columns_]
        combined = _base_feature_union(X, engineered=self.engineered_source_)
        return combined.reindex(columns=self.feature_columns_, fill_value=float("nan"))

    def _estimator(self, *, bridge: bool = False) -> Any:
        if LGBMClassifier is None:  # pragma: no cover - environment without lightgbm
            raise ImportError("LightGBM is required for juxtapose_lgbm_chain.")
        params = dict(self.estimator_params)
        params.setdefault("objective", "binary")
        params.setdefault("n_estimators", self.bridge_n_estimators if bridge else 160)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("num_leaves", 15 if bridge else 31)
        params.setdefault("subsample", 0.9)
        params.setdefault("colsample_bytree", 0.85)
        params.setdefault("random_state", self.random_state)
        params.setdefault("verbosity", -1)
        params.setdefault("n_jobs", 1)
        return LGBMClassifier(**params)

    def _fit_estimator_or_prior(self, design: np.ndarray, target: pd.Series, *, bridge: bool = False) -> Any:
        if target.nunique() < 2:
            return ConstantProbabilityEstimator(float(target.mean()))
        estimator = self._estimator(bridge=bridge)
        estimator.fit(design, target.to_numpy(dtype=int))
        return estimator

    def _primary_scores(self, design: np.ndarray) -> np.ndarray:
        columns: list[np.ndarray] = []
        for task in self.task_columns_:
            estimator = self.primary_estimators_[task]
            probabilities = estimator.predict_proba(design)
            columns.append(np.asarray(probabilities)[:, 1].astype(float))
        return np.column_stack(columns)

    def _bridge_feature_frame(self, scores: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(index=range(scores.shape[0]))
        for index, task in enumerate(self.task_columns_):
            frame[f"juxtapose_bridge__prob__{task}"] = scores[:, index]
        if scores.shape[1] > 1:
            frame["juxtapose_bridge__mean_prob"] = scores.mean(axis=1)
            frame["juxtapose_bridge__std_prob"] = scores.std(axis=1)
            frame["juxtapose_bridge__spread_prob"] = scores.max(axis=1) - scores.min(axis=1)
            for left_index, left_task in enumerate(self.task_columns_):
                for right_index, right_task in enumerate(self.task_columns_):
                    if left_index >= right_index:
                        continue
                    diff = scores[:, left_index] - scores[:, right_index]
                    frame[f"juxtapose_bridge__diff__{left_task}__{right_task}"] = diff
                    frame[f"juxtapose_bridge__absdiff__{left_task}__{right_task}"] = np.abs(diff)
        return frame

    def _bridge_design(self, design: np.ndarray, scores: np.ndarray, *, fit: bool) -> np.ndarray:
        bridge_frame = self._bridge_feature_frame(scores)
        if fit:
            self.bridge_columns_ = bridge_frame.columns.tolist()
        bridge_aligned = bridge_frame.reindex(columns=self.bridge_columns_, fill_value=0.5)
        return np.column_stack([design, bridge_aligned.to_numpy(dtype=float)])

    def _calibrate_tasks(self, targets: pd.DataFrame, raw_scores: np.ndarray) -> dict[str, Any]:
        calibration_size = max(24, int(round(len(targets) * float(self.calibration_fraction))))
        calibration_size = min(calibration_size, max(len(targets) - 16, 0))
        payload: dict[str, Any] = {}
        for task_index, task_name in enumerate(self.task_columns_):
            scores = raw_scores[:, task_index]
            y_task = targets[task_name].reset_index(drop=True)
            if calibration_size > 0:
                y_cal = y_task.iloc[-calibration_size:].reset_index(drop=True)
                score_cal = scores[-calibration_size:]
            else:
                y_cal = y_task
                score_cal = scores
            calibrator = BinaryProbabilityCalibrator(method=self.calibration_method).fit(score_cal, y_cal)
            self.calibrators_[task_name] = calibrator
            calibrated = calibrator.transform(score_cal)
            threshold = (
                float(select_optimal_threshold(y_cal, pd.Series(calibrated), metric=self.threshold_metric))
                if y_cal.nunique() >= 2
                else 0.5
            )
            self.thresholds_[task_name] = threshold
            raw_frame = _positive_probability_frame(_reference_probabilities(score_cal), y_cal.index)
            calibrated_frame = _positive_probability_frame(calibrated, y_cal.index)
            payload[task_name] = {
                "method": calibrator.kind,
                "threshold": threshold,
                "before": compute_classification_metrics(
                    y_cal,
                    (raw_frame["class_1"] >= 0.5).astype(int),
                    raw_frame,
                ),
                "after_optimal": {
                    **compute_classification_metrics(
                        y_cal,
                        (calibrated_frame["class_1"] >= threshold).astype(int),
                        calibrated_frame,
                    ),
                    "threshold": threshold,
                },
            }
        return payload

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
    ) -> "JuxtaposeLGBMChainClassifier":
        targets, active_task = _multi_task_targets(X, y, task_order=self.task_order)
        self.active_task_ = active_task
        self.task_columns_ = targets.columns.tolist()
        features = self._prepare_features(X, fit=True)
        self.imputer_ = SimpleImputer(strategy="median")
        design = self.imputer_.fit_transform(features)

        for task in self.task_columns_:
            self.primary_estimators_[task] = self._fit_estimator_or_prior(
                design,
                targets[task].astype(int).reset_index(drop=True),
            )
        primary_scores = self._primary_scores(design)
        bridge_design = self._bridge_design(design, primary_scores, fit=True)

        raw_scores = np.zeros((len(targets), len(self.task_columns_)), dtype=float)
        for task_index, task in enumerate(self.task_columns_):
            estimator = self._fit_estimator_or_prior(
                bridge_design,
                targets[task].astype(int).reset_index(drop=True),
                bridge=True,
            )
            self.bridge_estimators_[task] = estimator
            raw_scores[:, task_index] = np.asarray(estimator.predict_proba(bridge_design))[:, 1]

        calibration_payload = self._calibrate_tasks(targets, raw_scores)
        self.training_metadata = {
            "architecture": "Regalia-GBDT/Juxtapose",
            "architecture_blocks": [
                "contrastive_temporal_encoder",
                "task_relation_encoder",
                "cross_task_bridge",
                "calibrated_event_head",
            ],
            "is_legacy_wrapper": False,
            "variant_impl": "juxtapose_cross_task_bridge_architecture",
            "feature_policy": "contrastive_temporal_plus_bridge_features",
            "device_used": "cpu",
            "static_feature_count": int(len(self.feature_columns_)),
            "bridge_feature_count": int(len(self.bridge_columns_)),
            "parameter_count": self.estimator_params.get("n_estimators", 160),
            "active_task": self.active_task_,
            "known_tasks": self.task_columns_,
            "internal_calibration": calibration_payload,
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self._prepare_features(X, fit=False)
        design = self.imputer_.transform(features)
        primary_scores = self._primary_scores(design)
        bridge_design = self._bridge_design(design, primary_scores, fit=False)
        active_index = self.task_columns_.index(self.active_task_)
        active_task = self.task_columns_[active_index]
        raw_scores = np.asarray(self.bridge_estimators_[active_task].predict_proba(bridge_design))[:, 1]
        calibrated = self.calibrators_[active_task].transform(raw_scores)
        return _positive_probability_frame(calibrated, X.index)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(X)
        threshold = self.thresholds_.get(self.active_task_ or "", 0.5)
        return (probabilities["class_1"] >= threshold).astype(int).rename("prediction")


class ConstantProbabilityEstimator:
    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = np.full(shape=(len(X),), fill_value=self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


__all__ = ["JuxtaposeLGBMChainClassifier"]
