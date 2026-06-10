"""Zappa-inspired explicit multitask chain for LightGBM."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ...evaluation.metrics_classification import compute_classification_metrics
from ...training.calibration import select_optimal_threshold
from ..base import BaseModel
from .common import BinaryProbabilityCalibrator
from .common import LGBMClassifier
from .common import TASK_ORDER
from .common import _base_feature_union
from .common import _clean_feature_frame
from .common import _feature_source_from_X
from .common import _multi_task_targets
from .common import _positive_probability_frame
from .common import _reference_probabilities
from .common import _zappa_engineering


class ZappaLGBMChainV2Classifier(BaseModel):
    model_name = "zappa_lgbm_chain_v2"

    def __init__(
        self,
        *,
        calibration_method: str = "sigmoid",
        calibration_fraction: float = 0.2,
        threshold_metric: str = "f1",
        random_state: int = 42,
        task_order: Sequence[str] = TASK_ORDER,
        include_calibrated_feature: bool = True,
        **estimator_params: Any,
    ) -> None:
        self.calibration_method = calibration_method
        self.calibration_fraction = calibration_fraction
        self.threshold_metric = threshold_metric
        self.random_state = random_state
        self.task_order = list(task_order)
        self.include_calibrated_feature = include_calibrated_feature
        self.estimator_params = estimator_params

        self.active_task_: str | None = None
        self.task_columns_: list[str] = []
        self.feature_columns_: list[str] = []
        self.engineered_source_: pd.DataFrame | None = None
        self.imputer_: SimpleImputer | None = None
        self.estimators_: dict[str, Any] = {}
        self.task_feature_columns_: dict[str, list[str]] = {}
        self.calibrators_: dict[str, BinaryProbabilityCalibrator] = {}
        self.thresholds_: dict[str, float] = {}
        self.training_metadata: dict[str, Any] = {}

    def _ordered_tasks(self, target_columns: Sequence[str]) -> list[str]:
        ordered = [task for task in self.task_order if task in target_columns]
        ordered.extend(task for task in target_columns if task not in ordered)
        return ordered

    def _prepare_features(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit or self.engineered_source_ is None:
            source = _feature_source_from_X(X)
            self.engineered_source_ = _zappa_engineering(source)
            combined = _base_feature_union(X, engineered=self.engineered_source_)
            self.feature_columns_ = [column for column in combined.columns if combined[column].notna().any()]
            return combined[self.feature_columns_]
        combined = _base_feature_union(X, engineered=self.engineered_source_)
        aligned = combined.reindex(columns=self.feature_columns_, fill_value=np.nan)
        return _clean_feature_frame(aligned)

    def _build_estimator(self) -> Any:
        if LGBMClassifier is None:  # pragma: no cover - environment without lightgbm
            raise ImportError("LightGBM is required for zappa_lgbm_chain_v2.")
        params = dict(self.estimator_params)
        params.setdefault("objective", "binary")
        params.setdefault("n_estimators", 180)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("num_leaves", 31)
        params.setdefault("subsample", 0.9)
        params.setdefault("colsample_bytree", 0.85)
        params.setdefault("random_state", self.random_state)
        params.setdefault("verbosity", -1)
        params.setdefault("n_jobs", 1)
        return LGBMClassifier(**params)

    def _fit_calibration(self, y_true: pd.Series, raw_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        target = pd.Series(y_true).astype(int).reset_index(drop=True)
        calibration_size = max(24, int(round(len(target) * float(self.calibration_fraction))))
        calibration_size = min(calibration_size, max(len(target) - 16, 0))
        if calibration_size > 0:
            y_cal = target.iloc[-calibration_size:].reset_index(drop=True)
            score_cal = np.asarray(raw_scores[-calibration_size:], dtype=float)
        else:
            y_cal = target
            score_cal = np.asarray(raw_scores, dtype=float)

        calibrator = BinaryProbabilityCalibrator(method=self.calibration_method).fit(score_cal, y_cal)
        calibrated_all = calibrator.transform(raw_scores)
        calibrated_tail = calibrator.transform(score_cal)
        threshold = (
            float(select_optimal_threshold(y_cal, pd.Series(calibrated_tail), metric=self.threshold_metric))
            if y_cal.nunique() >= 2
            else 0.5
        )
        raw_frame = _positive_probability_frame(_reference_probabilities(score_cal), y_cal.index)
        calibrated_frame = _positive_probability_frame(calibrated_tail, y_cal.index)
        payload = {
            "method": calibrator.kind,
            "threshold": threshold,
            "before": compute_classification_metrics(y_cal, (raw_frame["class_1"] >= 0.5).astype(int), raw_frame),
            "after_default": compute_classification_metrics(
                y_cal,
                (calibrated_frame["class_1"] >= 0.5).astype(int),
                calibrated_frame,
            ),
            "after_optimal": {
                **compute_classification_metrics(
                    y_cal,
                    (calibrated_frame["class_1"] >= threshold).astype(int),
                    calibrated_frame,
                ),
                "threshold": threshold,
            },
            "calibrator": calibrator,
        }
        return calibrated_all, (calibrated_all >= threshold).astype(int), payload

    def _augment_with_chain_features(
        self,
        design: pd.DataFrame,
        task_name: str,
        raw_scores: np.ndarray,
        calibrated_scores: np.ndarray,
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        augmented = design.copy()
        augmented[f"zappa_v2__chain_prev_{task_name}__proba_raw"] = np.asarray(raw_scores, dtype=float)
        if self.include_calibrated_feature:
            augmented[f"zappa_v2__chain_prev_{task_name}__proba_calibrated"] = np.asarray(calibrated_scores, dtype=float)
        augmented[f"zappa_v2__chain_prev_{task_name}__pred"] = np.asarray(predictions, dtype=float)
        return augmented

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series | None = None) -> "ZappaLGBMChainV2Classifier":
        targets, active_task = _multi_task_targets(X, y, task_order=self.task_order)
        self.active_task_ = active_task
        self.task_columns_ = self._ordered_tasks(targets.columns.tolist())
        features = self._prepare_features(X, fit=True)
        self.imputer_ = SimpleImputer(strategy="median")
        design = self.imputer_.fit_transform(features)
        current_design = pd.DataFrame(design, columns=self.feature_columns_)
        calibration_payload: dict[str, Any] = {}

        for index, task_name in enumerate(self.task_columns_):
            self.task_feature_columns_[task_name] = current_design.columns.tolist()
            estimator = self._build_estimator()
            estimator.fit(current_design.to_numpy(), targets[task_name].to_numpy(dtype=int))
            self.estimators_[task_name] = estimator

            raw_scores = np.asarray(estimator.predict_proba(current_design.to_numpy())[:, 1], dtype=float)
            calibrated_scores, hard_predictions, payload = self._fit_calibration(targets[task_name], raw_scores)
            self.calibrators_[task_name] = payload.pop("calibrator")
            self.thresholds_[task_name] = float(payload["threshold"])
            calibration_payload[task_name] = payload

            if index < len(self.task_columns_) - 1:
                current_design = self._augment_with_chain_features(
                    current_design,
                    task_name,
                    raw_scores,
                    calibrated_scores,
                    hard_predictions,
                )

        self.training_metadata = {
            "device_used": "cpu",
            "static_feature_count": int(len(self.feature_columns_)),
            "parameter_count": self.estimator_params.get("n_estimators", 180),
            "active_task": self.active_task_,
            "known_tasks": self.task_columns_,
            "chain_order": self.task_columns_,
            "include_calibrated_feature": self.include_calibrated_feature,
            "internal_calibration": calibration_payload,
        }
        return self

    def _sequential_outputs(self, X: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
        features = self._prepare_features(X, fit=False)
        design = self.imputer_.transform(features)
        current_design = pd.DataFrame(design, columns=self.feature_columns_, index=X.index)
        outputs: dict[str, dict[str, np.ndarray]] = {}

        for index, task_name in enumerate(self.task_columns_):
            task_design = current_design.reindex(columns=self.task_feature_columns_[task_name], fill_value=np.nan)
            raw_scores = np.asarray(self.estimators_[task_name].predict_proba(task_design.to_numpy())[:, 1], dtype=float)
            calibrated_scores = self.calibrators_[task_name].transform(raw_scores)
            threshold = self.thresholds_.get(task_name, 0.5)
            predictions = (calibrated_scores >= threshold).astype(int)
            outputs[task_name] = {
                "raw": raw_scores,
                "calibrated": calibrated_scores,
                "pred": predictions,
            }
            if index < len(self.task_columns_) - 1:
                current_design = self._augment_with_chain_features(
                    current_design,
                    task_name,
                    raw_scores,
                    calibrated_scores,
                    predictions,
                )
        return outputs

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        outputs = self._sequential_outputs(X)
        requested_task = str(X.attrs.get("task_name", self.active_task_ or self.task_columns_[0]))
        task_name = requested_task if requested_task in outputs else (self.active_task_ or self.task_columns_[0])
        return _positive_probability_frame(outputs[task_name]["calibrated"], X.index)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        outputs = self._sequential_outputs(X)
        requested_task = str(X.attrs.get("task_name", self.active_task_ or self.task_columns_[0]))
        task_name = requested_task if requested_task in outputs else (self.active_task_ or self.task_columns_[0])
        threshold = self.thresholds_.get(task_name, 0.5)
        predictions = (outputs[task_name]["calibrated"] >= threshold).astype(int)
        return pd.Series(predictions, index=X.index, name="prediction")


__all__ = ["ZappaLGBMChainV2Classifier"]
