"""Juxtapose v2 relation-aware multitask GBDT."""

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
    _as_datetime,
    _base_feature_union,
    _clean_feature_frame,
    _feature_source_from_X,
    _multi_task_targets,
    _positive_probability_frame,
    _reference_probabilities,
)
from .juxtapose_lgbm_chain import ConstantProbabilityEstimator, _juxtapose_engineering


def _numeric(source: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(source[column], errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    clean_denominator = denominator.where(denominator.abs() > 1e-6)
    return numerator / clean_denominator


def _source_aware_engineering(source: pd.DataFrame) -> pd.DataFrame:
    """Build source-aware causal contrast features for v2."""
    frame = _juxtapose_engineering(source)
    source_columns = set(source.columns)
    if "timestamp" not in frame.columns and "timestamp" in source_columns:
        frame.insert(0, "timestamp", _as_datetime(source["timestamp"]))

    paired_signals = {
        "temperature": ("outside_openmeteo_temperature_2m", "outside_nasa_T2M"),
        "relative_humidity": ("outside_openmeteo_relative_humidity_2m", "outside_nasa_RH2M"),
        "precipitation": ("outside_openmeteo_precipitation", "outside_nasa_PRECTOTCORR"),
        "wind": ("outside_openmeteo_wind_speed_10m", "outside_nasa_WS10M"),
        "radiation": ("outside_openmeteo_shortwave_radiation", "outside_nasa_ALLSKY_SFC_SW_DWN"),
        "vpd": ("outside_openmeteo_vpd_kpa", "outside_nasa_vpd_kpa"),
    }
    for signal, (openmeteo_col, nasa_col) in paired_signals.items():
        if {openmeteo_col, nasa_col} <= source_columns:
            openmeteo = _numeric(source, openmeteo_col)
            nasa = _numeric(source, nasa_col)
            mean = pd.concat([openmeteo, nasa], axis=1).mean(axis=1)
            frame[f"juxtapose_v2__source_gap__{signal}"] = (openmeteo - nasa).abs()
            frame[f"juxtapose_v2__source_delta__{signal}"] = openmeteo - nasa
            frame[f"juxtapose_v2__source_ratio__{signal}"] = _safe_ratio(openmeteo, nasa)
            frame[f"juxtapose_v2__source_mean__{signal}"] = mean

    interior_exterior = {
        "temperature": ("air_temperature", "outside_fused_temperature_mean"),
        "relative_humidity": ("relative_humidity", "outside_fused_relative_humidity_mean"),
        "vpd": ("vpd", "outside_fused_vpd_mean"),
        "radiation": ("radiation_proxy", "outside_fused_shortwave_radiation_mean"),
    }
    fallback_exterior = {
        "temperature": ["outside_openmeteo_temperature_2m", "outside_nasa_T2M"],
        "relative_humidity": ["outside_openmeteo_relative_humidity_2m", "outside_nasa_RH2M"],
        "vpd": ["outside_openmeteo_vpd_kpa", "outside_nasa_vpd_kpa"],
        "radiation": ["outside_openmeteo_shortwave_radiation", "outside_nasa_ALLSKY_SFC_SW_DWN"],
    }
    for signal, (inside_col, fused_col) in interior_exterior.items():
        if inside_col not in source_columns:
            continue
        outside_col = fused_col if fused_col in source_columns else next(
            (candidate for candidate in fallback_exterior[signal] if candidate in source_columns),
            None,
        )
        if outside_col is None:
            continue
        inside = _numeric(source, inside_col)
        outside = _numeric(source, outside_col)
        frame[f"juxtapose_v2__inside_outside_delta__{signal}"] = inside - outside
        frame[f"juxtapose_v2__inside_outside_abs_gap__{signal}"] = (inside - outside).abs()

    causal_signals = {
        "rain": [
            "outside_fused_precipitation_mean",
            "outside_openmeteo_precipitation",
            "outside_nasa_PRECTOTCORR",
        ],
        "wind": [
            "outside_fused_wind_speed_10m_mean",
            "outside_openmeteo_wind_speed_10m",
            "outside_nasa_WS10M",
        ],
        "radiation": [
            "outside_fused_shortwave_radiation_mean",
            "outside_openmeteo_shortwave_radiation",
            "outside_nasa_ALLSKY_SFC_SW_DWN",
        ],
        "cloud": [
            "outside_fused_shortwave_radiation_mean",
            "outside_openmeteo_shortwave_radiation",
            "outside_nasa_ALLSKY_SFC_SW_DWN",
        ],
    }
    for signal, candidates in causal_signals.items():
        column = next((candidate for candidate in candidates if candidate in source_columns), None)
        if column is None:
            continue
        series = _numeric(source, column)
        short = series.rolling(window=12, min_periods=4).mean()
        long = series.rolling(window=72, min_periods=16).mean()
        frame[f"juxtapose_v2__{signal}__short_minus_long"] = short - long
        frame[f"juxtapose_v2__{signal}__short_over_long"] = _safe_ratio(short, long)
        frame[f"juxtapose_v2__{signal}__trend_1h"] = series - series.shift(12)
    return _clean_feature_frame(frame)


def estimate_task_relation_graph(
    targets: pd.DataFrame,
    *,
    max_lag_steps: int = 12,
) -> pd.DataFrame:
    """Estimate a compact task relation graph from correlation, co-occurrence, and lead-lag."""
    task_names = targets.columns.tolist()
    matrix = pd.DataFrame(np.eye(len(task_names)), index=task_names, columns=task_names, dtype=float)
    clean = targets.astype(float).reset_index(drop=True)
    for left in task_names:
        for right in task_names:
            if left == right:
                continue
            left_series = clean[left]
            right_series = clean[right]
            corr = abs(float(left_series.corr(right_series))) if left_series.nunique() > 1 and right_series.nunique() > 1 else 0.0
            both = float(((left_series == 1.0) & (right_series == 1.0)).mean())
            support = np.sqrt(max(float(left_series.mean()), 1e-6) * max(float(right_series.mean()), 1e-6))
            cooccurrence = float(np.clip(both / max(support, 1e-6), 0.0, 1.0))
            lead_lag = 0.0
            for lag in range(-max_lag_steps, max_lag_steps + 1):
                shifted = right_series.shift(lag)
                valid = left_series.notna() & shifted.notna()
                if valid.sum() < 8 or shifted[valid].nunique() < 2 or left_series[valid].nunique() < 2:
                    continue
                lead_lag = max(lead_lag, abs(float(left_series[valid].corr(shifted[valid]))))
            matrix.loc[left, right] = float(np.nan_to_num(np.mean([corr, cooccurrence, lead_lag])))
    return matrix


class JuxtaposeLGBMRelationV2Classifier(BaseModel):
    """Relation-aware Juxtapose v2 with guarded multitask transfer."""

    model_name = "juxtapose_lgbm_relation_v2"

    def __init__(
        self,
        *,
        calibration_method: str = "sigmoid",
        calibration_fraction: float = 0.2,
        threshold_metric: str = "f1",
        random_state: int = 42,
        task_order: Sequence[str] = TASK_ORDER,
        bridge_n_estimators: int = 72,
        relation_min_score: float = 0.08,
        cross_fit_splits: int = 4,
        use_source_aware_encoder: bool = True,
        use_relation_graph: bool = True,
        use_cross_fitted_bridge: bool = True,
        use_negative_transfer_guard: bool = True,
        single_task: bool = False,
        **estimator_params: Any,
    ) -> None:
        self.calibration_method = calibration_method
        self.calibration_fraction = calibration_fraction
        self.threshold_metric = threshold_metric
        self.random_state = random_state
        self.task_order = list(task_order)
        self.bridge_n_estimators = bridge_n_estimators
        self.relation_min_score = relation_min_score
        self.cross_fit_splits = cross_fit_splits
        self.use_source_aware_encoder = use_source_aware_encoder
        self.use_relation_graph = use_relation_graph
        self.use_cross_fitted_bridge = use_cross_fitted_bridge
        self.use_negative_transfer_guard = use_negative_transfer_guard
        self.single_task = single_task
        self.estimator_params = estimator_params

        self.active_task_: str | None = None
        self.task_columns_: list[str] = []
        self.feature_columns_: list[str] = []
        self.relation_columns_: dict[str, list[str]] = {}
        self.engineered_source_: pd.DataFrame | None = None
        self.imputer_: SimpleImputer | None = None
        self.local_estimators_: dict[str, Any] = {}
        self.relation_estimators_: dict[str, Any] = {}
        self.calibrators_: dict[str, BinaryProbabilityCalibrator] = {}
        self.thresholds_: dict[str, float] = {}
        self.gates_: dict[str, float] = {}
        self.relation_matrix_: pd.DataFrame | None = None
        self.relation_decisions_: dict[str, dict[str, Any]] = {}
        self.training_metadata: dict[str, Any] = {}

    def _prepare_features(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit or self.engineered_source_ is None:
            source = _feature_source_from_X(X)
            self.engineered_source_ = (
                _source_aware_engineering(source)
                if self.use_source_aware_encoder
                else _juxtapose_engineering(source)
            )
            combined = _base_feature_union(X, engineered=self.engineered_source_)
            self.feature_columns_ = [column for column in combined.columns if combined[column].notna().any()]
            return combined[self.feature_columns_]
        combined = _base_feature_union(X, engineered=self.engineered_source_)
        return combined.reindex(columns=self.feature_columns_, fill_value=float("nan"))

    def _estimator(self, *, relation: bool = False) -> Any:
        if LGBMClassifier is None:  # pragma: no cover - environment without lightgbm
            raise ImportError("LightGBM is required for juxtapose_lgbm_relation_v2.")
        params = dict(self.estimator_params)
        params.setdefault("objective", "binary")
        params.setdefault("n_estimators", self.bridge_n_estimators if relation else 160)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("num_leaves", 15 if relation else 31)
        params.setdefault("subsample", 0.9)
        params.setdefault("colsample_bytree", 0.85)
        params.setdefault("random_state", self.random_state)
        params.setdefault("verbosity", -1)
        params.setdefault("n_jobs", 1)
        return LGBMClassifier(**params)

    def _fit_estimator_or_prior(self, design: np.ndarray, target: pd.Series, *, relation: bool = False) -> Any:
        clean_target = pd.Series(target).astype(int).reset_index(drop=True)
        if clean_target.nunique() < 2:
            return ConstantProbabilityEstimator(float(clean_target.mean()))
        estimator = self._estimator(relation=relation)
        estimator.fit(design, clean_target.to_numpy(dtype=int))
        return estimator

    def _local_scores(self, design: np.ndarray) -> np.ndarray:
        columns: list[np.ndarray] = []
        for task in self.task_columns_:
            probabilities = self.local_estimators_[task].predict_proba(design)
            columns.append(np.asarray(probabilities)[:, 1].astype(float))
        return np.column_stack(columns)

    def _chronological_oof_scores(self, design: np.ndarray, targets: pd.DataFrame) -> tuple[np.ndarray, float]:
        if not self.use_cross_fitted_bridge or len(targets) < 48:
            return self._local_scores(design), 0.0
        n_rows = len(targets)
        scores = np.full((n_rows, len(self.task_columns_)), np.nan, dtype=float)
        first_start = max(16, n_rows // (self.cross_fit_splits + 1))
        edges = np.linspace(first_start, n_rows, self.cross_fit_splits + 1, dtype=int)
        for start, end in zip(edges[:-1], edges[1:]):
            if end <= start:
                continue
            train_index = np.arange(0, start)
            test_index = np.arange(start, end)
            for task_index, task in enumerate(self.task_columns_):
                estimator = self._fit_estimator_or_prior(
                    design[train_index],
                    targets[task].iloc[train_index],
                )
                scores[test_index, task_index] = np.asarray(estimator.predict_proba(design[test_index]))[:, 1]
        coverage = float(np.isfinite(scores).mean())
        priors = targets.mean(axis=0).to_numpy(dtype=float)
        missing_rows, missing_cols = np.where(~np.isfinite(scores))
        if len(missing_rows):
            scores[missing_rows, missing_cols] = priors[missing_cols]
        return scores, coverage

    def _relation_allowed(self, target_task: str, source_task: str) -> bool:
        if source_task == target_task:
            return False
        if not self.use_relation_graph or self.relation_matrix_ is None:
            return not self.single_task
        return float(self.relation_matrix_.loc[target_task, source_task]) >= self.relation_min_score

    def _relation_feature_frame(self, scores: np.ndarray, target_task: str) -> pd.DataFrame:
        frame = pd.DataFrame(index=range(scores.shape[0]))
        accepted_indices: list[int] = []
        for source_index, source_task in enumerate(self.task_columns_):
            if self._relation_allowed(target_task, source_task):
                accepted_indices.append(source_index)
                frame[f"juxtapose_v2_relation__prob__{source_task}"] = scores[:, source_index]
                if self.use_relation_graph and self.relation_matrix_ is not None:
                    weight = float(self.relation_matrix_.loc[target_task, source_task])
                    frame[f"juxtapose_v2_relation__weighted__{source_task}"] = scores[:, source_index] * weight
        if accepted_indices:
            related = scores[:, accepted_indices]
            frame["juxtapose_v2_relation__mean_prob"] = related.mean(axis=1)
            frame["juxtapose_v2_relation__max_prob"] = related.max(axis=1)
            frame["juxtapose_v2_relation__spread_prob"] = related.max(axis=1) - related.min(axis=1)
        return frame

    def _relation_design(self, design: np.ndarray, scores: np.ndarray, task: str, *, fit: bool) -> np.ndarray:
        relation_frame = self._relation_feature_frame(scores, task)
        if fit:
            self.relation_columns_[task] = relation_frame.columns.tolist()
        aligned = relation_frame.reindex(columns=self.relation_columns_.get(task, []), fill_value=0.5)
        if aligned.empty:
            return design
        return np.column_stack([design, aligned.to_numpy(dtype=float)])

    def _relation_scores(self, design: np.ndarray, scores: np.ndarray, *, fit: bool) -> np.ndarray:
        output = np.zeros((design.shape[0], len(self.task_columns_)), dtype=float)
        for task_index, task in enumerate(self.task_columns_):
            relation_design = self._relation_design(design, scores, task, fit=fit)
            estimator = self.relation_estimators_[task]
            output[:, task_index] = np.asarray(estimator.predict_proba(relation_design))[:, 1]
        return output

    def _metric_score(self, y_true: pd.Series, probability: np.ndarray) -> float:
        if y_true.nunique() < 2:
            return 0.0
        prob_frame = _positive_probability_frame(probability, y_true.index)
        threshold = select_optimal_threshold(y_true, prob_frame["class_1"], metric=self.threshold_metric)
        metrics = compute_classification_metrics(y_true, (prob_frame["class_1"] >= threshold).astype(int), prob_frame)
        return float(np.nan_to_num(metrics.get("pr_auc"), nan=0.0)) + float(np.nan_to_num(metrics.get("f1"), nan=0.0))

    def _calibrate_and_gate(
        self,
        targets: pd.DataFrame,
        local_scores: np.ndarray,
        relation_scores: np.ndarray,
        *,
        oof_coverage: float,
    ) -> dict[str, Any]:
        calibration_size = max(24, int(round(len(targets) * float(self.calibration_fraction))))
        calibration_size = min(calibration_size, max(len(targets) - 16, 0))
        calibration_payload: dict[str, Any] = {}
        for task_index, task in enumerate(self.task_columns_):
            y_task = targets[task].reset_index(drop=True)
            y_cal = y_task.iloc[-calibration_size:].reset_index(drop=True) if calibration_size > 0 else y_task
            local_cal = local_scores[-len(y_cal):, task_index]
            relation_cal = relation_scores[-len(y_cal):, task_index]
            local_score = self._metric_score(y_cal, local_cal)
            relation_score = self._metric_score(y_cal, relation_cal)
            relation_columns = self.relation_columns_.get(task, [])
            guard_rejects = (
                self.use_negative_transfer_guard
                and relation_columns
                and relation_score < local_score - 1e-9
            )
            candidate_gates = [0.0] if guard_rejects or not relation_columns else [0.0, 0.25, 0.5, 0.75, 1.0]
            best_gate = 0.0
            best_score = -np.inf
            for gate in candidate_gates:
                mixed = (1.0 - gate) * local_cal + gate * relation_cal
                score = self._metric_score(y_cal, mixed)
                if score > best_score:
                    best_score = score
                    best_gate = gate
            self.gates_[task] = float(best_gate)
            mixed_cal = (1.0 - best_gate) * local_cal + best_gate * relation_cal
            calibrator = BinaryProbabilityCalibrator(method=self.calibration_method).fit(mixed_cal, y_cal)
            self.calibrators_[task] = calibrator
            calibrated = calibrator.transform(mixed_cal)
            threshold = (
                float(select_optimal_threshold(y_cal, pd.Series(calibrated), metric=self.threshold_metric))
                if y_cal.nunique() >= 2
                else 0.5
            )
            self.thresholds_[task] = threshold
            calibrated_frame = _positive_probability_frame(calibrated, y_cal.index)
            self.relation_decisions_[task] = {
                "accepted": bool(best_gate > 0.0),
                "rejected_by_negative_transfer_guard": bool(guard_rejects),
                "gate": float(best_gate),
                "local_validation_score": float(local_score),
                "relation_validation_score": float(relation_score),
                "relation_feature_count": int(len(relation_columns)),
                "oof_bridge_coverage": float(oof_coverage),
            }
            calibration_payload[task] = {
                **self.relation_decisions_[task],
                "method": calibrator.kind,
                "threshold": threshold,
                "after_optimal": {
                    **compute_classification_metrics(
                        y_cal,
                        (calibrated_frame["class_1"] >= threshold).astype(int),
                        calibrated_frame,
                    ),
                    "threshold": threshold,
                },
            }
        return calibration_payload

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
    ) -> "JuxtaposeLGBMRelationV2Classifier":
        targets, active_task = _multi_task_targets(X, y, task_order=self.task_order)
        self.active_task_ = active_task
        if self.single_task:
            targets = pd.DataFrame({active_task: pd.Series(y).astype(int).reset_index(drop=True)})
        self.task_columns_ = targets.columns.tolist()
        features = self._prepare_features(X, fit=True)
        self.imputer_ = SimpleImputer(strategy="median")
        design = self.imputer_.fit_transform(features)
        self.relation_matrix_ = estimate_task_relation_graph(targets) if self.use_relation_graph else None

        for task in self.task_columns_:
            self.local_estimators_[task] = self._fit_estimator_or_prior(design, targets[task])
        local_scores = self._local_scores(design)
        bridge_scores, oof_coverage = self._chronological_oof_scores(design, targets)
        for task in self.task_columns_:
            relation_design = self._relation_design(design, bridge_scores, task, fit=True)
            self.relation_estimators_[task] = self._fit_estimator_or_prior(
                relation_design,
                targets[task],
                relation=True,
            )
        relation_scores = self._relation_scores(design, bridge_scores, fit=False)
        calibration_payload = self._calibrate_and_gate(
            targets,
            local_scores,
            relation_scores,
            oof_coverage=oof_coverage,
        )
        self.training_metadata = {
            "architecture": "Regalia-GBDT/Juxtapose v2",
            "architecture_blocks": [
                "source_aware_contrastive_encoder",
                "task_relation_graph",
                "cross_fitted_reliability_bridge",
                "negative_transfer_guard",
                "gated_calibrated_head",
            ],
            "variant_impl": "juxtapose_lgbm_relation_v2",
            "is_legacy_wrapper": False,
            "device_used": "cpu",
            "static_feature_count": int(len(self.feature_columns_)),
            "relation_feature_count": int(sum(len(value) for value in self.relation_columns_.values())),
            "parameter_count": self.estimator_params.get("n_estimators", 160),
            "active_task": self.active_task_,
            "known_tasks": self.task_columns_,
            "ablations": {
                "use_source_aware_encoder": self.use_source_aware_encoder,
                "use_relation_graph": self.use_relation_graph,
                "use_cross_fitted_bridge": self.use_cross_fitted_bridge,
                "use_negative_transfer_guard": self.use_negative_transfer_guard,
                "single_task": self.single_task,
            },
            "relation_matrix": self.relation_matrix_.to_dict() if self.relation_matrix_ is not None else None,
            "relation_decisions": self.relation_decisions_,
            "internal_calibration": calibration_payload,
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self._prepare_features(X, fit=False)
        design = self.imputer_.transform(features)
        local_scores = self._local_scores(design)
        relation_scores = self._relation_scores(design, local_scores, fit=False)
        active_index = self.task_columns_.index(self.active_task_)
        gate = self.gates_.get(self.active_task_ or "", 0.0)
        mixed = (1.0 - gate) * local_scores[:, active_index] + gate * relation_scores[:, active_index]
        calibrated = self.calibrators_[self.active_task_].transform(mixed)
        return _positive_probability_frame(calibrated, X.index)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(X)
        threshold = self.thresholds_.get(self.active_task_ or "", 0.5)
        return (probabilities["class_1"] >= threshold).astype(int).rename("prediction")


__all__ = ["JuxtaposeLGBMRelationV2Classifier", "estimate_task_relation_graph", "_source_aware_engineering"]
