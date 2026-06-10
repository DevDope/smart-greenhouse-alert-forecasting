"""GBDT variants that port stage3 ideas into tabular models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain

from ...data.io import read_dataframe
from ...evaluation.metrics_classification import compute_classification_metrics
from ...training.calibration import select_optimal_threshold
from ..base import BaseModel

try:
    from lightgbm import LGBMClassifier, LGBMRanker
except Exception:  # pragma: no cover - optional dependency fallback
    LGBMClassifier = None
    LGBMRanker = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None


TASK_ORDER = ("event_next_60m", "event_next_120m")
PRIMARY_SIGNALS = (
    "soil_moisture",
    "air_temperature",
    "relative_humidity",
    "radiation_proxy",
    "soil_temperature",
    "vpd",
    "light_lux",
    "irrigation_lph",
    "irrigation_event_observed",
    "ec25",
    "ph",
)

FRIPP_PRIORITY_COLUMNS = (
    "soil_moisture",
    "air_temperature",
    "relative_humidity",
    "radiation_proxy",
    "vpd",
    "irrigation_lph",
    "irrigation_event_observed",
    "soil_moisture__lag_1",
    "soil_moisture__lag_3",
    "soil_moisture__lag_6",
    "soil_moisture__lag_12",
    "soil_moisture__lag_24",
    "soil_moisture__roll_mean_3",
    "soil_moisture__roll_mean_12",
    "soil_moisture__roll_std_12",
    "soil_moisture__delta_1",
    "soil_moisture__delta_3",
    "radiation_proxy__roll_mean_3",
    "air_temperature__roll_mean_3",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "fripp__soil_moisture__lag_24",
    "fripp__soil_moisture__roll_mean_24",
    "fripp__soil_moisture__delta_24",
    "fripp__radiation_proxy__roll_mean_12",
    "fripp__air_temperature__roll_mean_12",
)


def _as_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _positive_probability_frame(probabilities: np.ndarray | pd.Series, index: pd.Index) -> pd.DataFrame:
    positive = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)
    return pd.DataFrame({"class_0": 1.0 - positive, "class_1": positive}, index=index)


def _clip_scores(scores: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.asarray(scores, dtype=float).reshape(-1)


def _reference_probabilities(scores: np.ndarray | Sequence[float]) -> np.ndarray:
    vector = _clip_scores(scores)
    finite = np.isfinite(vector)
    if not finite.all():
        vector = vector.copy()
        vector[~finite] = 0.0
    if vector.size == 0:
        return vector
    if float(np.nanmin(vector)) >= 0.0 and float(np.nanmax(vector)) <= 1.0:
        return np.clip(vector, 1e-6, 1.0 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip(vector, -20.0, 20.0)))


def _prepare_calibration_feature(scores: np.ndarray | Sequence[float]) -> np.ndarray:
    probabilities = _reference_probabilities(scores)
    if probabilities.size == 0:
        return probabilities.reshape(-1, 1)
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    transformed = np.log(clipped / (1.0 - clipped))
    return transformed.reshape(-1, 1)


def _rolling_quantile(series: pd.Series, window: int, quantile: float) -> pd.Series:
    return series.rolling(window=window, min_periods=max(3, window // 2)).quantile(quantile)


def _slope(series: pd.Series, steps: int) -> pd.Series:
    return (series - series.shift(steps)) / float(max(steps, 1))


def _available_signals(frame: pd.DataFrame, max_count: int | None = None) -> list[str]:
    ordered: list[str] = [name for name in PRIMARY_SIGNALS if name in frame.columns and frame[name].notna().any()]
    extras = [
        column
        for column in frame.columns
        if column not in {"timestamp", *ordered}
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().any()
    ]
    selected = ordered + extras
    return selected[:max_count] if max_count is not None else selected


def _default_timing_frame(X: pd.DataFrame) -> pd.DataFrame:
    timing = X.attrs.get("timing_frame")
    if isinstance(timing, pd.DataFrame) and "timestamp" in timing.columns:
        resolved = timing.reset_index(drop=True).copy()
        resolved["timestamp"] = _as_datetime(resolved["timestamp"])
        for column in ("observation_start", "observation_end", "label_end"):
            if column not in resolved.columns:
                resolved[column] = resolved["timestamp"]
            else:
                resolved[column] = _as_datetime(resolved[column])
        return resolved[["timestamp", "observation_start", "observation_end", "label_end"]]

    if "timestamp" in X.columns:
        timestamps = _as_datetime(X["timestamp"])
    else:
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=len(X), freq="5min"))
    return pd.DataFrame(
        {
            "timestamp": timestamps.reset_index(drop=True),
            "observation_start": timestamps.reset_index(drop=True),
            "observation_end": timestamps.reset_index(drop=True),
            "label_end": timestamps.reset_index(drop=True),
        }
    )


def _feature_source_from_X(X: pd.DataFrame) -> pd.DataFrame:
    metadata = X.attrs.get("prepared_metadata", {})
    canonical_cfg = metadata.get("canonical_dataset", {})
    canonical_path = canonical_cfg.get("path")
    if canonical_path:
        source = read_dataframe(canonical_path)
        if "timestamp" in source.columns:
            source = source.copy()
            source["timestamp"] = _as_datetime(source["timestamp"])
            return source.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    timing = _default_timing_frame(X)
    local = X.reset_index(drop=True).copy()
    if "timestamp" in local.columns:
        local = local.drop(columns=["timestamp"])
    local.insert(0, "timestamp", timing["timestamp"].to_numpy())
    return local


def _clean_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if column == "timestamp":
            cleaned[column] = _as_datetime(cleaned[column])
        else:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    return cleaned


def _base_feature_union(X: pd.DataFrame, engineered: pd.DataFrame | None = None) -> pd.DataFrame:
    base = X.reset_index(drop=True).copy()
    if "timestamp" in base.columns:
        base = base.drop(columns=["timestamp"])
    timing = _default_timing_frame(X)
    if engineered is None:
        combined = base
    else:
        extras = timing[["timestamp"]].merge(engineered, on="timestamp", how="left")
        combined = pd.concat([base, extras.drop(columns=["timestamp"])], axis=1)
    return _clean_feature_frame(combined)


def _patch_matrix(series: pd.Series, start: int, patch_size: int) -> pd.DataFrame:
    return pd.concat([series.shift(start + offset) for offset in range(1, patch_size + 1)], axis=1)


def _carey_engineering(source: pd.DataFrame, signals: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"timestamp": _as_datetime(source["timestamp"])})
    hour = frame["timestamp"].dt.hour + (frame["timestamp"].dt.minute / 60.0)
    hour_angle = (2.0 * np.pi * hour) / 24.0
    frame["carey__hour_sin_h2"] = np.sin(2.0 * hour_angle)
    frame["carey__hour_cos_h2"] = np.cos(2.0 * hour_angle)
    frame["carey__hour_sin_h3"] = np.sin(3.0 * hour_angle)
    frame["carey__hour_cos_h3"] = np.cos(3.0 * hour_angle)

    for column in signals:
        series = pd.to_numeric(source[column], errors="coerce")
        mu_48 = series.rolling(window=48, min_periods=12).mean()
        sd_48 = series.rolling(window=48, min_periods=12).std().replace(0.0, np.nan)
        frame[f"carey__{column}__mu_48"] = mu_48
        frame[f"carey__{column}__sd_48"] = sd_48
        for lag in (1, 2, 3, 6, 12, 24, 48):
            lagged = series.shift(lag)
            frame[f"carey__{column}__lag_{lag}"] = lagged
            frame[f"carey__{column}__lag_norm_{lag}"] = (lagged - mu_48) / sd_48
        for window in (6, 12, 24, 48):
            rolled = series.rolling(window=window, min_periods=max(3, window // 2))
            frame[f"carey__{column}__roll_mean_{window}"] = rolled.mean()
            frame[f"carey__{column}__roll_std_{window}"] = rolled.std()
            frame[f"carey__{column}__roll_min_{window}"] = rolled.min()
            frame[f"carey__{column}__roll_max_{window}"] = rolled.max()
            frame[f"carey__{column}__roll_q25_{window}"] = _rolling_quantile(series, window, 0.25)
            frame[f"carey__{column}__roll_q50_{window}"] = _rolling_quantile(series, window, 0.50)
            frame[f"carey__{column}__roll_q75_{window}"] = _rolling_quantile(series, window, 0.75)
            frame[f"carey__{column}__delta_{window}"] = series - series.shift(window)
            frame[f"carey__{column}__slope_{window}"] = _slope(series, window)
    return _clean_feature_frame(frame)


def _wilson_engineering(source: pd.DataFrame, signals: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"timestamp": _as_datetime(source["timestamp"])})
    for column in signals:
        series = pd.to_numeric(source[column], errors="coerce")
        mean_12 = series.rolling(window=12, min_periods=6).mean()
        mean_48 = series.rolling(window=48, min_periods=12).mean()
        frame[f"wilson__{column}__trend_12_48"] = mean_12 - mean_48
        frame[f"wilson__{column}__resid_48"] = series - mean_48
        frame[f"wilson__{column}__slope_12"] = _slope(series, 12)
        frame[f"wilson__{column}__slope_48"] = _slope(series, 48)
        for window in (3, 6, 12, 24, 48):
            rolled = series.rolling(window=window, min_periods=max(2, window // 2))
            frame[f"wilson__{column}__roll_mean_{window}"] = rolled.mean()
            frame[f"wilson__{column}__roll_std_{window}"] = rolled.std()
            frame[f"wilson__{column}__roll_min_{window}"] = rolled.min()
            frame[f"wilson__{column}__roll_max_{window}"] = rolled.max()
        for patch_size, patch_count in ((12, 4), (6, 8)):
            for patch_index in range(patch_count):
                patch = _patch_matrix(series, start=patch_index * patch_size, patch_size=patch_size)
                label = patch_index + 1
                frame[f"wilson__{column}__patch{patch_count}x{patch_size}_mean_p{label}"] = patch.mean(axis=1)
            frame[f"wilson__{column}__patch{patch_count}x{patch_size}_swing"] = (
                frame[f"wilson__{column}__patch{patch_count}x{patch_size}_mean_p1"]
                - frame[f"wilson__{column}__patch{patch_count}x{patch_size}_mean_p{patch_count}"]
            )
        for every, window in ((2, 24), (4, 48)):
            sampled = pd.concat([series.shift(offset) for offset in range(0, window, every)], axis=1)
            frame[f"wilson__{column}__sample{every}_mean_{window}"] = sampled.mean(axis=1)
            frame[f"wilson__{column}__sample{every}_std_{window}"] = sampled.std(axis=1)
    return _clean_feature_frame(frame)


def _fripp_engineering(source: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"timestamp": _as_datetime(source["timestamp"])})
    if "soil_moisture" in source.columns:
        soil = pd.to_numeric(source["soil_moisture"], errors="coerce")
        frame["fripp__soil_moisture__lag_24"] = soil.shift(24)
        frame["fripp__soil_moisture__roll_mean_24"] = soil.rolling(window=24, min_periods=8).mean()
        frame["fripp__soil_moisture__delta_24"] = soil - soil.shift(24)
    if "radiation_proxy" in source.columns:
        radiation = pd.to_numeric(source["radiation_proxy"], errors="coerce")
        frame["fripp__radiation_proxy__roll_mean_12"] = radiation.rolling(window=12, min_periods=6).mean()
    if "air_temperature" in source.columns:
        temperature = pd.to_numeric(source["air_temperature"], errors="coerce")
        frame["fripp__air_temperature__roll_mean_12"] = temperature.rolling(window=12, min_periods=6).mean()
    return _clean_feature_frame(frame)


def _zappa_engineering(source: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"timestamp": _as_datetime(source["timestamp"])})
    if {"soil_moisture", "radiation_proxy"} <= set(source.columns):
        soil = pd.to_numeric(source["soil_moisture"], errors="coerce")
        radiation = pd.to_numeric(source["radiation_proxy"], errors="coerce")
        frame["zappa__soil_radiation_gap"] = soil - radiation.rolling(window=12, min_periods=6).mean()
        frame["zappa__soil_trend_12"] = _slope(soil, 12)
    if {"air_temperature", "relative_humidity"} <= set(source.columns):
        temperature = pd.to_numeric(source["air_temperature"], errors="coerce")
        humidity = pd.to_numeric(source["relative_humidity"], errors="coerce")
        frame["zappa__stress_proxy"] = temperature * (100.0 - humidity)
    if "irrigation_lph" in source.columns:
        irrigation = pd.to_numeric(source["irrigation_lph"], errors="coerce")
        frame["zappa__irrigation_sum_12"] = irrigation.rolling(window=12, min_periods=3).sum()
    return _clean_feature_frame(frame)


def _fripp_selected_columns(combined: pd.DataFrame) -> list[str]:
    selected = [column for column in FRIPP_PRIORITY_COLUMNS if column in combined.columns]
    return selected or combined.columns.tolist()


def _monotone_value(column: str) -> int:
    if "soil_moisture" in column:
        return -1
    if "irrigation_lph" in column or "irrigation_event_observed" in column:
        return 1
    return 0


def _fripp_constraints(feature_names: Sequence[str]) -> tuple[list[int], list[list[int]]]:
    monotone = [_monotone_value(column) for column in feature_names]
    water_group = [
        index
        for index, name in enumerate(feature_names)
        if any(token in name for token in ("soil_moisture", "irrigation_lph", "irrigation_event_observed", "vpd"))
    ]
    climate_group = [
        index
        for index, name in enumerate(feature_names)
        if any(token in name for token in ("air_temperature", "relative_humidity", "radiation_proxy", "light_lux"))
    ]
    groups = [group for group in (water_group, climate_group) if group]
    return monotone, groups


@dataclass
class BinaryProbabilityCalibrator:
    method: str = "sigmoid"

    def __post_init__(self) -> None:
        self.kind = "reference"
        self.model: Any | None = None

    def fit(self, scores: np.ndarray | Sequence[float], y_true: pd.Series) -> "BinaryProbabilityCalibrator":
        target = pd.Series(y_true).astype(int)
        if len(target) < 16 or target.nunique() < 2:
            self.kind = "reference"
            self.model = None
            return self

        X_fit = _prepare_calibration_feature(scores)
        if self.method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(X_fit[:, 0], target.to_numpy())
            self.kind = "isotonic"
            self.model = model
            return self

        model = LogisticRegression(max_iter=1_000, class_weight="balanced")
        model.fit(X_fit, target.to_numpy())
        self.kind = "sigmoid"
        self.model = model
        return self

    def transform(self, scores: np.ndarray | Sequence[float]) -> np.ndarray:
        reference = _reference_probabilities(scores)
        if self.kind == "reference" or self.model is None:
            return reference

        X_score = _prepare_calibration_feature(scores)
        if self.kind == "isotonic":
            calibrated = np.asarray(self.model.predict(X_score[:, 0]), dtype=float)
            return np.clip(calibrated, 1e-6, 1.0 - 1e-6)
        calibrated = self.model.predict_proba(X_score)[:, 1]
        return np.clip(calibrated, 1e-6, 1.0 - 1e-6)


class FeatureAwareBinaryGBDT(BaseModel):
    """Shared wrapper for single-output binary GBDT variants."""

    model_name = "feature_aware_gbdt"

    def __init__(
        self,
        *,
        calibration_method: str = "sigmoid",
        calibration_fraction: float = 0.2,
        threshold_metric: str = "f1",
        random_state: int = 42,
        max_source_signals: int = 6,
        **estimator_params: Any,
    ) -> None:
        self.calibration_method = calibration_method
        self.calibration_fraction = calibration_fraction
        self.threshold_metric = threshold_metric
        self.random_state = random_state
        self.max_source_signals = max_source_signals
        self.estimator_params = estimator_params

        self.active_task_: str | None = None
        self.source_frame_: pd.DataFrame | None = None
        self.engineered_source_: pd.DataFrame | None = None
        self.feature_columns_: list[str] = []
        self.threshold_: float = 0.5
        self.imputer_: SimpleImputer | None = None
        self.estimator_: Any | None = None
        self.calibrator_: BinaryProbabilityCalibrator | None = None
        self.training_metadata: dict[str, Any] = {}

    def _build_engineered_source(self, source_frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"timestamp": _as_datetime(source_frame["timestamp"])})

    def _select_feature_columns(self, combined: pd.DataFrame) -> list[str]:
        return combined.columns.tolist()

    def _build_estimator(self, feature_names: Sequence[str]) -> Any:
        raise NotImplementedError

    def _raw_positive_scores(self, estimator: Any, features: np.ndarray) -> np.ndarray:
        probabilities = estimator.predict_proba(features)
        if probabilities.ndim == 1:
            return probabilities.astype(float)
        return probabilities[:, 1].astype(float)

    def _fit_calibration(self, y_true: pd.Series, raw_scores: np.ndarray) -> dict[str, Any]:
        target = pd.Series(y_true).astype(int).reset_index(drop=True)
        calibration_size = max(24, int(round(len(target) * float(self.calibration_fraction))))
        calibration_size = min(calibration_size, max(len(target) - 16, 0))
        if calibration_size <= 0:
            self.calibrator_ = BinaryProbabilityCalibrator(method=self.calibration_method)
            self.threshold_ = 0.5
            return {"method": "reference", "threshold": self.threshold_}

        y_cal = target.iloc[-calibration_size:].reset_index(drop=True)
        score_cal = np.asarray(raw_scores[-calibration_size:], dtype=float)
        self.calibrator_ = BinaryProbabilityCalibrator(method=self.calibration_method).fit(score_cal, y_cal)
        raw_prob = _positive_probability_frame(_reference_probabilities(score_cal), y_cal.index)
        calibrated = _positive_probability_frame(self.calibrator_.transform(score_cal), y_cal.index)
        if y_cal.nunique() >= 2:
            self.threshold_ = float(
                select_optimal_threshold(
                    y_cal,
                    calibrated["class_1"],
                    metric=self.threshold_metric,
                )
            )
        else:
            self.threshold_ = 0.5

        raw_pred = (raw_prob["class_1"] >= 0.5).astype(int)
        calibrated_default = (calibrated["class_1"] >= 0.5).astype(int)
        calibrated_opt = (calibrated["class_1"] >= self.threshold_).astype(int)
        return {
            "method": self.calibrator_.kind,
            "fraction": float(self.calibration_fraction),
            "threshold_metric": self.threshold_metric,
            "threshold": self.threshold_,
            "calibration_rows": int(calibration_size),
            "before": compute_classification_metrics(y_cal, raw_pred, raw_prob),
            "after_default": compute_classification_metrics(y_cal, calibrated_default, calibrated),
            "after_optimal": {
                **compute_classification_metrics(y_cal, calibrated_opt, calibrated),
                "threshold": self.threshold_,
            },
        }

    def _prepare_features(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit or self.engineered_source_ is None:
            self.source_frame_ = _feature_source_from_X(X)
            signals = _available_signals(self.source_frame_, max_count=self.max_source_signals)
            self.engineered_source_ = self._build_engineered_source(self.source_frame_[["timestamp", *signals]].copy())

        combined = _base_feature_union(X, engineered=self.engineered_source_)
        if fit:
            selected = [
                column
                for column in self._select_feature_columns(combined)
                if combined[column].notna().any()
            ]
            self.feature_columns_ = selected
        aligned = combined.reindex(columns=self.feature_columns_, fill_value=np.nan)
        return _clean_feature_frame(aligned)

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series | None = None) -> "FeatureAwareBinaryGBDT":
        self.active_task_ = str(X.attrs.get("task_name", self.active_task_ or "event_next_60m"))
        features = self._prepare_features(X, fit=True)
        target = pd.Series(y).astype(int).reset_index(drop=True)
        self.imputer_ = SimpleImputer(strategy="median")
        design = self.imputer_.fit_transform(features)
        self.estimator_ = self._build_estimator(self.feature_columns_)
        self.estimator_.fit(design, target.to_numpy())
        raw_scores = self._raw_positive_scores(self.estimator_, design)
        calibration = self._fit_calibration(target, raw_scores)
        self.training_metadata = {
            "device_used": "cpu",
            "static_feature_count": int(len(self.feature_columns_)),
            "parameter_count": self.estimator_params.get("n_estimators"),
            "active_task": self.active_task_,
            "internal_calibration": calibration,
            "feature_columns": self.feature_columns_[:64],
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self._prepare_features(X, fit=False)
        design = self.imputer_.transform(features)
        raw_scores = self._raw_positive_scores(self.estimator_, design)
        calibrated = self.calibrator_.transform(raw_scores) if self.calibrator_ is not None else _reference_probabilities(raw_scores)
        return _positive_probability_frame(calibrated, X.index)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(X)
        prediction = (probabilities["class_1"] >= self.threshold_).astype(int)
        return prediction.rename("prediction")


class CareyLGBMFourierClassifier(FeatureAwareBinaryGBDT):
    model_name = "carey_lgbm_fourier"

    def _build_engineered_source(self, source_frame: pd.DataFrame) -> pd.DataFrame:
        return _carey_engineering(source_frame, _available_signals(source_frame, max_count=self.max_source_signals))

    def _build_estimator(self, feature_names: Sequence[str]) -> Any:
        if LGBMClassifier is None:  # pragma: no cover - environment without lightgbm
            raise ImportError("LightGBM is required for carey_lgbm_fourier.")
        params = dict(self.estimator_params)
        params.setdefault("objective", "binary")
        params.setdefault("n_estimators", 180)
        params.setdefault("learning_rate", 0.04)
        params.setdefault("num_leaves", 31)
        params.setdefault("subsample", 0.9)
        params.setdefault("colsample_bytree", 0.85)
        params.setdefault("random_state", self.random_state)
        params.setdefault("verbosity", -1)
        params.setdefault("n_jobs", 1)
        return LGBMClassifier(**params)


class CareyXGBFourierClassifier(FeatureAwareBinaryGBDT):
    model_name = "carey_xgb_fourier"

    def _build_engineered_source(self, source_frame: pd.DataFrame) -> pd.DataFrame:
        return _carey_engineering(source_frame, _available_signals(source_frame, max_count=self.max_source_signals))

    def _build_estimator(self, feature_names: Sequence[str]) -> Any:
        if XGBClassifier is None:  # pragma: no cover - environment without xgboost
            raise ImportError("XGBoost is required for carey_xgb_fourier.")
        params = dict(self.estimator_params)
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "logloss")
        params.setdefault("tree_method", "hist")
        params.setdefault("n_estimators", 180)
        params.setdefault("max_depth", 5)
        params.setdefault("learning_rate", 0.04)
        params.setdefault("subsample", 0.9)
        params.setdefault("colsample_bytree", 0.85)
        params.setdefault("random_state", self.random_state)
        params.setdefault("n_jobs", 1)
        return XGBClassifier(**params)


class FrippXGBSparseClassifier(FeatureAwareBinaryGBDT):
    model_name = "fripp_xgb_sparse"

    def _build_engineered_source(self, source_frame: pd.DataFrame) -> pd.DataFrame:
        return _fripp_engineering(source_frame)

    def _select_feature_columns(self, combined: pd.DataFrame) -> list[str]:
        return _fripp_selected_columns(combined)

    def _build_estimator(self, feature_names: Sequence[str]) -> Any:
        if XGBClassifier is None:  # pragma: no cover - environment without xgboost
            raise ImportError("XGBoost is required for fripp_xgb_sparse.")
        params = dict(self.estimator_params)
        monotone, _ = _fripp_constraints(feature_names)
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "logloss")
        params.setdefault("tree_method", "hist")
        params.setdefault("n_estimators", 120)
        params.setdefault("max_depth", 3)
        params.setdefault("min_child_weight", 6)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("subsample", 0.75)
        params.setdefault("colsample_bytree", 0.65)
        params.setdefault("reg_alpha", 1.5)
        params.setdefault("reg_lambda", 3.0)
        params.setdefault("gamma", 0.2)
        params.setdefault("random_state", self.random_state)
        params.setdefault("n_jobs", 1)
        params.setdefault("monotone_constraints", f"({','.join(str(value) for value in monotone)})")
        return XGBClassifier(**params)


class FrippLGBMSparseClassifier(FeatureAwareBinaryGBDT):
    model_name = "fripp_lgbm_sparse"

    def _build_engineered_source(self, source_frame: pd.DataFrame) -> pd.DataFrame:
        return _fripp_engineering(source_frame)

    def _select_feature_columns(self, combined: pd.DataFrame) -> list[str]:
        return _fripp_selected_columns(combined)

    def _build_estimator(self, feature_names: Sequence[str]) -> Any:
        if LGBMClassifier is None:  # pragma: no cover - environment without lightgbm
            raise ImportError("LightGBM is required for fripp_lgbm_sparse.")
        params = dict(self.estimator_params)
        monotone, groups = _fripp_constraints(feature_names)
        params.setdefault("objective", "binary")
        params.setdefault("n_estimators", 140)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("num_leaves", 15)
        params.setdefault("max_depth", 4)
        params.setdefault("min_child_samples", 30)
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


class WilsonXGBMultiscaleClassifier(FeatureAwareBinaryGBDT):
    model_name = "wilson_xgb_multiscale"

    def _build_engineered_source(self, source_frame: pd.DataFrame) -> pd.DataFrame:
        return _wilson_engineering(source_frame, _available_signals(source_frame, max_count=4))

    def _build_estimator(self, feature_names: Sequence[str]) -> Any:
        if XGBClassifier is None:  # pragma: no cover - environment without xgboost
            raise ImportError("XGBoost is required for wilson_xgb_multiscale.")
        params = dict(self.estimator_params)
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "logloss")
        params.setdefault("tree_method", "hist")
        params.setdefault("n_estimators", 160)
        params.setdefault("max_depth", 4)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("subsample", 0.85)
        params.setdefault("colsample_bytree", 0.8)
        params.setdefault("random_state", self.random_state)
        params.setdefault("n_jobs", 1)
        return XGBClassifier(**params)


class WilsonLGBMMultiscaleClassifier(FeatureAwareBinaryGBDT):
    model_name = "wilson_lgbm_multiscale"

    def _build_engineered_source(self, source_frame: pd.DataFrame) -> pd.DataFrame:
        return _wilson_engineering(source_frame, _available_signals(source_frame, max_count=4))

    def _build_estimator(self, feature_names: Sequence[str]) -> Any:
        if LGBMClassifier is None:  # pragma: no cover - environment without lightgbm
            raise ImportError("LightGBM is required for wilson_lgbm_multiscale.")
        params = dict(self.estimator_params)
        params.setdefault("objective", "binary")
        params.setdefault("n_estimators", 160)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("num_leaves", 31)
        params.setdefault("subsample", 0.85)
        params.setdefault("colsample_bytree", 0.8)
        params.setdefault("random_state", self.random_state)
        params.setdefault("verbosity", -1)
        params.setdefault("n_jobs", 1)
        return LGBMClassifier(**params)


def _resolve_task_order(active_task: str, auxiliary_targets: pd.DataFrame | None, configured_order: Sequence[str]) -> list[str]:
    ordered = [task for task in configured_order if task == active_task or (auxiliary_targets is not None and task in auxiliary_targets.columns)]
    if active_task not in ordered:
        ordered.insert(0, active_task)
    if auxiliary_targets is not None:
        for column in auxiliary_targets.columns:
            if column not in ordered and auxiliary_targets[column].notna().all():
                ordered.append(column)
    return ordered


def _multi_task_targets(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task_order: Sequence[str],
) -> tuple[pd.DataFrame, str]:
    active_task = str(X.attrs.get("task_name", "event_next_60m"))
    auxiliary = X.attrs.get("auxiliary_targets")
    auxiliary_frame = auxiliary.copy() if isinstance(auxiliary, pd.DataFrame) else pd.DataFrame(index=range(len(X)))
    ordered_tasks = _resolve_task_order(active_task, auxiliary_frame, task_order)
    target_frame = pd.DataFrame(index=range(len(X)))
    for task in ordered_tasks:
        if task == active_task:
            target_frame[task] = pd.Series(y).astype(int).reset_index(drop=True)
        elif task in auxiliary_frame.columns:
            series = pd.Series(auxiliary_frame[task]).reset_index(drop=True)
            if series.notna().all():
                target_frame[task] = series.astype(int)
    return target_frame.dropna(axis=1, how="any"), active_task


def _xgb_multitask_positive_scores(estimator: Any, features: np.ndarray, task_count: int) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    if isinstance(probabilities, list):
        return np.column_stack([np.asarray(task_prob)[:, 1] for task_prob in probabilities])
    array = np.asarray(probabilities)
    if array.ndim == 3 and array.shape[-1] == 2:
        return array[:, :, 1]
    if array.ndim == 2 and array.shape[1] == task_count:
        return array.astype(float)
    if array.ndim == 2 and array.shape[1] == 2 and task_count == 1:
        return array[:, 1].reshape(-1, 1).astype(float)
    return array.reshape(len(features), task_count).astype(float)


def _chain_positive_scores(estimator: Any, features: np.ndarray, task_count: int) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    array = np.asarray(probabilities)
    if array.ndim == 2 and array.shape[1] == task_count:
        return array.astype(float)
    if array.ndim == 3 and array.shape[-1] == 2:
        return array[:, :, 1].astype(float)
    if isinstance(probabilities, list):
        return np.column_stack([np.asarray(task_prob)[:, 1] for task_prob in probabilities])
    return array.reshape(len(features), task_count).astype(float)


class ZappaXGBMultiOutputClassifier(BaseModel):
    model_name = "zappa_xgb_multioutput"

    def __init__(
        self,
        *,
        calibration_method: str = "sigmoid",
        calibration_fraction: float = 0.2,
        threshold_metric: str = "f1",
        random_state: int = 42,
        task_order: Sequence[str] = TASK_ORDER,
        **estimator_params: Any,
    ) -> None:
        self.calibration_method = calibration_method
        self.calibration_fraction = calibration_fraction
        self.threshold_metric = threshold_metric
        self.random_state = random_state
        self.task_order = list(task_order)
        self.estimator_params = estimator_params

        self.active_task_: str | None = None
        self.task_columns_: list[str] = []
        self.feature_columns_: list[str] = []
        self.imputer_: SimpleImputer | None = None
        self.estimator_: Any | None = None
        self.calibrators_: dict[str, BinaryProbabilityCalibrator] = {}
        self.thresholds_: dict[str, float] = {}
        self.engineered_source_: pd.DataFrame | None = None
        self.training_metadata: dict[str, Any] = {}

    def _prepare_features(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit or self.engineered_source_ is None:
            source = _feature_source_from_X(X)
            engineered = _zappa_engineering(source)
            self.engineered_source_ = engineered
            combined = _base_feature_union(X, engineered=engineered)
            self.feature_columns_ = [column for column in combined.columns if combined[column].notna().any()]
            return combined[self.feature_columns_]
        combined = _base_feature_union(X, engineered=self.engineered_source_)
        return combined.reindex(columns=self.feature_columns_, fill_value=np.nan)

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series | None = None) -> "ZappaXGBMultiOutputClassifier":
        if XGBClassifier is None:  # pragma: no cover - environment without xgboost
            raise ImportError("XGBoost is required for zappa_xgb_multioutput.")
        targets, active_task = _multi_task_targets(X, y, task_order=self.task_order)
        self.active_task_ = active_task
        self.task_columns_ = targets.columns.tolist()
        features = self._prepare_features(X, fit=True)
        self.imputer_ = SimpleImputer(strategy="median")
        design = self.imputer_.fit_transform(features)
        params = dict(self.estimator_params)
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "logloss")
        params.setdefault("tree_method", "hist")
        params.setdefault("multi_strategy", "one_output_per_tree")
        params.setdefault("n_estimators", 180)
        params.setdefault("max_depth", 4)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("subsample", 0.9)
        params.setdefault("colsample_bytree", 0.85)
        params.setdefault("random_state", self.random_state)
        params.setdefault("n_jobs", 1)
        self.estimator_ = XGBClassifier(**params)
        self.estimator_.fit(design, targets.to_numpy(dtype=int))

        raw_scores = _xgb_multitask_positive_scores(self.estimator_, design, len(self.task_columns_))
        calibration_size = max(24, int(round(len(targets) * float(self.calibration_fraction))))
        calibration_size = min(calibration_size, max(len(targets) - 16, 0))
        calibration_payload: dict[str, Any] = {}
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
            if y_cal.nunique() >= 2:
                threshold = float(select_optimal_threshold(y_cal, pd.Series(calibrated), metric=self.threshold_metric))
            else:
                threshold = 0.5
            self.thresholds_[task_name] = threshold
            raw_frame = _positive_probability_frame(_reference_probabilities(score_cal), y_cal.index)
            calibrated_frame = _positive_probability_frame(calibrated, y_cal.index)
            calibration_payload[task_name] = {
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
            }
        self.training_metadata = {
            "device_used": "cpu",
            "static_feature_count": int(len(self.feature_columns_)),
            "parameter_count": self.estimator_params.get("n_estimators", 180),
            "active_task": self.active_task_,
            "known_tasks": self.task_columns_,
            "internal_calibration": calibration_payload,
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self._prepare_features(X, fit=False)
        design = self.imputer_.transform(features)
        raw_scores = _xgb_multitask_positive_scores(self.estimator_, design, len(self.task_columns_))
        active_index = self.task_columns_.index(self.active_task_)
        task_scores = raw_scores[:, active_index]
        calibrated = self.calibrators_[self.active_task_].transform(task_scores)
        return _positive_probability_frame(calibrated, X.index)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(X)
        threshold = self.thresholds_.get(self.active_task_ or "", 0.5)
        return (probabilities["class_1"] >= threshold).astype(int).rename("prediction")


class ZappaLGBMChainClassifier(BaseModel):
    model_name = "zappa_lgbm_chain"

    def __init__(
        self,
        *,
        calibration_method: str = "sigmoid",
        calibration_fraction: float = 0.2,
        threshold_metric: str = "f1",
        random_state: int = 42,
        task_order: Sequence[str] = TASK_ORDER,
        **estimator_params: Any,
    ) -> None:
        self.calibration_method = calibration_method
        self.calibration_fraction = calibration_fraction
        self.threshold_metric = threshold_metric
        self.random_state = random_state
        self.task_order = list(task_order)
        self.estimator_params = estimator_params

        self.active_task_: str | None = None
        self.task_columns_: list[str] = []
        self.feature_columns_: list[str] = []
        self.engineered_source_: pd.DataFrame | None = None
        self.imputer_: SimpleImputer | None = None
        self.estimator_: Any | None = None
        self.calibrators_: dict[str, BinaryProbabilityCalibrator] = {}
        self.thresholds_: dict[str, float] = {}
        self.training_metadata: dict[str, Any] = {}

    def _prepare_features(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit or self.engineered_source_ is None:
            source = _feature_source_from_X(X)
            engineered = _zappa_engineering(source)
            self.engineered_source_ = engineered
            combined = _base_feature_union(X, engineered=engineered)
            self.feature_columns_ = [column for column in combined.columns if combined[column].notna().any()]
            return combined[self.feature_columns_]
        combined = _base_feature_union(X, engineered=self.engineered_source_)
        return combined.reindex(columns=self.feature_columns_, fill_value=np.nan)

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series | None = None) -> "ZappaLGBMChainClassifier":
        if LGBMClassifier is None:  # pragma: no cover - environment without lightgbm
            raise ImportError("LightGBM is required for zappa_lgbm_chain.")
        targets, active_task = _multi_task_targets(X, y, task_order=self.task_order)
        self.active_task_ = active_task
        self.task_columns_ = targets.columns.tolist()
        features = self._prepare_features(X, fit=True)
        self.imputer_ = SimpleImputer(strategy="median")
        design = self.imputer_.fit_transform(features)
        params = dict(self.estimator_params)
        params.setdefault("objective", "binary")
        params.setdefault("n_estimators", 160)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("num_leaves", 31)
        params.setdefault("subsample", 0.9)
        params.setdefault("colsample_bytree", 0.85)
        params.setdefault("random_state", self.random_state)
        params.setdefault("verbosity", -1)
        params.setdefault("n_jobs", 1)
        order_indices = [self.task_columns_.index(task) for task in self.task_columns_]
        base_estimator = LGBMClassifier(**params)
        self.estimator_ = ClassifierChain(base_estimator=base_estimator, order=order_indices)
        self.estimator_.fit(design, targets.to_numpy(dtype=int))

        raw_scores = _chain_positive_scores(self.estimator_, design, len(self.task_columns_))
        calibration_size = max(24, int(round(len(targets) * float(self.calibration_fraction))))
        calibration_size = min(calibration_size, max(len(targets) - 16, 0))
        calibration_payload: dict[str, Any] = {}
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
            threshold = float(select_optimal_threshold(y_cal, pd.Series(calibrated), metric=self.threshold_metric)) if y_cal.nunique() >= 2 else 0.5
            self.thresholds_[task_name] = threshold
            raw_frame = _positive_probability_frame(_reference_probabilities(score_cal), y_cal.index)
            calibrated_frame = _positive_probability_frame(calibrated, y_cal.index)
            calibration_payload[task_name] = {
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
            }
        self.training_metadata = {
            "device_used": "cpu",
            "static_feature_count": int(len(self.feature_columns_)),
            "parameter_count": self.estimator_params.get("n_estimators", 160),
            "active_task": self.active_task_,
            "known_tasks": self.task_columns_,
            "chain_order": self.task_columns_,
            "internal_calibration": calibration_payload,
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self._prepare_features(X, fit=False)
        design = self.imputer_.transform(features)
        raw_scores = _chain_positive_scores(self.estimator_, design, len(self.task_columns_))
        active_index = self.task_columns_.index(self.active_task_)
        task_scores = raw_scores[:, active_index]
        calibrated = self.calibrators_[self.active_task_].transform(task_scores)
        return _positive_probability_frame(calibrated, X.index)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(X)
        threshold = self.thresholds_.get(self.active_task_ or "", 0.5)
        return (probabilities["class_1"] >= threshold).astype(int).rename("prediction")


def _query_keys(X: pd.DataFrame, frequency: str) -> pd.Series:
    timing = _default_timing_frame(X)
    base = timing["label_end"] if "label_end" in timing.columns else timing["timestamp"]
    return _as_datetime(base).dt.floor(str(frequency).lower())


def _group_sizes(keys: pd.Series) -> list[int]:
    ordered = pd.Series(keys).reset_index(drop=True)
    return ordered.groupby(ordered, sort=False).size().astype(int).tolist()


def _informative_group_mask(y: pd.Series, keys: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"key": keys.reset_index(drop=True), "y": pd.Series(y).astype(int).reset_index(drop=True)})
    informative = frame.groupby("key")["y"].transform(lambda values: values.nunique() >= 2 and len(values) >= 2)
    return informative.astype(bool)


class ZappaLGBMRankerClassifier(BaseModel):
    model_name = "zappa_lgbm_ranker"

    def __init__(
        self,
        *,
        calibration_method: str = "isotonic",
        calibration_fraction: float = 0.2,
        threshold_metric: str = "f1",
        random_state: int = 42,
        query_frequency: str = "6h",
        **estimator_params: Any,
    ) -> None:
        self.calibration_method = calibration_method
        self.calibration_fraction = calibration_fraction
        self.threshold_metric = threshold_metric
        self.random_state = random_state
        self.query_frequency = query_frequency
        self.estimator_params = estimator_params

        self.active_task_: str | None = None
        self.feature_columns_: list[str] = []
        self.engineered_source_: pd.DataFrame | None = None
        self.imputer_: SimpleImputer | None = None
        self.ranker_: Any | None = None
        self.fallback_classifier_: Any | None = None
        self.calibrator_: BinaryProbabilityCalibrator | None = None
        self.threshold_: float = 0.5
        self.training_metadata: dict[str, Any] = {}

    def _prepare_features(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit or self.engineered_source_ is None:
            source = _feature_source_from_X(X)
            engineered = _zappa_engineering(source)
            self.engineered_source_ = engineered
            combined = _base_feature_union(X, engineered=engineered)
            self.feature_columns_ = [column for column in combined.columns if combined[column].notna().any()]
            return combined[self.feature_columns_]
        combined = _base_feature_union(X, engineered=self.engineered_source_)
        return combined.reindex(columns=self.feature_columns_, fill_value=np.nan)

    def _raw_scores(self, features: np.ndarray) -> np.ndarray:
        if self.ranker_ is not None:
            return np.asarray(self.ranker_.predict(features), dtype=float).reshape(-1)
        return np.asarray(self.fallback_classifier_.predict_proba(features)[:, 1], dtype=float).reshape(-1)

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series | None = None) -> "ZappaLGBMRankerClassifier":
        if LGBMRanker is None or LGBMClassifier is None:  # pragma: no cover - environment without lightgbm
            raise ImportError("LightGBM is required for zappa_lgbm_ranker.")
        self.active_task_ = str(X.attrs.get("task_name", "event_next_60m"))
        features = self._prepare_features(X, fit=True)
        target = pd.Series(y).astype(int).reset_index(drop=True)
        keys = _query_keys(X, self.query_frequency)
        informative_mask = _informative_group_mask(target, keys)
        rank_features = features.loc[informative_mask].reset_index(drop=True)
        rank_target = target.loc[informative_mask].reset_index(drop=True)
        rank_keys = keys.loc[informative_mask].reset_index(drop=True)

        self.imputer_ = SimpleImputer(strategy="median")
        design = self.imputer_.fit_transform(features)

        if informative_mask.sum() >= 24 and rank_keys.nunique() >= 2:
            rank_design = self.imputer_.transform(rank_features)
            params = dict(self.estimator_params)
            params.setdefault("objective", "lambdarank")
            params.setdefault("metric", "ndcg")
            params.setdefault("n_estimators", 140)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("num_leaves", 31)
            params.setdefault("subsample", 0.9)
            params.setdefault("colsample_bytree", 0.85)
            params.setdefault("random_state", self.random_state)
            params.setdefault("verbosity", -1)
            params.setdefault("n_jobs", 1)
            self.ranker_ = LGBMRanker(**params)
            self.ranker_.fit(rank_design, rank_target.to_numpy(), group=_group_sizes(rank_keys))
        else:
            params = dict(self.estimator_params)
            params.setdefault("objective", "binary")
            params.setdefault("n_estimators", 140)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("num_leaves", 31)
            params.setdefault("subsample", 0.9)
            params.setdefault("colsample_bytree", 0.85)
            params.setdefault("random_state", self.random_state)
            params.setdefault("verbosity", -1)
            params.setdefault("n_jobs", 1)
            self.fallback_classifier_ = LGBMClassifier(**params)
            self.fallback_classifier_.fit(design, target.to_numpy())

        raw_scores = self._raw_scores(design)
        calibration_size = max(24, int(round(len(target) * float(self.calibration_fraction))))
        calibration_size = min(calibration_size, max(len(target) - 16, 0))
        if calibration_size > 0:
            y_cal = target.iloc[-calibration_size:].reset_index(drop=True)
            score_cal = raw_scores[-calibration_size:]
        else:
            y_cal = target
            score_cal = raw_scores
        self.calibrator_ = BinaryProbabilityCalibrator(method=self.calibration_method).fit(score_cal, y_cal)
        calibrated = self.calibrator_.transform(score_cal)
        self.threshold_ = float(select_optimal_threshold(y_cal, pd.Series(calibrated), metric=self.threshold_metric)) if y_cal.nunique() >= 2 else 0.5
        raw_frame = _positive_probability_frame(_reference_probabilities(score_cal), y_cal.index)
        calibrated_frame = _positive_probability_frame(calibrated, y_cal.index)
        self.training_metadata = {
            "device_used": "cpu",
            "static_feature_count": int(len(self.feature_columns_)),
            "parameter_count": self.estimator_params.get("n_estimators", 140),
            "active_task": self.active_task_,
            "query_frequency": self.query_frequency,
            "used_ranker": self.ranker_ is not None,
            "internal_calibration": {
                "method": self.calibrator_.kind,
                "threshold": self.threshold_,
                "before": compute_classification_metrics(y_cal, (raw_frame["class_1"] >= 0.5).astype(int), raw_frame),
                "after_default": compute_classification_metrics(
                    y_cal,
                    (calibrated_frame["class_1"] >= 0.5).astype(int),
                    calibrated_frame,
                ),
                "after_optimal": {
                    **compute_classification_metrics(
                        y_cal,
                        (calibrated_frame["class_1"] >= self.threshold_).astype(int),
                        calibrated_frame,
                    ),
                    "threshold": self.threshold_,
                },
            },
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self._prepare_features(X, fit=False)
        design = self.imputer_.transform(features)
        scores = self._raw_scores(design)
        calibrated = self.calibrator_.transform(scores) if self.calibrator_ is not None else _reference_probabilities(scores)
        return _positive_probability_frame(calibrated, X.index)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(X)
        return (probabilities["class_1"] >= self.threshold_).astype(int).rename("prediction")
