"""Training and evaluation orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.datasets import PreparedDataset
from ..data.io import read_dataframe
from ..evaluation.diagnostics import save_classification_diagnostics
from ..evaluation.metrics_classification import compute_classification_metrics
from ..models.registry import create_model
from .calibration import TemperatureScaler, metrics_at_threshold, select_optimal_threshold
from ..utils.paths import ensure_dir
from ..utils.reproducibility import save_json
from ..utils.seed import set_global_seed


def _runtime_model_config(
    model_config: dict[str, Any],
    prepared: PreparedDataset,
) -> dict[str, Any]:
    runtime_config = dict(model_config)
    params = dict(runtime_config.get("params", {}))
    registry_name = str(runtime_config.get("registry_name", runtime_config.get("name", ""))).lower()
    if registry_name.startswith("tft"):
        params.setdefault("active_task", prepared.task_name)
        params.setdefault("class_names", prepared.class_names)
    runtime_config["params"] = params
    return runtime_config


def _attach_feature_attrs(
    X: pd.DataFrame,
    subset: pd.DataFrame,
    prepared: PreparedDataset,
    auxiliary_targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    attached = X.copy()
    attached.attrs["timing_frame"] = subset[["timestamp", "observation_start", "observation_end", "label_end"]].reset_index(drop=True)
    attached.attrs["prepared_metadata"] = prepared.metadata
    attached.attrs["task_name"] = prepared.task_name
    attached.attrs["class_names"] = prepared.class_names
    attached.attrs["auxiliary_targets"] = (
        auxiliary_targets.reset_index(drop=True) if auxiliary_targets is not None else pd.DataFrame(index=range(len(attached)))
    )
    return attached


def _build_auxiliary_lookup(
    prepared: PreparedDataset,
    peer_datasets: list[PreparedDataset] | None = None,
) -> pd.DataFrame:
    base = prepared.frame[["timestamp", "split"]].copy()
    if not peer_datasets:
        return base
    for peer in peer_datasets:
        if peer.task_name == prepared.task_name:
            continue
        auxiliary = peer.frame[["timestamp", "split", peer.label_column]].rename(columns={peer.label_column: peer.task_name})
        base = base.merge(auxiliary, on=["timestamp", "split"], how="left")
    return base


def _split_frame(
    prepared: PreparedDataset,
    feature_columns: list[str],
    label_column: str,
    peer_datasets: list[PreparedDataset] | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    subsets: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {}
    auxiliary_lookup = _build_auxiliary_lookup(prepared, peer_datasets)
    auxiliary_columns = [column for column in auxiliary_lookup.columns if column not in {"timestamp", "split"}]
    for split_name in ("train", "val", "test"):
        subset = prepared.frame[prepared.frame["split"] == split_name].copy()
        if auxiliary_columns:
            auxiliary_targets = subset[["timestamp", "split"]].merge(
                auxiliary_lookup,
                on=["timestamp", "split"],
                how="left",
            )[auxiliary_columns]
        else:
            auxiliary_targets = pd.DataFrame(index=range(len(subset)))
        X = _attach_feature_attrs(subset[feature_columns], subset, prepared, auxiliary_targets=auxiliary_targets)
        subsets[split_name] = (X, subset[label_column].astype(int), subset)
    return subsets


def _format_prediction_frame(
    subset: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    output = subset[["timestamp", "observation_start", "observation_end", "label_end"]].copy()
    output["split"] = split_name
    output["y_true"] = y_true.to_numpy()
    output["y_pred"] = y_pred.to_numpy()
    for column in y_prob.columns:
        output[f"prob_{column}"] = y_prob[column].to_numpy()
    return output


def _compare_predictions(
    original_pred: pd.Series,
    original_prob: pd.DataFrame,
    candidate_pred: pd.Series,
    candidate_prob: pd.DataFrame,
) -> dict[str, Any]:
    pred_match = bool(np.array_equal(original_pred.to_numpy(), candidate_pred.to_numpy()))
    prob_match = bool(np.allclose(original_prob.to_numpy(), candidate_prob.to_numpy(), atol=1e-6, rtol=1e-6))
    max_probability_diff = float(np.max(np.abs(original_prob.to_numpy() - candidate_prob.to_numpy()))) if len(original_prob) else 0.0
    return {
        "prediction_match": pred_match,
        "probability_match": prob_match,
        "max_probability_diff": max_probability_diff,
    }


def _is_binary_classification(prepared: PreparedDataset, y_prob: pd.DataFrame) -> bool:
    return len(prepared.class_names) == 2 and y_prob.shape[1] == 2


def _build_posthoc_payload(
    *,
    prepared: PreparedDataset,
    val_true: pd.Series,
    val_prob: pd.DataFrame,
    split_outputs: dict[str, dict[str, Any]],
    selection_metric: str,
) -> dict[str, Any]:
    if not _is_binary_classification(prepared, val_prob):
        return {}

    scaler = TemperatureScaler().fit(val_true, val_prob)
    calibrated_val_prob = scaler.transform(val_prob)
    raw_threshold = select_optimal_threshold(val_true, val_prob.iloc[:, 1], metric=selection_metric)
    calibrated_threshold = select_optimal_threshold(
        val_true,
        calibrated_val_prob.iloc[:, 1],
        metric=selection_metric,
    )

    posthoc: dict[str, Any] = {
        "temperature": scaler.temperature,
        "threshold_selection_metric": selection_metric,
        "thresholds": {
            "raw": raw_threshold,
            "calibrated": calibrated_threshold,
        },
        "splits": {},
    }

    for split_name, split_payload in split_outputs.items():
        y_true = split_payload["y_true"]
        y_prob = split_payload["y_prob"]
        calibrated_prob = scaler.transform(y_prob)
        raw_tuned_pred, raw_tuned_metrics = metrics_at_threshold(y_true, y_prob, raw_threshold)
        calibrated_default_pred, calibrated_default_metrics = metrics_at_threshold(y_true, calibrated_prob, 0.5)
        calibrated_tuned_pred, calibrated_tuned_metrics = metrics_at_threshold(y_true, calibrated_prob, calibrated_threshold)

        split_payload["prediction_frame"]["prob_class_1_calibrated"] = calibrated_prob.iloc[:, 1].to_numpy()
        split_payload["prediction_frame"]["y_pred_raw_threshold_tuned"] = raw_tuned_pred.to_numpy()
        split_payload["prediction_frame"]["y_pred_calibrated_default"] = calibrated_default_pred.to_numpy()
        split_payload["prediction_frame"]["y_pred_calibrated_threshold_tuned"] = calibrated_tuned_pred.to_numpy()

        posthoc["splits"][split_name] = {
            "raw_tuned": raw_tuned_metrics,
            "calibrated_default": calibrated_default_metrics,
            "calibrated_tuned": calibrated_tuned_metrics,
        }
    return posthoc


def train_and_evaluate_model(
    prepared: PreparedDataset,
    model_config: dict[str, Any],
    resolved_config: dict[str, Any],
    run_id: str,
    peer_datasets: list[PreparedDataset] | None = None,
) -> dict[str, Any]:
    model_name = model_config["name"]
    registry_name = model_config.get("registry_name", model_name)
    model_runtime_config = _runtime_model_config(model_config, prepared)
    feature_columns = prepared.feature_columns
    label_column = prepared.label_column
    splits = _split_frame(prepared, feature_columns, label_column, peer_datasets=peer_datasets)
    X_train, y_train, _ = splits["train"]
    seed = int(resolved_config.get("seed", 42))
    set_global_seed(seed)
    model = create_model(registry_name, model_runtime_config)
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    result_root = Path(resolved_config["artifacts"]["results_dir"])
    model_dir = ensure_dir(result_root / "models" / run_id / prepared.task_name)
    prediction_dir = ensure_dir(result_root / "predictions" / run_id / prepared.task_name)
    metric_dir = ensure_dir(result_root / "metrics" / run_id / prepared.task_name)
    figure_dir = ensure_dir(result_root / "figures" / run_id / prepared.task_name / model_name)
    table_dir = ensure_dir(result_root / "tables" / run_id / prepared.task_name / model_name)

    model_path = model.save(model_dir / f"{model_name}.joblib")
    model_size_bytes = int(model_path.stat().st_size)

    all_predictions: list[pd.DataFrame] = []
    metrics_by_split: dict[str, dict[str, float | None]] = {}
    split_prediction_cache: dict[str, tuple[pd.Series, pd.DataFrame]] = {}
    diagnostics_by_split: dict[str, dict[str, str]] = {}
    split_outputs: dict[str, dict[str, Any]] = {}
    canonical_frame = read_dataframe(prepared.metadata["canonical_dataset"]["path"])

    for split_name in ("val", "test"):
        X_split, y_split, subset = splits[split_name]
        inference_start = time.perf_counter()
        y_pred = model.predict(X_split)
        y_prob = model.predict_proba(X_split)
        inference_seconds = time.perf_counter() - inference_start
        split_prediction_cache[split_name] = (y_pred, y_prob)
        split_metrics = compute_classification_metrics(
            y_true=y_split,
            y_pred=y_pred,
            y_prob=y_prob,
        )
        split_metrics["inference_seconds"] = float(round(inference_seconds, 6))
        split_metrics["inference_rows_per_second"] = float(round(len(X_split) / max(inference_seconds, 1e-9), 6))
        metrics_by_split[split_name] = split_metrics
        prediction_output = _format_prediction_frame(subset, y_split, y_pred, y_prob, split_name)
        split_outputs[split_name] = {
            "y_true": y_split,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "prediction_frame": prediction_output,
        }
        diagnostics_by_split[split_name] = save_classification_diagnostics(
            y_true=y_split,
            y_pred=y_pred,
            y_prob=y_prob,
            figure_dir=figure_dir,
            table_dir=table_dir,
            prefix=f"{prepared.task_name}_{split_name}",
            class_names=prepared.class_names,
            prediction_frame=prediction_output,
            canonical_frame=canonical_frame,
        )

    posthoc_payload: dict[str, Any] = {}
    if resolved_config.get("posthoc", {}).get("temperature_scaling", False) or resolved_config.get("posthoc", {}).get("threshold_tuning", False):
        val_y_true = split_outputs["val"]["y_true"]
        val_y_prob = split_outputs["val"]["y_prob"]
        posthoc_payload = _build_posthoc_payload(
            prepared=prepared,
            val_true=val_y_true,
            val_prob=val_y_prob,
            split_outputs=split_outputs,
            selection_metric=str(resolved_config.get("posthoc", {}).get("selection_metric", "f1")),
        )

    for split_name in ("val", "test"):
        all_predictions.append(split_outputs[split_name]["prediction_frame"])

    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    prediction_path = prediction_dir / f"{model_name}.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    training_info = dict(getattr(model, "training_metadata", {}))
    training_info["training_seconds"] = float(round(fit_seconds, 6))
    training_info["artifact_size_bytes"] = model_size_bytes
    training_info["artifact_size_kb"] = float(round(model_size_bytes / 1024.0, 4))
    training_info["device_used"] = training_info.get("device_used", "cpu")
    training_info["registry_name"] = registry_name
    training_info["model_family"] = model_config.get("family", model_config.get("type", "unknown"))
    training_info["representation_tag"] = model_config.get("representation_tag", "current")
    training_info["parameter_count"] = training_info.get("parameter_count")

    reloaded_model = create_model(registry_name, model_runtime_config).load(model_path)
    reload_verification: dict[str, dict[str, Any]] = {}
    for split_name in ("val", "test"):
        X_split, _, _ = splits[split_name]
        original_pred, original_prob = split_prediction_cache[split_name]
        reloaded_pred = reloaded_model.predict(X_split)
        reloaded_prob = reloaded_model.predict_proba(X_split)
        reload_verification[split_name] = _compare_predictions(
            original_pred,
            original_prob,
            reloaded_pred,
            reloaded_prob,
        )

    reproducibility_verification: dict[str, dict[str, Any]] = {}
    if resolved_config.get("diagnostics", {}).get("reproducibility_check", False):
        set_global_seed(seed)
        repro_model = create_model(registry_name, model_runtime_config)
        repro_model.fit(X_train, y_train)
        for split_name in ("val", "test"):
            X_split, _, _ = splits[split_name]
            original_pred, original_prob = split_prediction_cache[split_name]
            repro_pred = repro_model.predict(X_split)
            repro_prob = repro_model.predict_proba(X_split)
            reproducibility_verification[split_name] = _compare_predictions(
                original_pred,
                original_prob,
                repro_pred,
                repro_prob,
            )

    metrics_payload = {
        "run_id": run_id,
        "task": prepared.task_name,
        "model": model_name,
        "registry_name": registry_name,
        "model_family": model_config.get("family", model_config.get("type", "unknown")),
        "representation_tag": model_config.get("representation_tag", "current"),
        "dataset_id": prepared.dataset_id,
        "class_names": prepared.class_names,
        "splits": metrics_by_split,
        "training": training_info,
        "verification": {
            "reload": reload_verification,
            "same_seed_refit": reproducibility_verification,
        },
        "diagnostics": diagnostics_by_split,
        "posthoc": posthoc_payload,
        "artifacts": {
            "model_path": str(model_path),
            "prediction_path": str(prediction_path),
            "table_dir": str(table_dir),
            "figure_dir": str(figure_dir),
        },
    }
    metric_path = metric_dir / f"{model_name}.json"
    save_json(metrics_payload, metric_path)
    return metrics_payload


def evaluate_saved_model(
    prepared: PreparedDataset,
    model_config: dict[str, Any],
    resolved_config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    result_root = Path(resolved_config["artifacts"]["results_dir"])
    model_path = result_root / "models" / run_id / prepared.task_name / f"{model_config['name']}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    registry_name = model_config.get("registry_name", model_config["name"])
    model_runtime_config = _runtime_model_config(model_config, prepared)
    model = create_model(registry_name, model_runtime_config).load(model_path)

    feature_columns = prepared.feature_columns
    label_column = prepared.label_column
    splits = _split_frame(prepared, feature_columns, label_column)
    metrics_by_split: dict[str, dict[str, float | None]] = {}
    for split_name in ("val", "test"):
        X_split, y_split, _ = splits[split_name]
        y_pred = model.predict(X_split)
        y_prob = model.predict_proba(X_split)
        metrics_by_split[split_name] = compute_classification_metrics(y_split, y_pred, y_prob)
    return {
        "run_id": run_id,
        "task": prepared.task_name,
        "model": model_config["name"],
        "registry_name": registry_name,
        "splits": metrics_by_split,
        "training": getattr(model, "training_metadata", {}),
    }
