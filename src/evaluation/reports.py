"""Results aggregation, diagnostics, and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..evaluation.metrics_classification import compute_classification_metrics
from ..utils.paths import ensure_dir
from .plots import export_paper_figures

if TYPE_CHECKING:
    from ..data.datasets import CanonicalDataset, PreparedDataset


def collect_metric_payloads(metrics_root: str | Path, run_id: str | None = None) -> list[dict[str, Any]]:
    root = Path(metrics_root)
    candidates = list(root.glob("*/*/*.json")) if run_id is None else list((root / run_id).glob("*/*.json"))
    payloads: list[dict[str, Any]] = []
    for candidate in sorted(candidates):
        payloads.append(json.loads(candidate.read_text(encoding="utf-8")))
    return payloads


def collect_metric_rows(metrics_root: str | Path, run_id: str | None = None) -> pd.DataFrame:
    payloads = collect_metric_payloads(metrics_root, run_id=run_id)
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        training = payload.get("training", {})
        posthoc = payload.get("posthoc", {})
        thresholds = posthoc.get("thresholds", {})
        for split_name, split_metrics in payload["splits"].items():
            split_posthoc = posthoc.get("splits", {}).get(split_name, {})
            row = {
                "run_id": payload["run_id"],
                "task": payload["task"],
                "model": payload["model"],
                "registry_name": payload.get("registry_name", payload["model"]),
                "model_family": payload.get("model_family", training.get("model_family")),
                "representation_tag": payload.get("representation_tag", training.get("representation_tag", "current")),
                "split": split_name,
            }
            row.update(split_metrics)
            row["training_seconds"] = training.get("training_seconds")
            row["epochs_trained"] = training.get("epochs_trained")
            row["sequence_length"] = training.get("sequence_length")
            row["sequence_feature_count"] = training.get("sequence_feature_count")
            row["static_feature_count"] = training.get("static_feature_count")
            row["parameter_count"] = training.get("parameter_count")
            row["artifact_size_kb"] = training.get("artifact_size_kb")
            row["device_used"] = training.get("device_used")
            row["sequence_input_shape"] = json.dumps(training.get("sequence_input_shape"))
            row["static_input_shape"] = json.dumps(training.get("static_input_shape"))
            row["temperature"] = posthoc.get("temperature")
            row["threshold_raw_selected"] = thresholds.get("raw")
            row["threshold_calibrated_selected"] = thresholds.get("calibrated")
            for prefix, block in split_posthoc.items():
                for metric_name, metric_value in block.items():
                    row[f"{metric_name}_{prefix}"] = metric_value
            rows.append(row)
    return pd.DataFrame(rows)


def _write_frame(frame: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    csv_path = output_dir / f"{stem}.csv"
    frame.to_csv(csv_path, index=False)
    md_path = output_dir / f"{stem}.md"
    if frame.empty:
        md_path.write_text("_empty_", encoding="utf-8")
    else:
        try:
            markdown = frame.to_markdown(index=False)
        except ImportError:
            markdown = frame.to_csv(index=False)
        md_path.write_text(markdown, encoding="utf-8")
    return csv_path


def _round_columns(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> pd.DataFrame:
    rounded = frame.copy()
    for column in columns:
        if column in rounded.columns:
            rounded[column] = rounded[column].apply(
                lambda value: round(float(value), digits) if pd.notna(value) else value
            )
    return rounded


def _best_rows(summary: pd.DataFrame, metric: str, *, ascending: bool) -> pd.DataFrame:
    if summary.empty or metric not in summary.columns:
        return pd.DataFrame()
    subset = summary.dropna(subset=[metric]).copy()
    if subset.empty:
        return subset
    return (
        subset.sort_values(["task", metric, "pr_auc", "f1"], ascending=[True, ascending, False, False])
        .groupby("task", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _best_calibrated(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    subset = summary.dropna(subset=["ece"]).copy()
    if subset.empty:
        return subset
    if "brier_score" not in subset.columns:
        subset["brier_score"] = np.nan
    return (
        subset.sort_values(["task", "ece", "brier_score", "f1"], ascending=[True, True, True, False])
        .groupby("task", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _generalization_gap(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    keys = ["run_id", "task", "model", "registry_name", "model_family", "representation_tag"]
    val_frame = summary[summary["split"] == "val"].copy()
    test_frame = summary[summary["split"] == "test"].copy()
    if val_frame.empty or test_frame.empty:
        return pd.DataFrame()
    merged = val_frame.merge(test_frame, on=keys, suffixes=("_val", "_test"))
    for metric in ("f1", "pr_auc", "roc_auc", "ece"):
        if f"{metric}_val" in merged.columns and f"{metric}_test" in merged.columns:
            merged[f"{metric}_gap"] = merged[f"{metric}_val"] - merged[f"{metric}_test"]
            merged[f"{metric}_abs_gap"] = (merged[f"{metric}_gap"]).abs()
    keep_columns = keys + [
        column
        for column in merged.columns
        if column.endswith("_gap") or column.endswith("_abs_gap")
    ]
    return merged[keep_columns].sort_values(["task", "f1_abs_gap", "pr_auc_abs_gap"], ascending=[True, True, True])


def _flatten_verification(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        verification = payload.get("verification", {})
        for check_name, split_mapping in verification.items():
            for split_name, values in split_mapping.items():
                row = {
                    "run_id": payload["run_id"],
                    "task": payload["task"],
                    "model": payload["model"],
                    "registry_name": payload.get("registry_name", payload["model"]),
                    "representation_tag": payload.get("representation_tag", "current"),
                    "check": check_name,
                    "split": split_name,
                }
                row.update(values)
                rows.append(row)
    return pd.DataFrame(rows)


def _prediction_metric_consistency(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        prediction_path = payload.get("artifacts", {}).get("prediction_path")
        if prediction_path is None or not Path(prediction_path).exists():
            continue
        frame = pd.read_csv(prediction_path)
        probability_columns = [column for column in frame.columns if column.startswith("prob_class_")]
        if not probability_columns:
            continue
        for split_name, stored_metrics in payload["splits"].items():
            split_frame = frame[frame["split"] == split_name].copy()
            if split_frame.empty:
                continue
            y_true = split_frame["y_true"].astype(int)
            y_pred = split_frame["y_pred"].astype(int)
            y_prob = split_frame[probability_columns].rename(columns=lambda value: value.replace("prob_", ""))
            recomputed = compute_classification_metrics(y_true, y_pred, y_prob)
            row = {
                "run_id": payload["run_id"],
                "task": payload["task"],
                "model": payload["model"],
                "split": split_name,
            }
            for metric_name, stored_value in stored_metrics.items():
                recomputed_value = recomputed.get(metric_name)
                row[f"{metric_name}_stored"] = stored_value
                row[f"{metric_name}_recomputed"] = recomputed_value
                if stored_value is None or recomputed_value is None:
                    row[f"{metric_name}_abs_diff"] = None
                else:
                    row[f"{metric_name}_abs_diff"] = float(abs(stored_value - recomputed_value))
            rows.append(row)
    return pd.DataFrame(rows)


def _family_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    families = ["baseline_tabular", "deep_classic", "transformer"]
    rows: list[dict[str, Any]] = []
    for task_name, task_frame in summary.groupby("task"):
        for family in families:
            family_frame = task_frame[task_frame["model_family"] == family].dropna(subset=["pr_auc", "f1"])
            if family_frame.empty:
                continue
            best = family_frame.sort_values(["pr_auc", "f1", "roc_auc"], ascending=[False, False, False]).iloc[0]
            rows.append(
                {
                    "task": task_name,
                    "family": family,
                    "model": best["model"],
                    "representation_tag": best["representation_tag"],
                    "pr_auc": best.get("pr_auc"),
                    "roc_auc": best.get("roc_auc"),
                    "f1": best.get("f1"),
                    "precision": best.get("precision"),
                    "recall": best.get("recall"),
                    "balanced_accuracy": best.get("balanced_accuracy"),
                    "brier_score": best.get("brier_score"),
                    "ece": best.get("ece"),
                    "training_seconds": best.get("training_seconds"),
                    "inference_seconds": best.get("inference_seconds"),
                }
            )
    return pd.DataFrame(rows).sort_values(["task", "family"]).reset_index(drop=True)


def _representation_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    deep_frame = summary[summary["model_family"].isin(["deep_classic", "transformer"])].copy()
    if deep_frame.empty:
        return deep_frame
    rows: list[dict[str, Any]] = []
    for (task_name, registry_name, representation_tag), group in deep_frame.groupby(
        ["task", "registry_name", "representation_tag"]
    ):
        best = group.sort_values(["pr_auc", "f1", "roc_auc"], ascending=[False, False, False]).iloc[0]
        rows.append(
            {
                "task": task_name,
                "registry_name": registry_name,
                "representation_tag": representation_tag,
                "model": best["model"],
                "model_family": best["model_family"],
                "pr_auc": best.get("pr_auc"),
                "roc_auc": best.get("roc_auc"),
                "f1": best.get("f1"),
                "ece": best.get("ece"),
                "training_seconds": best.get("training_seconds"),
                "inference_seconds": best.get("inference_seconds"),
                "sequence_length": best.get("sequence_length"),
                "sequence_feature_count": best.get("sequence_feature_count"),
                "static_feature_count": best.get("static_feature_count"),
                "sequence_input_shape": best.get("sequence_input_shape"),
                "static_input_shape": best.get("static_input_shape"),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "registry_name", "representation_tag"]).reset_index(drop=True)


def _future_matrix(series: pd.Series, horizon_steps: int) -> pd.DataFrame:
    return pd.concat([series.shift(-step) for step in range(1, horizon_steps + 1)], axis=1)


def _trigger_steps(source: pd.Series, task_config: dict[str, Any], horizon_steps: int) -> pd.Series:
    rule = task_config["rule"]
    future_values = _future_matrix(source, horizon_steps)
    if rule["type"] == "future_delta_above_threshold":
        trigger = future_values.subtract(source, axis=0) >= float(rule["threshold"])
    elif rule["type"] == "any_above_threshold":
        trigger = future_values > float(rule["threshold"])
    elif rule["type"] == "any_below_threshold":
        trigger = future_values < float(rule["threshold"])
    else:
        return pd.Series(np.nan, index=source.index, dtype=float)
    return trigger.apply(lambda row: float(np.argmax(row.to_numpy()) + 1) if row.any() else np.nan, axis=1)


def _build_label_diagnostics(
    prepared_datasets: list["PreparedDataset"],
    canonical_dataset: "CanonicalDataset",
    figure_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical_frame = canonical_dataset.frame.copy()
    timestamp_column = canonical_dataset.metadata["timestamp_column"]
    canonical_frame[timestamp_column] = pd.to_datetime(canonical_frame[timestamp_column])
    support_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for prepared in prepared_datasets:
        frame = prepared.frame.sort_values("timestamp").reset_index(drop=True)
        label_column = prepared.label_column
        task_config = prepared.metadata["task"]
        label_metadata = prepared.metadata["label_metadata"]
        rule = task_config["rule"]
        rule_type = rule["type"]
        source_column = rule.get("source_column")
        horizon_steps = int(label_metadata["horizon_steps"])
        label_counts = frame[label_column].astype(int).value_counts().sort_index()
        class_names = prepared.class_names
        class_support = {
            class_name: int(label_counts.get(index, 0))
            for index, class_name in enumerate(class_names)
        }
        for split_name, split_frame in frame.groupby("split"):
            split_counts = split_frame[label_column].astype(int).value_counts().sort_index()
            for class_index, class_name in enumerate(class_names):
                support_rows.append(
                    {
                        "task": prepared.task_name,
                        "split": split_name,
                        "class_index": class_index,
                        "class_name": class_name,
                        "support": int(split_counts.get(class_index, 0)),
                    }
                )

        row: dict[str, Any] = {
            "task": prepared.task_name,
            "rule_type": rule_type,
            "alarm_name": label_metadata.get("alarm_name"),
            "adaptive_mode": label_metadata.get("adaptive_mode"),
            "adaptive_adjusted": label_metadata.get("adaptive_adjusted"),
            "source_column": source_column,
            "horizon_steps": horizon_steps,
            "row_count": int(len(frame)),
            "train_rows": int((frame["split"] == "train").sum()),
            "val_rows": int((frame["split"] == "val").sum()),
            "test_rows": int((frame["split"] == "test").sum()),
            "class_names": json.dumps(class_names),
            "class_support": json.dumps(class_support),
            "required_columns": json.dumps(label_metadata.get("required_columns", [])),
            "resolved_columns": json.dumps(label_metadata.get("resolved_columns", {})),
            "resolved_thresholds": json.dumps(label_metadata.get("resolved_thresholds", {})),
            "adaptive_requested_percentiles": json.dumps(label_metadata.get("adaptive_requested_percentiles", {})),
            "adaptive_selected_percentiles": json.dumps(label_metadata.get("adaptive_selected_percentiles", {})),
            "adaptive_validation": json.dumps(label_metadata.get("adaptive_validation", {})),
            "adaptive_threshold_sources": json.dumps(label_metadata.get("adaptive_threshold_sources", {})),
            "dataset_statistics": json.dumps(label_metadata.get("dataset_statistics", {})),
            "radiation_fallback_used": label_metadata.get("radiation_fallback_used"),
            "radiation_column": label_metadata.get("radiation_column"),
        }

        if len(class_names) == 2:
            positive_rate = float(frame[label_column].mean())
            row["positive_rate"] = positive_rate
            adaptive_validation = label_metadata.get("adaptive_validation", {})
            if adaptive_validation:
                row["adaptive_validation_overall_positive_rate"] = adaptive_validation.get("overall_positive_rate")
                row["adaptive_validation_stability_range"] = adaptive_validation.get("stability_range")
                row["adaptive_validation_passes"] = adaptive_validation.get("passes")
            for split_name in ("train", "val", "test"):
                split_frame = frame[frame["split"] == split_name]
                if split_frame.empty:
                    row[f"positive_rate_{split_name}"] = None
                    row[f"degenerate_{split_name}"] = True
                    continue
                row[f"positive_rate_{split_name}"] = float(split_frame[label_column].mean())
                row[f"degenerate_{split_name}"] = bool(split_frame[label_column].nunique(dropna=True) < 2)
            row["degenerate_any_split"] = bool(
                row.get("degenerate_train", False)
                or row.get("degenerate_val", False)
                or row.get("degenerate_test", False)
            )

            if source_column and source_column in canonical_frame.columns:
                source = pd.to_numeric(canonical_frame[source_column], errors="coerce")
                trigger_steps = _trigger_steps(source, task_config, horizon_steps)
                trigger_frame = canonical_frame[[timestamp_column]].copy()
                trigger_frame["trigger_step"] = trigger_steps.to_numpy()
                merged = frame[[timestamp_column, label_column]].merge(trigger_frame, on=timestamp_column, how="left")
                positive_steps = merged.loc[merged[label_column] == 1, "trigger_step"].dropna()
                if not positive_steps.empty:
                    row["lead_step_mean"] = float(positive_steps.mean())
                    row["lead_step_median"] = float(positive_steps.median())
                    row["lead_step_min"] = float(positive_steps.min())
                    row["lead_step_max"] = float(positive_steps.max())
                    row["fraction_step_1"] = float((positive_steps == 1).mean())
                    row["fraction_leq_2"] = float((positive_steps <= 2).mean())
                    row["fraction_leq_3"] = float((positive_steps <= 3).mean())
                    row["label_proxy_locality_flag"] = bool(
                        row["fraction_step_1"] >= 0.5 or row["fraction_leq_2"] >= 0.75
                    )

                    fig, ax = plt.subplots()
                    positive_steps.plot(kind="hist", bins=min(horizon_steps, 20), ax=ax, color="tab:orange", alpha=0.8)
                    ax.set_xlabel("First trigger step within horizon")
                    ax.set_ylabel("Count")
                    fig.tight_layout()
                    fig.savefig(figure_dir / f"{prepared.task_name}_lead_step_distribution.png")
                    plt.close(fig)

                onset_mask = (frame[label_column] == 1) & (frame[label_column].shift(1, fill_value=0) == 0)
                onset_timestamps = pd.to_datetime(frame.loc[onset_mask, timestamp_column])
                if not onset_timestamps.empty:
                    index_lookup = {
                        pd.Timestamp(ts): idx
                        for idx, ts in enumerate(canonical_frame[timestamp_column])
                    }
                    relative = np.arange(-24, 1)
                    traces: list[np.ndarray] = []
                    for timestamp in onset_timestamps:
                        center = index_lookup.get(pd.Timestamp(timestamp))
                        if center is None:
                            continue
                        values: list[float] = []
                        for offset in relative:
                            idx = center + offset
                            if 0 <= idx < len(canonical_frame):
                                values.append(float(canonical_frame.iloc[idx][source_column]))
                            else:
                                values.append(np.nan)
                        traces.append(np.asarray(values, dtype=float))
                    if traces:
                        profile = np.nanmean(np.vstack(traces), axis=0)
                        fig, ax = plt.subplots()
                        ax.plot(relative, profile, color="tab:blue")
                        ax.axvline(0, linestyle="--", color="gray", linewidth=1)
                        ax.set_xlabel("Steps before positive onset")
                        ax.set_ylabel(source_column)
                        fig.tight_layout()
                        fig.savefig(figure_dir / f"{prepared.task_name}_pre_onset_profile.png")
                        plt.close(fig)
        else:
            counts = frame[label_column].astype(int).value_counts(normalize=True).sort_index()
            row["majority_class_rate"] = float(counts.max()) if not counts.empty else None
            row["class_balance_entropy"] = float(
                -np.sum([value * np.log2(value) for value in counts if value > 0])
            ) if not counts.empty else None

        summary_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(support_rows)


def _task_difficulty(summary_test: pd.DataFrame, label_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if summary_test.empty or label_diagnostics.empty:
        return pd.DataFrame()
    best = _best_rows(summary_test, "pr_auc", ascending=False)[["task", "model", "pr_auc", "f1", "ece"]]
    difficulty = label_diagnostics.merge(best, on="task", how="left", suffixes=("", "_best"))
    difficulty["difficulty_rank_hint"] = difficulty["pr_auc"].rank(ascending=True, method="dense")
    return difficulty.sort_values(["difficulty_rank_hint", "task"]).reset_index(drop=True)


def _paper_summary_main(summary_test: pd.DataFrame) -> pd.DataFrame:
    if summary_test.empty:
        return pd.DataFrame(
            columns=["task", "model", "pr_auc", "f1", "roc_auc", "brier", "precision", "recall"]
        )
    frame = summary_test[
        ["task", "model", "pr_auc", "f1", "roc_auc", "brier_score", "precision", "recall"]
    ].rename(columns={"brier_score": "brier"})
    frame = frame.sort_values(["task", "pr_auc", "f1"], ascending=[True, False, False]).reset_index(drop=True)
    return _round_columns(frame, ["pr_auc", "f1", "roc_auc", "brier", "precision", "recall"])


def _paper_task_tables(summary_test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    if summary_test.empty:
        return tables
    for task_name, task_frame in summary_test.groupby("task"):
        frame = task_frame[
            [
                "model",
                "pr_auc",
                "f1",
                "roc_auc",
                "brier_score",
                "precision",
                "recall",
                "training_seconds",
                "inference_seconds",
            ]
        ].rename(
            columns={
                "brier_score": "brier",
                "training_seconds": "train_time",
                "inference_seconds": "inference_time",
            }
        )
        frame = frame.sort_values(["pr_auc", "f1"], ascending=[False, False]).reset_index(drop=True)
        tables[task_name] = _round_columns(
            frame,
            ["pr_auc", "f1", "roc_auc", "brier", "precision", "recall", "train_time", "inference_time"],
        )
    return tables


def _pick_best_model(
    task_frame: pd.DataFrame,
    metric_column: str,
    *,
    ascending: bool,
) -> tuple[str | None, float | None]:
    if metric_column not in task_frame.columns:
        return None, None
    subset = task_frame.dropna(subset=[metric_column]).copy()
    if subset.empty:
        return None, None
    sort_columns = [metric_column]
    ascending_list = [ascending]
    if metric_column != "pr_auc" and "pr_auc" in subset.columns:
        sort_columns.append("pr_auc")
        ascending_list.append(False)
    if metric_column != "f1" and "f1" in subset.columns:
        sort_columns.append("f1")
        ascending_list.append(False)
    best = subset.sort_values(sort_columns, ascending=ascending_list).iloc[0]
    return str(best["model"]), float(best[metric_column])


def _paper_best_model_by_task(summary_test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_name, task_frame in summary_test.groupby("task"):
        best_model_pr_auc, best_pr_auc = _pick_best_model(task_frame, "pr_auc", ascending=False)
        best_model_f1, best_f1 = _pick_best_model(task_frame, "f1", ascending=False)
        best_model_brier, best_brier = _pick_best_model(task_frame, "brier_score", ascending=True)
        rows.append(
            {
                "task": task_name,
                "best_model_pr_auc": best_model_pr_auc,
                "best_pr_auc": best_pr_auc,
                "best_model_f1": best_model_f1,
                "best_f1": best_f1,
                "best_model_brier": best_model_brier,
                "best_brier": best_brier,
            }
        )
    frame = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)
    return _round_columns(frame, ["best_pr_auc", "best_f1", "best_brier"])


def _task_validity_status(row: pd.Series) -> str:
    degenerate_flags = [
        bool(value) if pd.notna(value) else False
        for value in [
            row.get("is_degenerate_train"),
            row.get("is_degenerate_val"),
            row.get("is_degenerate_test"),
        ]
    ]
    if any(degenerate_flags):
        return "degenerate"
    prevalence_values = [
        row.get("prevalence_global"),
        row.get("prevalence_train"),
        row.get("prevalence_val"),
        row.get("prevalence_test"),
    ]
    if any(pd.isna(value) for value in prevalence_values):
        return "warning"
    if any(float(value) < 0.03 or float(value) > 0.25 for value in prevalence_values):
        return "warning"
    adaptive_passes = row.get("adaptive_validation_passes")
    if pd.notna(adaptive_passes) and not bool(adaptive_passes):
        return "warning"
    return "valid"


def _paper_task_validity(label_diagnostics: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    if label_diagnostics.empty:
        return pd.DataFrame(
            {
                "task": tasks,
                "prevalence_global": [np.nan] * len(tasks),
                "prevalence_train": [np.nan] * len(tasks),
                "prevalence_val": [np.nan] * len(tasks),
                "prevalence_test": [np.nan] * len(tasks),
                "is_degenerate_train": [True] * len(tasks),
                "is_degenerate_val": [True] * len(tasks),
                "is_degenerate_test": [True] * len(tasks),
                "status": ["degenerate"] * len(tasks),
            }
        )

    columns = {
        "task": "task",
        "positive_rate": "prevalence_global",
        "positive_rate_train": "prevalence_train",
        "positive_rate_val": "prevalence_val",
        "positive_rate_test": "prevalence_test",
        "degenerate_train": "is_degenerate_train",
        "degenerate_val": "is_degenerate_val",
        "degenerate_test": "is_degenerate_test",
        "adaptive_validation_passes": "adaptive_validation_passes",
    }
    available = [column for column in columns if column in label_diagnostics.columns]
    frame = label_diagnostics[available].rename(columns=columns).copy()
    if "task" in frame.columns:
        frame = frame.drop_duplicates(subset=["task"]).set_index("task").reindex(tasks).reset_index()
    frame["status"] = frame.apply(_task_validity_status, axis=1)
    rounded = _round_columns(
        frame,
        ["prevalence_global", "prevalence_train", "prevalence_val", "prevalence_test"],
    )
    keep_columns = [
        "task",
        "prevalence_global",
        "prevalence_train",
        "prevalence_val",
        "prevalence_test",
        "is_degenerate_train",
        "is_degenerate_val",
        "is_degenerate_test",
        "status",
    ]
    return rounded[keep_columns]


def _paper_top_models(summary_test: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if summary_test.empty:
        return pd.DataFrame(columns=["task", "rank", "model", "pr_auc", "f1", "brier"])
    frame = (
        summary_test.sort_values(["task", "pr_auc", "f1"], ascending=[True, False, False])
        .groupby("task", as_index=False)
        .head(top_n)
        .copy()
    )
    frame["rank"] = frame.groupby("task").cumcount() + 1
    frame = frame[["task", "rank", "model", "pr_auc", "f1", "brier_score"]].rename(columns={"brier_score": "brier"})
    return _round_columns(frame.reset_index(drop=True), ["pr_auc", "f1", "brier"])


def export_paper_tables(
    summary_test: pd.DataFrame,
    label_diagnostics: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    output_path = ensure_dir(output_dir)
    task_order = list(dict.fromkeys(summary_test["task"].tolist()))
    paper_summary = _paper_summary_main(summary_test)
    paper_best = _paper_best_model_by_task(summary_test)
    paper_validity = _paper_task_validity(label_diagnostics, task_order)
    paper_top = _paper_top_models(summary_test, top_n=3)

    _write_frame(paper_summary, output_path, "paper_summary_main")
    _write_frame(paper_best, output_path, "paper_best_model_by_task")
    _write_frame(paper_validity, output_path, "paper_task_validity")
    _write_frame(paper_top, output_path, "paper_top_models")

    task_tables = _paper_task_tables(summary_test)
    for task_name, task_frame in task_tables.items():
        _write_frame(task_frame, output_path, f"paper_table_{task_name}")

    return {
        "paper_summary_main": paper_summary,
        "paper_best_model_by_task": paper_best,
        "paper_task_validity": paper_validity,
        "paper_top_models": paper_top,
        **{f"paper_table_{task_name}": task_frame for task_name, task_frame in task_tables.items()},
    }


def _technical_note(
    summary_current_test: pd.DataFrame,
    summary_all_test: pd.DataFrame,
    representation_comparison: pd.DataFrame,
    label_diagnostics: pd.DataFrame,
    gaps: pd.DataFrame,
) -> str:
    lines = ["# Etapa 2.5 - Hallazgos", ""]

    winners = _best_rows(summary_current_test, "pr_auc", ascending=False)
    if not winners.empty:
        lines.append("## Ganadores por tarea")
        for _, row in winners.iterrows():
            lines.append(
                f"- `{row['task']}`: gana `{row['model']}` con PR-AUC `{row.get('pr_auc', float('nan')):.4f}` "
                f"y F1 `{row.get('f1', float('nan')):.4f}`."
            )
        lines.append("")

    calibrated = _best_calibrated(summary_current_test)
    if not calibrated.empty:
        lines.append("## Mejor calibrados")
        for _, row in calibrated.iterrows():
            lines.append(
                f"- `{row['task']}`: `{row['model']}` con ECE `{row.get('ece', float('nan')):.4f}` "
                f"y Brier `{row.get('brier_score', float('nan')):.4f}`."
            )
        lines.append("")

    if not gaps.empty:
        model_gap = (
            gaps.groupby("model")[["f1_abs_gap", "pr_auc_abs_gap"]]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["f1_abs_gap", "pr_auc_abs_gap"])
        )
        if not model_gap.empty:
            most_stable = model_gap.iloc[0]
            most_overfit = model_gap.sort_values(["f1_abs_gap", "pr_auc_abs_gap"], ascending=[False, False]).iloc[0]
            lines.append("## Estabilidad y sobreajuste")
            lines.append(
                f"- Más estable en promedio: `{most_stable['model']}` "
                f"(gap abs F1 `{most_stable.get('f1_abs_gap', float('nan')):.4f}`)."
            )
            lines.append(
                f"- Mayor indicio de sobreajuste: `{most_overfit['model']}` "
                f"(gap abs F1 `{most_overfit.get('f1_abs_gap', float('nan')):.4f}`)."
            )
            lines.append("")

    representation_bias = False
    tft_note = "No hubo evidencia suficiente para abrir Etapa 3."
    if not representation_comparison.empty:
        tft_rows = representation_comparison[representation_comparison["registry_name"] == "tft_base"].copy()
        if not tft_rows.empty:
            current = (
                tft_rows[tft_rows["representation_tag"] == "current"][["task", "f1", "pr_auc"]]
                .rename(columns={"f1": "f1_current", "pr_auc": "pr_auc_current"})
            )
            best = (
                tft_rows.sort_values(["task", "f1", "pr_auc"], ascending=[True, False, False])
                .groupby("task", as_index=False)
                .head(1)[["task", "representation_tag", "f1", "pr_auc"]]
                .rename(columns={"representation_tag": "best_representation", "f1": "f1_best", "pr_auc": "pr_auc_best"})
            )
            merged = current.merge(best, on="task", how="inner")
            if not merged.empty:
                merged["delta_f1"] = merged["f1_best"] - merged["f1_current"]
                merged["delta_pr_auc"] = merged["pr_auc_best"] - merged["pr_auc_current"]
                representation_bias = bool((merged["delta_f1"] >= 0.03).any() or (merged["delta_pr_auc"] >= 0.03).any())
                if representation_bias:
                    tft_note = (
                        "La representación actual sí estaba castigando a `tft_base`: "
                        "aparecen mejoras claras al pasar a `sequence_first`."
                    )
                else:
                    tft_note = (
                        "`tft_base` no mostró una mejora clara frente a su setting actual; "
                        "conviene seguir ajustando representación/labels antes de abrir variantes."
                    )

    lines.append("## Representación y decisión sobre Etapa 3")
    if not label_diagnostics.empty and "label_proxy_locality_flag" in label_diagnostics.columns:
        local_rows = label_diagnostics[label_diagnostics["label_proxy_locality_flag"] == True]  # noqa: E712
        if not local_rows.empty:
            local_tasks = ", ".join(f"`{task}`" for task in local_rows["task"].tolist())
            lines.append(f"- Hay evidencia de labels proxy muy locales en: {local_tasks}.")
    lines.append(f"- {tft_note}")
    if representation_bias:
        lines.append("- Recomendación: sí vale la pena abrir Etapa 3 con control estricto de representación.")
    else:
        lines.append("- Recomendación: no abrir Etapa 3 todavía; primero conviene ajustar representación o labels proxy.")
    lines.append("")
    return "\n".join(lines)


def write_summary_report(
    metrics_root: str | Path,
    tables_root: str | Path,
    run_id: str | None = None,
    *,
    prepared_datasets: list["PreparedDataset"] | None = None,
    canonical_dataset: "CanonicalDataset" | None = None,
    figures_root: str | Path | None = None,
) -> Path:
    payloads = collect_metric_payloads(metrics_root, run_id=run_id)
    summary = collect_metric_rows(metrics_root, run_id=run_id)
    output_dir = ensure_dir(tables_root if run_id is None else Path(tables_root) / run_id)
    run_figure_dir = ensure_dir(
        Path(figures_root) / run_id if figures_root is not None and run_id is not None else output_dir
    )
    analysis_figure_dir = ensure_dir(run_figure_dir / "analysis")
    if summary.empty:
        output_path = output_dir / "summary.csv"
        summary.to_csv(output_path, index=False)
        return output_path

    summary = summary.sort_values(["task", "split", "model"]).reset_index(drop=True)
    _write_frame(summary, output_dir, "summary")

    test_summary = summary[summary["split"] == "test"].copy()
    current_test = test_summary[test_summary["representation_tag"].fillna("current") == "current"].copy()

    _write_frame(_best_rows(test_summary, "pr_auc", ascending=False), output_dir, "best_by_pr_auc")
    _write_frame(_best_rows(test_summary, "f1", ascending=False), output_dir, "best_by_f1")
    _write_frame(_best_calibrated(test_summary), output_dir, "best_by_calibration")
    _write_frame(_best_rows(test_summary, "training_seconds", ascending=True), output_dir, "best_by_training_time")
    _write_frame(_best_rows(test_summary, "inference_seconds", ascending=True), output_dir, "best_by_inference_time")
    _write_frame(_family_comparison(test_summary), output_dir, "family_comparison")
    _write_frame(_representation_comparison(test_summary), output_dir, "representation_comparison")
    _write_frame(_generalization_gap(summary), output_dir, "generalization_gap")
    _write_frame(_flatten_verification(payloads), output_dir, "verification_summary")
    _write_frame(_prediction_metric_consistency(payloads), output_dir, "prediction_metric_consistency")

    model_costs = test_summary[
        [
            "task",
            "model",
            "registry_name",
            "model_family",
            "representation_tag",
            "training_seconds",
            "inference_seconds",
            "inference_rows_per_second",
            "parameter_count",
            "artifact_size_kb",
            "device_used",
            "sequence_length",
            "sequence_feature_count",
            "static_feature_count",
            "sequence_input_shape",
            "static_input_shape",
        ]
    ].sort_values(["task", "model_family", "model"])
    _write_frame(model_costs, output_dir, "model_costs")

    label_diagnostics = pd.DataFrame()
    if prepared_datasets is not None and canonical_dataset is not None:
        label_diagnostics, label_support = _build_label_diagnostics(
            prepared_datasets,
            canonical_dataset,
            analysis_figure_dir,
        )
        _write_frame(label_diagnostics, output_dir, "label_diagnostics")
        _write_frame(label_support, output_dir, "label_support_by_split")
        _write_frame(_task_difficulty(current_test, label_diagnostics), output_dir, "task_difficulty")

    paper_tables = export_paper_tables(test_summary, label_diagnostics, output_dir)
    export_paper_figures(
        summary_test=test_summary,
        task_validity=paper_tables["paper_task_validity"],
        best_model_frame=paper_tables["paper_best_model_by_task"],
        payloads=payloads,
        output_dir=run_figure_dir,
    )

    note = _technical_note(
        summary_current_test=current_test,
        summary_all_test=test_summary,
        representation_comparison=_representation_comparison(test_summary),
        label_diagnostics=label_diagnostics,
        gaps=_generalization_gap(summary),
    )
    (output_dir / "technical_note.md").write_text(note, encoding="utf-8")

    return output_dir / "summary.csv"
