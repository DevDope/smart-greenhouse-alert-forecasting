"""Detailed evaluation diagnostics and plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from ..utils.paths import ensure_dir
from .metrics_classification import compute_classification_report


def compute_threshold_sweep(
    y_true: pd.Series,
    positive_probabilities: pd.Series,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    target = pd.Series(y_true).astype(int)
    probs = pd.Series(positive_probabilities).astype(float)
    thresholds = thresholds if thresholds is not None else np.linspace(0.0, 1.0, 101)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        prediction = (probs >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(target, prediction, zero_division=0)),
                "recall": float(recall_score(target, prediction, zero_division=0)),
                "f1": float(f1_score(target, prediction, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
                "predicted_positive_rate": float(prediction.mean()),
            }
        )
    return pd.DataFrame(rows)


def _save_binary_plots(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.DataFrame,
    figure_dir: Path,
    table_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    positive_probs = y_prob.iloc[:, 1]
    artifacts: dict[str, Path] = {}

    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, positive_probs, ax=ax)
    fig.tight_layout()
    artifacts["roc_curve"] = figure_dir / f"{prefix}_roc_curve.png"
    fig.savefig(artifacts["roc_curve"])
    plt.close(fig)

    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, positive_probs, ax=ax)
    fig.tight_layout()
    artifacts["pr_curve"] = figure_dir / f"{prefix}_pr_curve.png"
    fig.savefig(artifacts["pr_curve"])
    plt.close(fig)

    fig, ax = plt.subplots()
    class_zero = positive_probs[pd.Series(y_true).astype(int) == 0]
    class_one = positive_probs[pd.Series(y_true).astype(int) == 1]
    ax.hist(class_zero, bins=20, alpha=0.6, label="true_0", density=True)
    ax.hist(class_one, bins=20, alpha=0.6, label="true_1", density=True)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    artifacts["probability_histogram"] = figure_dir / f"{prefix}_probability_histogram.png"
    fig.savefig(artifacts["probability_histogram"])
    plt.close(fig)

    fig, ax = plt.subplots()
    CalibrationDisplay.from_predictions(y_true, positive_probs, n_bins=10, ax=ax)
    fig.tight_layout()
    artifacts["calibration_plot"] = figure_dir / f"{prefix}_calibration.png"
    fig.savefig(artifacts["calibration_plot"])
    plt.close(fig)

    threshold_sweep = compute_threshold_sweep(y_true, positive_probs)
    threshold_sweep_path = table_dir / f"{prefix}_threshold_sweep.csv"
    threshold_sweep.to_csv(threshold_sweep_path, index=False)
    artifacts["threshold_sweep"] = threshold_sweep_path

    candidate_path = table_dir / f"{prefix}_threshold_candidates.csv"
    threshold_sweep.sort_values(["f1", "precision", "recall"], ascending=[False, False, False]).head(10).to_csv(
        candidate_path,
        index=False,
    )
    artifacts["threshold_candidates"] = candidate_path

    fig, ax = plt.subplots()
    ax.plot(threshold_sweep["threshold"], threshold_sweep["precision"], label="precision")
    ax.plot(threshold_sweep["threshold"], threshold_sweep["recall"], label="recall")
    ax.plot(threshold_sweep["threshold"], threshold_sweep["f1"], label="f1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    artifacts["threshold_curve"] = figure_dir / f"{prefix}_threshold_curve.png"
    fig.savefig(artifacts["threshold_curve"])
    plt.close(fig)
    return artifacts


def _save_multiclass_plots(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.DataFrame,
    class_names: list[str],
    figure_dir: Path,
    table_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    report = compute_classification_report(y_true, y_pred, class_names=class_names)
    report_frame = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    report_path = table_dir / f"{prefix}_classification_report.csv"
    report_frame.to_csv(report_path, index=False)
    artifacts["classification_report"] = report_path

    support = pd.Series(y_true).astype(int).value_counts().sort_index()
    pred_dist = pd.Series(y_pred).astype(int).value_counts().sort_index()
    support_frame = pd.DataFrame(
        {
            "class_index": np.arange(len(class_names)),
            "class_name": class_names,
            "true_support": [int(support.get(index, 0)) for index in range(len(class_names))],
            "predicted_support": [int(pred_dist.get(index, 0)) for index in range(len(class_names))],
        }
    )
    support_path = table_dir / f"{prefix}_class_support.csv"
    support_frame.to_csv(support_path, index=False)
    artifacts["class_support"] = support_path

    fig, ax = plt.subplots()
    x_axis = np.arange(len(class_names))
    ax.bar(x_axis - 0.2, support_frame["true_support"], width=0.4, label="true")
    ax.bar(x_axis + 0.2, support_frame["predicted_support"], width=0.4, label="predicted")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    artifacts["class_distribution"] = figure_dir / f"{prefix}_class_distribution.png"
    fig.savefig(artifacts["class_distribution"])
    plt.close(fig)

    if y_prob.shape[1] == len(class_names) and len(np.unique(y_true)) > 1:
        y_true_bin = label_binarize(pd.Series(y_true).astype(int), classes=np.arange(len(class_names)))
        fig, ax = plt.subplots()
        for class_index, class_name in enumerate(class_names):
            if y_true_bin[:, class_index].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], y_prob.iloc[:, class_index])
            ax.plot(fpr, tpr, label=class_name)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        fig.tight_layout()
        artifacts["multiclass_roc"] = figure_dir / f"{prefix}_multiclass_roc.png"
        fig.savefig(artifacts["multiclass_roc"])
        plt.close(fig)

    return artifacts


def _plot_temporal_window(
    frame: pd.DataFrame,
    signal_column: str,
    positive_probability_column: str,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(frame["timestamp"], frame[signal_column], label=signal_column, color="tab:blue")
    axes[0].set_ylabel(signal_column)
    axes[0].legend(loc="upper left")

    axes[1].plot(frame["timestamp"], frame[positive_probability_column], label="predicted_probability", color="tab:orange")
    axes[1].axhline(0.5, linestyle="--", color="gray", linewidth=1)
    axes[1].set_ylabel("Probability")
    axes[1].legend(loc="upper left")

    axes[2].step(frame["timestamp"], frame["y_true"], where="post", label="y_true", color="tab:green")
    axes[2].step(frame["timestamp"], frame["y_pred"], where="post", label="y_pred", color="tab:red", alpha=0.8)
    axes[2].set_ylabel("Label")
    axes[2].legend(loc="upper left")
    axes[2].set_xlabel("Timestamp")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_temporal_binary_diagnostics(
    prediction_frame: pd.DataFrame,
    canonical_frame: pd.DataFrame,
    figure_dir: Path,
    table_dir: Path,
    prefix: str,
    signal_column: str = "soil_moisture",
    context_points: int = 24,
) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    if prediction_frame.empty or signal_column not in canonical_frame.columns or "prob_class_1" not in prediction_frame.columns:
        return artifacts

    merged = prediction_frame.merge(
        canonical_frame[["timestamp", signal_column]],
        on="timestamp",
        how="left",
    ).sort_values("timestamp").reset_index(drop=True)
    selections: list[tuple[str, pd.DataFrame]] = []

    selection_rules = {
        "false_positive": merged[(merged["y_true"] == 0) & (merged["y_pred"] == 1)].sort_values("prob_class_1", ascending=False),
        "false_negative": merged[(merged["y_true"] == 1) & (merged["y_pred"] == 0)].sort_values("prob_class_1", ascending=True),
        "true_positive": merged[(merged["y_true"] == 1) & (merged["y_pred"] == 1)].sort_values("prob_class_1", ascending=False),
        "stable_negative": merged[(merged["y_true"] == 0) & (merged["y_pred"] == 0)].sort_values("prob_class_1", ascending=True),
    }
    selection_rows: list[dict[str, Any]] = []

    for label, candidates in selection_rules.items():
        if candidates.empty:
            continue
        center_index = int(candidates.index[0])
        start = max(center_index - context_points, 0)
        end = min(center_index + context_points + 1, len(merged))
        window = merged.iloc[start:end].copy()
        output_path = figure_dir / f"{prefix}_{label}_temporal.png"
        _plot_temporal_window(window, signal_column, "prob_class_1", output_path, f"{prefix} - {label}")
        artifacts[label] = output_path
        selection_rows.append(
            {
                "example_type": label,
                "center_timestamp": str(merged.iloc[center_index]["timestamp"]),
                "center_probability": float(merged.iloc[center_index]["prob_class_1"]),
                "center_y_true": int(merged.iloc[center_index]["y_true"]),
                "center_y_pred": int(merged.iloc[center_index]["y_pred"]),
            }
        )

    onset_mask = (merged["y_true"] == 1) & (merged["y_true"].shift(1, fill_value=0) == 0)
    onset_indices = merged.index[onset_mask].tolist()
    if onset_indices:
        relative = np.arange(-context_points, 1)
        trajectories: list[np.ndarray] = []
        for onset_index in onset_indices:
            values: list[float] = []
            for offset in relative:
                index = onset_index + offset
                values.append(float(merged.iloc[index]["prob_class_1"]) if 0 <= index < len(merged) else np.nan)
            trajectories.append(np.asarray(values, dtype=float))
        profile = np.nanmean(np.vstack(trajectories), axis=0)
        fig, ax = plt.subplots()
        ax.plot(relative, profile, color="tab:orange")
        ax.axvline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("Steps before positive onset")
        ax.set_ylabel("Mean predicted probability")
        fig.tight_layout()
        output_path = figure_dir / f"{prefix}_pre_event_probability_profile.png"
        fig.savefig(output_path)
        plt.close(fig)
        artifacts["pre_event_probability_profile"] = output_path

    stable_mask = (merged["y_true"] == 0).rolling(window=12, min_periods=12).sum() == 12
    stable_candidates = merged.loc[stable_mask.fillna(False), "prob_class_1"]
    if not stable_candidates.empty:
        fig, ax = plt.subplots()
        stable_candidates.plot(kind="hist", bins=20, ax=ax, color="tab:blue", alpha=0.7)
        ax.set_xlabel("Predicted probability during stable negative windows")
        fig.tight_layout()
        output_path = figure_dir / f"{prefix}_negative_stability_histogram.png"
        fig.savefig(output_path)
        plt.close(fig)
        artifacts["negative_stability"] = output_path

    if selection_rows:
        selection_path = table_dir / f"{prefix}_temporal_examples.csv"
        pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
        artifacts["temporal_examples"] = selection_path
    return artifacts


def save_classification_diagnostics(
    *,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.DataFrame,
    figure_dir: str | Path,
    table_dir: str | Path,
    prefix: str,
    class_names: list[str],
    prediction_frame: pd.DataFrame | None = None,
    canonical_frame: pd.DataFrame | None = None,
    signal_column: str = "soil_moisture",
) -> dict[str, str]:
    fig_dir = ensure_dir(figure_dir)
    tbl_dir = ensure_dir(table_dir)
    artifacts: dict[str, Path] = {}

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    fig.tight_layout()
    artifacts["confusion_matrix"] = fig_dir / f"{prefix}_confusion_matrix.png"
    fig.savefig(artifacts["confusion_matrix"])
    plt.close(fig)

    is_binary = y_prob.shape[1] == 2
    if is_binary:
        artifacts.update(_save_binary_plots(y_true, y_pred, y_prob, fig_dir, tbl_dir, prefix))
        if prediction_frame is not None and canonical_frame is not None:
            artifacts.update(
                _save_temporal_binary_diagnostics(
                    prediction_frame=prediction_frame,
                    canonical_frame=canonical_frame,
                    figure_dir=fig_dir,
                    table_dir=tbl_dir,
                    prefix=prefix,
                    signal_column=signal_column,
                )
            )
    else:
        artifacts.update(_save_multiclass_plots(y_true, y_pred, y_prob, class_names, fig_dir, tbl_dir, prefix))

    return {name: str(path) for name, path in artifacts.items()}

