"""Evaluation plots and paper-ready benchmark figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from ..utils.paths import ensure_dir
from .diagnostics import compute_threshold_sweep


PAPER_MODEL_LABELS = {
    "fripp_lgbm_sparse": "discipline_wrapper",
    "discipline_lgbm_sparse": "discipline_native",
    "zappa_lgbm_chain": "juxtapose_wrapper",
    "juxtapose_lgbm_chain": "juxtapose_native_v1",
    "juxtapose_lgbm_relation_v2": "juxtapose_native_v2",
}


def _model_label(model_name: object) -> str:
    text = str(model_name)
    return PAPER_MODEL_LABELS.get(text, text)


def _model_slug(model_name: object) -> str:
    return _model_label(model_name).replace(" ", "_").replace("/", "_")


_PAPER_METRICS: dict[str, dict[str, Any]] = {
    "f1": {
        "column": "f1",
        "filename": "f1_by_model_{task}.png",
        "label": "F1",
        "ascending": False,
        "color": "#1f77b4",
    },
    "prauc": {
        "column": "pr_auc",
        "filename": "prauc_by_model_{task}.png",
        "label": "PR-AUC",
        "ascending": False,
        "color": "#2ca02c",
    },
    "brier": {
        "column": "brier_score",
        "filename": "brier_by_model_{task}.png",
        "label": "Brier",
        "ascending": True,
        "color": "#d62728",
    },
    "train_time": {
        "column": "training_seconds",
        "filename": "train_time_by_model_{task}.png",
        "label": "Train Time (s)",
        "ascending": True,
        "color": "#9467bd",
    },
}


def _placeholder_plot(output_path: Path, title: str, message: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.4, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_barh(
    frame: pd.DataFrame,
    *,
    category_column: str,
    value_column: str,
    label: str,
    title: str,
    color: str,
    output_path: Path,
    ascending: bool,
) -> Path:
    plot_frame = frame[[category_column, value_column]].dropna().sort_values(value_column, ascending=ascending)
    if plot_frame.empty:
        return _placeholder_plot(output_path, title, "No hay datos suficientes para esta figura.")
    labels = plot_frame[category_column].map(_model_label) if category_column == "model" else plot_frame[category_column]

    height = max(4.0, 0.45 * len(plot_frame) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(labels, plot_frame[value_column], color=color, alpha=0.88)
    ax.set_xlabel(label)
    ax.set_ylabel(category_column.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    value_max = float(plot_frame[value_column].max()) if len(plot_frame) else 0.0
    x_pad = value_max * 0.02 if value_max else 0.02
    for index, (_, row) in enumerate(plot_frame.iterrows()):
        ax.text(float(row[value_column]) + x_pad, index, f"{float(row[value_column]):.4f}", va="center", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_heatmap(
    summary_test: pd.DataFrame,
    *,
    metric_column: str,
    title: str,
    output_path: Path,
    ascending: bool,
) -> Path:
    if summary_test.empty or metric_column not in summary_test.columns:
        return _placeholder_plot(output_path, title, "No hay datos suficientes para esta figura.")

    summary_test = summary_test.copy()
    summary_test["model_label"] = summary_test["model"].map(_model_label)
    task_order = list(dict.fromkeys(summary_test["task"].tolist()))
    model_order = list(dict.fromkeys(summary_test["model_label"].tolist()))
    pivot = summary_test.pivot_table(index="task", columns="model_label", values=metric_column, aggfunc="first")
    pivot = pivot.reindex(index=task_order, columns=model_order)
    if pivot.empty:
        return _placeholder_plot(output_path, title, "No hay datos suficientes para esta figura.")

    data = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = "viridis_r" if ascending else "viridis"
    width = max(9.0, 0.9 * len(model_order) + 3.5)
    height = max(4.5, 0.75 * len(task_order) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels(model_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(task_order)))
    ax.set_yticklabels(task_order)
    ax.set_title(title)
    for row_index in range(len(task_order)):
        for column_index in range(len(model_order)):
            value = data[row_index, column_index]
            label_text = "NA" if np.isnan(value) else f"{value:.4f}"
            ax.text(column_index, row_index, label_text, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_prevalence(validity_frame: pd.DataFrame, output_path: Path) -> Path:
    if validity_frame.empty:
        return _placeholder_plot(output_path, "Task prevalence", "No hay datos de prevalencia disponibles.")

    prevalence_columns = {
        "prevalence_global": "global",
        "prevalence_train": "train",
        "prevalence_val": "val",
        "prevalence_test": "test",
    }
    plot_frame = validity_frame[["task", "status", *prevalence_columns.keys()]].copy()
    melted = plot_frame.melt(
        id_vars=["task", "status"],
        value_vars=list(prevalence_columns.keys()),
        var_name="split",
        value_name="prevalence",
    )
    melted["split"] = melted["split"].map(prevalence_columns)
    melted["task_label"] = melted.apply(
        lambda row: row["task"] if row["status"] == "valid" else f"{row['task']} ({row['status']})",
        axis=1,
    )

    tasks = list(dict.fromkeys(melted["task_label"].tolist()))
    split_order = ["global", "train", "val", "test"]
    width = 0.2
    x_axis = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(max(10.0, 1.45 * len(tasks)), 5.5))
    colors = {"global": "#4c78a8", "train": "#72b7b2", "val": "#f58518", "test": "#e45756"}
    for offset, split_name in enumerate(split_order):
        split_frame = melted[melted["split"] == split_name].set_index("task_label").reindex(tasks)
        ax.bar(
            x_axis + ((offset - 1.5) * width),
            split_frame["prevalence"].fillna(0.0),
            width=width,
            label=split_name,
            color=colors[split_name],
            alpha=0.9,
        )
    ax.set_xticks(x_axis)
    ax.set_xticklabels(tasks, rotation=25, ha="right")
    ax.set_ylabel("Positive rate")
    ax.set_title("Prevalence by task")
    ax.legend()
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_best_model_by_task(best_frame: pd.DataFrame, output_path: Path) -> Path:
    required = {"task", "best_model_pr_auc", "best_pr_auc"}
    if best_frame.empty or not required.issubset(best_frame.columns):
        return _placeholder_plot(output_path, "Best model by task", "No hay ganadores disponibles.")

    plot_frame = best_frame[list(required)].dropna(subset=["best_pr_auc"]).sort_values("best_pr_auc", ascending=True)
    if plot_frame.empty:
        return _placeholder_plot(output_path, "Best model by task", "No hay ganadores disponibles.")

    height = max(4.0, 0.6 * len(plot_frame) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(plot_frame["task"], plot_frame["best_pr_auc"], color="#17becf", alpha=0.9)
    ax.set_xlabel("Best PR-AUC")
    ax.set_title("Best model by task")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    for index, (_, row) in enumerate(plot_frame.iterrows()):
        ax.text(
            float(row["best_pr_auc"]) + 0.01,
            index,
            f"{_model_label(row['best_model_pr_auc'])} ({float(row['best_pr_auc']):.4f})",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_roc_curve(y_true: pd.Series, probabilities: pd.Series, output_path: Path, title: str) -> Path:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return _placeholder_plot(output_path, title, "Split degenerado: no hay dos clases para ROC.")
    fig, ax = plt.subplots(figsize=(6, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RocCurveDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_pr_curve(y_true: pd.Series, probabilities: pd.Series, output_path: Path, title: str) -> Path:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return _placeholder_plot(output_path, title, "Split degenerado: no hay dos clases para PR.")
    fig, ax = plt.subplots(figsize=(6, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_calibration_curve(
    y_true: pd.Series,
    raw_probabilities: pd.Series,
    calibrated_probabilities: pd.Series | None,
    output_path: Path,
    title: str,
) -> Path:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return _placeholder_plot(output_path, title, "Split degenerado: no hay dos clases para calibracion.")
    fig, ax = plt.subplots(figsize=(6, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CalibrationDisplay.from_predictions(y_true, raw_probabilities, n_bins=10, ax=ax, name="raw")
        if calibrated_probabilities is not None:
            CalibrationDisplay.from_predictions(y_true, calibrated_probabilities, n_bins=10, ax=ax, name="calibrated")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_threshold_curve(
    y_true: pd.Series,
    probabilities: pd.Series,
    output_path: Path,
    title: str,
    selected_threshold: float | None = None,
) -> Path:
    sweep = compute_threshold_sweep(y_true, probabilities)
    if sweep.empty:
        return _placeholder_plot(output_path, title, "No hay datos suficientes para threshold sweep.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep["threshold"], sweep["precision"], label="precision")
    ax.plot(sweep["threshold"], sweep["recall"], label="recall")
    ax.plot(sweep["threshold"], sweep["f1"], label="f1")
    if selected_threshold is not None and not pd.isna(selected_threshold):
        ax.axvline(float(selected_threshold), color="black", linestyle="--", linewidth=1.0, label="selected")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def export_paper_figures(
    summary_test: pd.DataFrame,
    task_validity: pd.DataFrame,
    best_model_frame: pd.DataFrame,
    payloads: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    output_path = ensure_dir(output_dir)
    generated: list[Path] = []

    for task_name, task_frame in summary_test.groupby("task"):
        for config in _PAPER_METRICS.values():
            generated.append(
                _plot_barh(
                    task_frame,
                    category_column="model",
                    value_column=config["column"],
                    label=config["label"],
                    title=f"{config['label']} by model - {task_name}",
                    color=config["color"],
                    output_path=output_path / config["filename"].format(task=task_name),
                    ascending=bool(config["ascending"]),
                )
            )

    generated.append(
        _plot_heatmap(
            summary_test,
            metric_column="f1",
            title="F1 heatmap",
            output_path=output_path / "heatmap_f1.png",
            ascending=False,
        )
    )
    generated.append(
        _plot_heatmap(
            summary_test,
            metric_column="pr_auc",
            title="PR-AUC heatmap",
            output_path=output_path / "heatmap_prauc.png",
            ascending=False,
        )
    )
    generated.append(
        _plot_heatmap(
            summary_test,
            metric_column="brier_score",
            title="Brier heatmap",
            output_path=output_path / "heatmap_brier.png",
            ascending=True,
        )
    )
    generated.append(_plot_prevalence(task_validity, output_path / "prevalence_by_task.png"))
    generated.append(_plot_best_model_by_task(best_model_frame, output_path / "best_model_by_task.png"))

    for payload in payloads:
        prediction_path = Path(payload.get("artifacts", {}).get("prediction_path", ""))
        if not prediction_path.exists():
            continue
        frame = pd.read_csv(prediction_path)
        if "split" in frame.columns:
            frame = frame[frame["split"] == "test"].copy()
        if frame.empty or "y_true" not in frame.columns or "y_pred" not in frame.columns or "prob_class_1" not in frame.columns:
            continue

        task_name = str(payload.get("task"))
        model_name = str(payload.get("model"))
        model_label = _model_label(model_name)
        model_slug = _model_slug(model_name)
        y_true = frame["y_true"].astype(int)
        y_pred = frame["y_pred"].astype(int)
        raw_probabilities = frame["prob_class_1"].astype(float)
        calibrated_probabilities = (
            frame["prob_class_1_calibrated"].astype(float) if "prob_class_1_calibrated" in frame.columns else None
        )
        threshold_raw = payload.get("posthoc", {}).get("thresholds", {}).get("raw")

        generated.append(
            _plot_roc_curve(
                y_true,
                raw_probabilities,
                output_path / f"roc_{task_name}_{model_slug}.png",
                f"ROC - {task_name} - {model_label}",
            )
        )
        generated.append(
            _plot_pr_curve(
                y_true,
                raw_probabilities,
                output_path / f"pr_{task_name}_{model_slug}.png",
                f"PR - {task_name} - {model_label}",
            )
        )
        generated.append(
            _plot_calibration_curve(
                y_true,
                raw_probabilities,
                calibrated_probabilities,
                output_path / f"calibration_{task_name}_{model_slug}.png",
                f"Calibration - {task_name} - {model_label}",
            )
        )
        generated.append(
            _plot_confusion_matrix(
                y_true,
                y_pred,
                output_path / f"confusion_{task_name}_{model_slug}.png",
                f"Confusion - {task_name} - {model_label}",
            )
        )
        generated.append(
            _plot_threshold_curve(
                y_true,
                raw_probabilities,
                output_path / f"threshold_sweep_{task_name}_{model_slug}.png",
                f"Threshold sweep - {task_name} - {model_label}",
                selected_threshold=float(threshold_raw) if threshold_raw is not None else None,
            )
        )
    return generated


def generate_classification_plots(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.DataFrame,
    output_dir: str | Path,
    prefix: str,
) -> None:
    output_path = ensure_dir(output_dir)
    _plot_confusion_matrix(y_true, y_pred, output_path / f"{prefix}_confusion_matrix.png", prefix)
    if y_prob.shape[1] >= 2:
        positive_probabilities = y_prob.iloc[:, 1]
        _plot_roc_curve(y_true, positive_probabilities, output_path / f"{prefix}_roc_curve.png", prefix)
        _plot_pr_curve(y_true, positive_probabilities, output_path / f"{prefix}_pr_curve.png", prefix)
