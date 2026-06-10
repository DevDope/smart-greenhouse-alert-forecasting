"""Classification metrics with safe fallbacks."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _safe_metric(func: Any, *args: Any, **kwargs: Any) -> float | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(func(*args, **kwargs))
    except Exception:
        return None


def _probability_matrix(y_prob: pd.DataFrame) -> np.ndarray:
    if isinstance(y_prob, pd.DataFrame):
        return y_prob.to_numpy()
    return np.asarray(y_prob)


def expected_calibration_error(
    y_true: pd.Series,
    y_prob: pd.DataFrame,
    n_bins: int = 10,
) -> float | None:
    prob_matrix = _probability_matrix(y_prob)
    if prob_matrix.size == 0:
        return None
    target = pd.Series(y_true).astype(int).to_numpy()
    if prob_matrix.shape[1] == 2:
        confidences = prob_matrix[:, 1]
        predictions = (confidences >= 0.5).astype(int)
        correctness = (predictions == target).astype(float)
    else:
        predictions = np.argmax(prob_matrix, axis=1)
        confidences = prob_matrix[np.arange(len(prob_matrix)), predictions]
        correctness = (predictions == target).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidences >= lower) & (confidences < upper if upper < 1.0 else confidences <= upper)
        if not np.any(in_bin):
            continue
        bin_accuracy = correctness[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        ece += np.abs(bin_accuracy - bin_confidence) * (np.sum(in_bin) / len(confidences))
    return float(ece)


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.DataFrame,
) -> dict[str, float | None]:
    target = pd.Series(y_true).astype(int)
    prediction = pd.Series(y_pred).astype(int)
    prob_matrix = _probability_matrix(y_prob)
    classes = np.arange(prob_matrix.shape[1])
    is_binary = prob_matrix.shape[1] == 2

    metrics: dict[str, float | None] = {
        "f1": _safe_metric(
            f1_score,
            target,
            prediction,
            average="binary" if is_binary else "macro",
            zero_division=0,
        ),
        "precision": _safe_metric(
            precision_score,
            target,
            prediction,
            average="binary" if is_binary else "macro",
            zero_division=0,
        ),
        "recall": _safe_metric(
            recall_score,
            target,
            prediction,
            average="binary" if is_binary else "macro",
            zero_division=0,
        ),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, target, prediction),
        "f1_macro": _safe_metric(f1_score, target, prediction, average="macro", zero_division=0),
        "f1_weighted": _safe_metric(f1_score, target, prediction, average="weighted", zero_division=0),
        "precision_macro": _safe_metric(precision_score, target, prediction, average="macro", zero_division=0),
        "recall_macro": _safe_metric(recall_score, target, prediction, average="macro", zero_division=0),
    }

    if is_binary and prob_matrix.shape[1] >= 2:
        positive_probs = prob_matrix[:, 1]
        metrics["pr_auc"] = _safe_metric(average_precision_score, target, positive_probs)
        metrics["roc_auc"] = _safe_metric(roc_auc_score, target, positive_probs)
        metrics["brier_score"] = _safe_metric(brier_score_loss, target, positive_probs)
    else:
        target_matrix = label_binarize(target, classes=classes)
        metrics["pr_auc"] = _safe_metric(
            average_precision_score,
            target_matrix,
            prob_matrix,
            average="macro",
        )
        metrics["roc_auc"] = _safe_metric(
            roc_auc_score,
            target_matrix,
            prob_matrix,
            multi_class="ovr",
            average="macro",
        )
        metrics["brier_score"] = float(np.mean(np.sum((target_matrix - prob_matrix) ** 2, axis=1)))
    metrics["ece"] = expected_calibration_error(target, y_prob)
    return metrics


def compute_classification_report(
    y_true: pd.Series,
    y_pred: pd.Series,
    class_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    target = pd.Series(y_true).astype(int)
    prediction = pd.Series(y_pred).astype(int)
    labels = sorted(target.unique().tolist())
    if class_names is not None and len(class_names) >= max(labels, default=-1) + 1:
        target_names = [class_names[label] for label in labels]
    else:
        target_names = [str(label) for label in labels]
    return classification_report(
        target,
        prediction,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
