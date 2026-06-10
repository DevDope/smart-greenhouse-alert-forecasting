"""Post-hoc calibration and threshold tuning utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..evaluation.diagnostics import compute_threshold_sweep
from ..evaluation.metrics_classification import compute_classification_metrics


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(probabilities.astype(np.float64), 1e-6, 1.0 - 1e-6)


def _binary_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = _clip_probabilities(probabilities)
    return np.log(clipped / (1.0 - clipped))


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, y_true: pd.Series, y_prob: pd.DataFrame, max_steps: int = 100) -> "TemperatureScaler":
        target = pd.Series(y_true).astype(int).to_numpy()
        prob_matrix = y_prob.to_numpy(dtype=np.float64)
        if prob_matrix.shape[1] != 2:
            self.temperature = 1.0
            return self

        logits = torch.tensor(_binary_logit(prob_matrix[:, 1]), dtype=torch.float32)
        labels = torch.tensor(target.astype(np.float32), dtype=torch.float32)
        log_temperature = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_steps)

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            temperature = torch.exp(log_temperature).clamp_min(1e-3)
            loss = nn.functional.binary_cross_entropy_with_logits(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temperature).detach().cpu().item())
        return self

    def transform(self, y_prob: pd.DataFrame) -> pd.DataFrame:
        prob_matrix = y_prob.to_numpy(dtype=np.float64)
        if prob_matrix.shape[1] != 2 or abs(self.temperature - 1.0) < 1e-8:
            return y_prob.copy()
        logits = _binary_logit(prob_matrix[:, 1]) / max(self.temperature, 1e-6)
        positive = 1.0 / (1.0 + np.exp(-logits))
        calibrated = np.column_stack([1.0 - positive, positive])
        return pd.DataFrame(calibrated, index=y_prob.index, columns=y_prob.columns)


def select_optimal_threshold(
    y_true: pd.Series,
    positive_probabilities: pd.Series,
    metric: str = "f1",
) -> float:
    sweep = compute_threshold_sweep(y_true, positive_probabilities)
    ordered = sweep.sort_values([metric, "precision", "recall", "threshold"], ascending=[False, False, False, True])
    return float(ordered.iloc[0]["threshold"])


def metrics_at_threshold(
    y_true: pd.Series,
    y_prob: pd.DataFrame,
    threshold: float,
) -> tuple[pd.Series, dict[str, float | None]]:
    positive_probs = y_prob.iloc[:, 1]
    prediction = (positive_probs >= threshold).astype(int)
    metrics = compute_classification_metrics(y_true, prediction, y_prob)
    metrics["threshold"] = float(threshold)
    return prediction.rename("prediction"), metrics
