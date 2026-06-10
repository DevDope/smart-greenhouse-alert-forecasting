"""Forecasting metrics for future continuous tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_forecasting_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    true = y_true.astype(float).to_numpy()
    pred = y_pred.astype(float).to_numpy()
    mae = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    denominator = np.maximum((np.abs(true) + np.abs(pred)) / 2.0, 1e-8)
    smape = float(np.mean(np.abs(true - pred) / denominator) * 100.0)
    return {"mae": mae, "rmse": rmse, "smape": smape}
