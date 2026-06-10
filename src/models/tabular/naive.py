"""Naive baseline classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseModel


class NaiveClassifier(BaseModel):
    model_name = "naive"

    def __init__(self, strategy: str = "most_frequent") -> None:
        self.strategy = strategy
        self.classes_: np.ndarray | None = None
        self.most_frequent_class_: int | None = None
        self.class_probabilities_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series | None = None) -> "NaiveClassifier":
        target = pd.Series(y).astype(int)
        self.classes_ = np.sort(target.unique())
        counts = target.value_counts().sort_index()
        self.most_frequent_class_ = int(counts.idxmax())
        probs = np.zeros(len(self.classes_), dtype=float)
        for index, cls in enumerate(self.classes_):
            probs[index] = counts.get(cls, 0) / len(target)
        self.class_probabilities_ = probs
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.most_frequent_class_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return pd.Series([self.most_frequent_class_] * len(X), index=X.index, name="prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.classes_ is None or self.class_probabilities_ is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        repeated = np.tile(self.class_probabilities_, (len(X), 1))
        columns = [f"class_{int(cls)}" for cls in self.classes_]
        return pd.DataFrame(repeated, index=X.index, columns=columns)
