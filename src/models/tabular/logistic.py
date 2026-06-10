"""Logistic regression baseline."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..base import BaseModel


class LogisticClassifier(BaseModel):
    model_name = "logistic"

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | dict[str, float] | None = "balanced",
        solver: str = "lbfgs",
    ) -> None:
        self.estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        class_weight=class_weight,
                        solver=solver,
                        multi_class="auto",
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series | None = None) -> "LogisticClassifier":
        self.estimator.fit(X, y.astype(int))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.estimator.predict(X), index=X.index, name="prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.estimator.predict_proba(X)
        columns = [f"class_{int(cls)}" for cls in self.estimator.named_steps["model"].classes_]
        return pd.DataFrame(probabilities, index=X.index, columns=columns)
