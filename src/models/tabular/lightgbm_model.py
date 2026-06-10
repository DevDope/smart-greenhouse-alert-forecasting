"""LightGBM wrapper with sklearn fallback."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ..base import BaseModel

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    LGBMClassifier = None


class LightGBMClassifier(BaseModel):
    model_name = "lightgbm"

    def __init__(self, **params: float | int) -> None:
        self.params = params
        self.estimator = None
        self.classes_: list[int] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series | None = None) -> "LightGBMClassifier":
        target = y.astype(int)
        self.classes_ = sorted(target.unique().tolist())
        if LGBMClassifier is not None:
            objective = "binary" if len(self.classes_) == 2 else "multiclass"
            estimator_kwargs = dict(self.params)
            estimator_kwargs["objective"] = objective
            estimator_kwargs.setdefault("verbosity", -1)
            if len(self.classes_) > 2:
                estimator_kwargs["num_class"] = len(self.classes_)
            estimator = LGBMClassifier(**estimator_kwargs)
            self.estimator = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", estimator),
                ]
            )
        else:
            self.estimator = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", HistGradientBoostingClassifier(random_state=int(self.params.get("random_state", 42)))),
                ]
            )
        self.estimator.fit(X, target)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.estimator.predict(X), index=X.index, name="prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.estimator.predict_proba(X)
        columns = [f"class_{int(cls)}" for cls in self.classes_ or []]
        return pd.DataFrame(probabilities, index=X.index, columns=columns)
