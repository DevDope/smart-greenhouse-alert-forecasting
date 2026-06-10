"""XGBoost wrapper with sklearn fallback."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ..base import BaseModel

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None


class XGBoostClassifier(BaseModel):
    model_name = "xgboost"

    def __init__(self, **params: float | int) -> None:
        self.params = params
        self.estimator = None
        self.classes_: list[int] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series | None = None) -> "XGBoostClassifier":
        target = y.astype(int)
        self.classes_ = sorted(target.unique().tolist())
        if XGBClassifier is not None:
            objective = "binary:logistic" if len(self.classes_) == 2 else "multi:softprob"
            estimator = XGBClassifier(
                objective=objective,
                num_class=None if len(self.classes_) == 2 else len(self.classes_),
                eval_metric="logloss" if len(self.classes_) == 2 else "mlogloss",
                **self.params,
            )
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
