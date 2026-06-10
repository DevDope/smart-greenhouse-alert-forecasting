"""Common model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class BaseModel(ABC):
    """Shared interface for all models."""

    model_name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series | None = None) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        return joblib.load(path)


class NotImplementedModel(BaseModel):
    """Placeholder model registered for future work."""

    def __init__(self, model_name: str, params: dict[str, Any] | None = None) -> None:
        self.model_name = model_name
        self.params = params or {}

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight: pd.Series | None = None) -> "BaseModel":
        raise NotImplementedError(f"Model '{self.model_name}' is registered as a skeleton only.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError(f"Model '{self.model_name}' is registered as a skeleton only.")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(f"Model '{self.model_name}' is registered as a skeleton only.")
