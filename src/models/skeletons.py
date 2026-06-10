"""Registered skeletons for future model families."""

from __future__ import annotations

from typing import Any

from .base import NotImplementedModel


def build_skeleton(model_name: str, params: dict[str, Any] | None = None) -> NotImplementedModel:
    return NotImplementedModel(model_name=model_name, params=params)
