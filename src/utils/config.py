"""Configuration utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .paths import project_root


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating inputs."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in YAML file: {resolved}")
    if "extends" in data:
        base = load_yaml(data.pop("extends"))
        return merge_dicts(base, data)
    return data


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root() / candidate


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Use key=value syntax.")
        dotted_key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = updated
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
    return updated


def resolve_experiment_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    experiment = apply_overrides(load_yaml(path), overrides)
    data_config = load_yaml(experiment["data_config"]) if "data_config" in experiment else None
    task_configs = [load_yaml(task_path) for task_path in experiment.get("task_configs", [])]
    model_configs = [load_yaml(model_path) for model_path in experiment["model_configs"]]

    resolved = copy.deepcopy(experiment)
    if data_config is not None:
        resolved["data"] = data_config
    resolved["tasks"] = task_configs
    resolved["models"] = {model["name"]: model for model in model_configs}
    if "weather_source_views" in resolved:
        resolved_views = []
        for view in resolved["weather_source_views"]:
            view_resolved = copy.deepcopy(view)
            view_resolved["data"] = load_yaml(view["data_config"])
            view_task_paths = view.get("task_configs", experiment.get("task_configs", []))
            view_resolved["tasks"] = [load_yaml(task_path) for task_path in view_task_paths]
            resolved_views.append(view_resolved)
        resolved["weather_source_views"] = resolved_views
    return apply_overrides(resolved, overrides)
