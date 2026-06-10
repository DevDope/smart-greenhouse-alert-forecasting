"""Experiment execution helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..data.datasets import CanonicalDataset, PreparedDataset, prepare_canonical_dataset, prepare_task_dataset
from ..training.trainer import evaluate_saved_model, train_and_evaluate_model
from ..utils.reproducibility import make_run_id, save_json
from ..utils.seed import set_global_seed


def prepare_all_datasets(resolved_config: dict[str, Any]) -> tuple[CanonicalDataset, list[PreparedDataset]]:
    if "weather_source_views" in resolved_config:
        raise ValueError("prepare_all_datasets requires a single resolved data view, not weather_source_views.")
    canonical_dataset = prepare_canonical_dataset(resolved_config)
    prepared_datasets = [
        prepare_task_dataset(resolved_config, task_config, canonical_dataset)
        for task_config in resolved_config["tasks"]
    ]
    return canonical_dataset, prepared_datasets


def _view_config(resolved_config: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
    view_name = str(view["name"])
    config = copy.deepcopy(resolved_config)
    config.pop("weather_source_views", None)
    config["name"] = f"{resolved_config['name']}__{view_name}"
    config["weather_source_view"] = view_name
    config["data"] = view["data"]
    config["tasks"] = view["tasks"]
    return config


def run_experiment(
    resolved_config: dict[str, Any],
    run_id: str | None = None,
    only_task: str | None = None,
    only_model: str | None = None,
) -> dict[str, Any]:
    if "weather_source_views" in resolved_config:
        view_payloads = []
        combined_results = []
        base_run_id = run_id
        for view in resolved_config["weather_source_views"]:
            view_name = str(view["name"])
            view_run_id = f"{base_run_id}__{view_name}" if base_run_id else None
            payload = run_experiment(
                _view_config(resolved_config, view),
                run_id=view_run_id,
                only_task=only_task,
                only_model=only_model,
            )
            payload["weather_source_view"] = view_name
            view_payloads.append(payload)
            combined_results.extend(
                {**result, "weather_source_view": view_name}
                for result in payload.get("results", [])
            )
        return {
            "run_id": base_run_id or "__".join(payload["run_id"] for payload in view_payloads),
            "weather_source_views": view_payloads,
            "results": combined_results,
        }

    seed = int(resolved_config.get("seed", 42))
    set_global_seed(seed)
    run_id = run_id or make_run_id(resolved_config["name"], resolved_config)
    results_dir = Path(resolved_config["artifacts"]["results_dir"])
    save_json({"seed": seed, "resolved_config": resolved_config}, results_dir / "metrics" / run_id / "manifest.json")

    canonical_dataset, prepared_datasets = prepare_all_datasets(resolved_config)
    payloads: list[dict[str, Any]] = []
    for prepared in prepared_datasets:
        if only_task and prepared.task_name != only_task:
            continue
        for model_name, model_config in resolved_config["models"].items():
            if only_model and model_name != only_model:
                continue
            payloads.append(
                train_and_evaluate_model(
                    prepared,
                    model_config,
                    resolved_config,
                    run_id,
                    peer_datasets=prepared_datasets,
                )
            )
    return {
        "run_id": run_id,
        "canonical_dataset": {
            "canonical_id": canonical_dataset.canonical_id,
            "path": str(canonical_dataset.data_path),
        },
        "results": payloads,
    }


def evaluate_experiment(
    resolved_config: dict[str, Any],
    run_id: str,
    only_task: str | None = None,
    only_model: str | None = None,
) -> dict[str, Any]:
    if "weather_source_views" in resolved_config:
        view_payloads = []
        combined_results = []
        for view in resolved_config["weather_source_views"]:
            view_name = str(view["name"])
            payload = evaluate_experiment(
                _view_config(resolved_config, view),
                run_id=f"{run_id}__{view_name}",
                only_task=only_task,
                only_model=only_model,
            )
            payload["weather_source_view"] = view_name
            view_payloads.append(payload)
            combined_results.extend(
                {**result, "weather_source_view": view_name}
                for result in payload.get("results", [])
            )
        return {
            "run_id": run_id,
            "weather_source_views": view_payloads,
            "results": combined_results,
        }

    canonical_dataset, prepared_datasets = prepare_all_datasets(resolved_config)
    payloads: list[dict[str, Any]] = []
    for prepared in prepared_datasets:
        if only_task and prepared.task_name != only_task:
            continue
        for model_name, model_config in resolved_config["models"].items():
            if only_model and model_name != only_model:
                continue
            payloads.append(evaluate_saved_model(prepared, model_config, resolved_config, run_id))
    return {
        "run_id": run_id,
        "canonical_dataset": {
            "canonical_id": canonical_dataset.canonical_id,
            "path": str(canonical_dataset.data_path),
        },
        "results": payloads,
    }
