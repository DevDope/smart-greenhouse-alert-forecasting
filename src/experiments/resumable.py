"""Resumable experiment runner."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_experiment import run_experiment
from ..utils.reproducibility import make_run_id, save_json


def _single_view_config(resolved_config: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
    view_name = str(view["name"])
    config = copy.deepcopy(resolved_config)
    config.pop("weather_source_views", None)
    config["name"] = f"{resolved_config['name']}__{view_name}"
    config["weather_source_view"] = view_name
    config["data"] = view["data"]
    config["tasks"] = view["tasks"]
    return config


def _metric_path(config: dict[str, Any], run_id: str, task_name: str, model_name: str) -> Path:
    return Path(config["artifacts"]["results_dir"]) / "metrics" / run_id / task_name / f"{model_name}.json"


def _manifest_path(resolved_config: dict[str, Any], run_id: str) -> Path:
    return Path(resolved_config["artifacts"]["results_dir"]) / "metrics" / run_id / "progress_manifest.json"


def _write_progress(
    resolved_config: dict[str, Any],
    run_id: str,
    rows: list[dict[str, Any]],
) -> None:
    save_json(
        {
            "run_id": run_id,
            "experiment": resolved_config["name"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "completed": sum(row["status"] in {"completed", "skipped_existing"} for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "pending": sum(row["status"] == "pending" for row in rows),
            "rows": rows,
        },
        _manifest_path(resolved_config, run_id),
    )


def iter_run_units(
    resolved_config: dict[str, Any],
    *,
    only_view: str | None = None,
    only_task: str | None = None,
    only_model: str | None = None,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    if "weather_source_views" in resolved_config:
        for view in resolved_config["weather_source_views"]:
            view_name = str(view["name"])
            if only_view and view_name != only_view:
                continue
            view_config = _single_view_config(resolved_config, view)
            for task in view_config["tasks"]:
                task_name = str(task["name"])
                if only_task and task_name != only_task:
                    continue
                for model_name in view_config["models"]:
                    if only_model and model_name != only_model:
                        continue
                    units.append(
                        {
                            "view": view_name,
                            "config": view_config,
                            "task": task_name,
                            "model": model_name,
                        }
                    )
        return units

    for task in resolved_config["tasks"]:
        task_name = str(task["name"])
        if only_task and task_name != only_task:
            continue
        for model_name in resolved_config["models"]:
            if only_model and model_name != only_model:
                continue
            units.append(
                {
                    "view": resolved_config.get("weather_source_view"),
                    "config": resolved_config,
                    "task": task_name,
                    "model": model_name,
                }
            )
    return units


def run_resumable_experiment(
    resolved_config: dict[str, Any],
    *,
    run_id: str | None = None,
    only_view: str | None = None,
    only_task: str | None = None,
    only_model: str | None = None,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    base_run_id = run_id or make_run_id(resolved_config["name"], resolved_config)
    units = iter_run_units(
        resolved_config,
        only_view=only_view,
        only_task=only_task,
        only_model=only_model,
    )
    rows = [
        {
            "view": unit["view"],
            "run_id": f"{base_run_id}__{unit['view']}" if unit["view"] else base_run_id,
            "task": unit["task"],
            "model": unit["model"],
            "status": "pending",
        }
        for unit in units
    ]
    _write_progress(resolved_config, base_run_id, rows)
    results: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        unit_run_id = f"{base_run_id}__{unit['view']}" if unit["view"] else base_run_id
        metric_path = _metric_path(unit["config"], unit_run_id, unit["task"], unit["model"])
        rows[index]["metric_path"] = str(metric_path)
        if metric_path.exists():
            rows[index]["status"] = "skipped_existing"
            _write_progress(resolved_config, base_run_id, rows)
            continue
        try:
            payload = run_experiment(
                unit["config"],
                run_id=unit_run_id,
                only_task=unit["task"],
                only_model=unit["model"],
            )
            rows[index]["status"] = "completed"
            results.extend(payload.get("results", []))
        except Exception as exc:
            rows[index]["status"] = "failed"
            rows[index]["error"] = f"{type(exc).__name__}: {exc}"
            _write_progress(resolved_config, base_run_id, rows)
            if not continue_on_error:
                raise
        _write_progress(resolved_config, base_run_id, rows)
    return {
        "run_id": base_run_id,
        "result_count": len(results),
        "progress_manifest": str(_manifest_path(resolved_config, base_run_id)),
        "completed": sum(row["status"] in {"completed", "skipped_existing"} for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "total": len(rows),
    }
