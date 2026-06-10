"""Command line interface."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .experiments.run_experiment import evaluate_experiment, prepare_all_datasets, run_experiment
from .experiments.resumable import run_resumable_experiment
from .experiments.summarize_results import summarize_results
from .utils.config import resolve_experiment_config
from .utils.logger import get_logger

LOGGER = get_logger(__name__)


def _compact_run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    return {
        "run_id": payload.get("run_id"),
        "canonical_dataset": payload.get("canonical_dataset"),
        "result_count": len(results),
        "tasks": sorted({result.get("task") for result in results}),
        "models": sorted({result.get("model") for result in results}),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Greenhouse event forecasting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("prepare-data", "train", "evaluate", "run-experiment", "run-resumable", "report"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, help="Path to experiment YAML")
        subparser.add_argument("--run-id", required=False, help="Optional existing or custom run id")
        subparser.add_argument("--seed", required=False, type=int, help="Optional seed override")
        subparser.add_argument("--view", required=False, help="Run a single weather source view")
        subparser.add_argument("--task", required=False, help="Run a single task")
        subparser.add_argument("--model", required=False, help="Run a single model")
        subparser.add_argument("--fail-fast", action="store_true", help="Stop resumable runs at first error")
        subparser.add_argument(
            "--override",
            action="append",
            default=[],
            help="Override config values with dotted.key=value",
        )
    return parser


def _resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    overrides = list(args.override or [])
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    return resolve_experiment_config(args.config, overrides=overrides)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _resolved_config(args)

    if args.command == "prepare-data":
        if "weather_source_views" in config:
            prepared_views = []
            for view in config["weather_source_views"]:
                view_config = dict(config)
                view_config.pop("weather_source_views", None)
                view_config["name"] = f"{config['name']}__{view['name']}"
                view_config["weather_source_view"] = view["name"]
                view_config["data"] = view["data"]
                view_config["tasks"] = view["tasks"]
                canonical_dataset, datasets = prepare_all_datasets(view_config)
                prepared_views.append(
                    {
                        "weather_source_view": view["name"],
                        "canonical": {
                            "canonical_id": canonical_dataset.canonical_id,
                            "parquet_path": str(canonical_dataset.data_path),
                            "csv_path": str(canonical_dataset.csv_path),
                            "time_summary": canonical_dataset.metadata["time_summary"],
                            "columns_final": canonical_dataset.metadata["columns_final"],
                            "missing_optional_columns": canonical_dataset.metadata["missing_optional_columns"],
                        },
                        "prepared": [
                            {"task": dataset.task_name, "dataset_id": dataset.dataset_id, "path": str(dataset.data_path)}
                            for dataset in datasets
                        ],
                    }
                )
            print(json.dumps({"weather_source_views": prepared_views}, indent=2))
            return

        canonical_dataset, datasets = prepare_all_datasets(config)
        payload = {
            "canonical": {
                "canonical_id": canonical_dataset.canonical_id,
                "parquet_path": str(canonical_dataset.data_path),
                "csv_path": str(canonical_dataset.csv_path),
                "time_summary": canonical_dataset.metadata["time_summary"],
                "columns_final": canonical_dataset.metadata["columns_final"],
                "missing_optional_columns": canonical_dataset.metadata["missing_optional_columns"],
            },
            "prepared": [
                {"task": dataset.task_name, "dataset_id": dataset.dataset_id, "path": str(dataset.data_path)}
                for dataset in datasets
            ]
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "train":
        result = run_experiment(
            config,
            run_id=args.run_id,
            only_task=args.task,
            only_model=args.model,
        )
        print(json.dumps(_compact_run_payload(result), indent=2))
        return

    if args.command == "evaluate":
        if not args.run_id:
            raise ValueError("--run-id is required for evaluate.")
        result = evaluate_experiment(
            config,
            run_id=args.run_id,
            only_task=args.task,
            only_model=args.model,
        )
        print(json.dumps(_compact_run_payload(result), indent=2))
        return

    if args.command == "run-experiment":
        result = run_experiment(
            config,
            run_id=args.run_id,
            only_task=args.task,
            only_model=args.model,
        )
        print(json.dumps(_compact_run_payload(result), indent=2))
        return

    if args.command == "run-resumable":
        result = run_resumable_experiment(
            config,
            run_id=args.run_id,
            only_view=args.view,
            only_task=args.task,
            only_model=args.model,
            continue_on_error=not args.fail_fast,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "report":
        report_path = summarize_results(config, run_id=args.run_id)
        print(json.dumps({"report_path": str(report_path)}, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
