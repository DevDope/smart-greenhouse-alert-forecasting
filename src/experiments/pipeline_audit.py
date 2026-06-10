"""Strict pipeline audit for paper/Q1 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.datasets import PreparedDataset
from ..utils.config import load_yaml, resolve_experiment_config
from ..utils.paths import project_root
from .run_experiment import prepare_all_datasets


DANGEROUS_TEXCOCO_GLOBS = {
    "data/raw/texcoco/*.csv",
    "data/raw/texcoco/*.parquet",
}
EXPECTED_WEATHER_VIEWS = {"openmeteo", "nasa", "fused"}


@dataclass
class AuditFinding:
    check: str
    status: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "status": self.status,
            "message": self.message,
            "path": self.path,
        }


def _experiment_paths() -> list[Path]:
    return sorted((project_root() / "configs" / "experiments").glob("*.yaml"))


def _is_q1_or_paper(path: Path, config: dict[str, Any]) -> bool:
    name = str(config.get("name", path.stem)).lower()
    return "q1" in name or "paper" in name or "paper_eval" in config


def _file_patterns(data_config: dict[str, Any]) -> list[str]:
    return [str(pattern).replace("\\", "/") for pattern in data_config.get("source", {}).get("file_patterns", [])]


def _contains_dangerous_glob(data_config: dict[str, Any]) -> bool:
    return any(pattern in DANGEROUS_TEXCOCO_GLOBS for pattern in _file_patterns(data_config))


def _data_configs_for_experiment(resolved: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if "weather_source_views" in resolved:
        return [(str(view["name"]), view["data"]) for view in resolved["weather_source_views"]]
    return [(str(resolved.get("weather_source_view", "default")), resolved["data"])]


def audit_config_resolution(paths: list[Path] | None = None) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for path in paths or _experiment_paths():
        try:
            resolve_experiment_config(path)
        except Exception as exc:
            findings.append(AuditFinding("config_resolution", "fail", f"{type(exc).__name__}: {exc}", str(path)))
        else:
            findings.append(AuditFinding("config_resolution", "pass", "resolved", str(path)))
    return findings


def audit_q1_paper_sources(paths: list[Path] | None = None) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for path in paths or _experiment_paths():
        raw = load_yaml(path)
        if not _is_q1_or_paper(path, raw):
            continue
        resolved = resolve_experiment_config(path)
        for view_name, data_config in _data_configs_for_experiment(resolved):
            if _contains_dangerous_glob(data_config):
                findings.append(
                    AuditFinding(
                        "q1_paper_sources",
                        "fail",
                        f"{raw.get('name', path.stem)} view {view_name} uses a dangerous Texcoco glob.",
                        str(path),
                    )
                )
            else:
                findings.append(
                    AuditFinding(
                        "q1_paper_sources",
                        "pass",
                        f"{raw.get('name', path.stem)} view {view_name} uses explicit source files.",
                        str(path),
                    )
                )
    return findings


def audit_weather_views(path: str | Path = "configs/experiments/exp17_q1_enriched_weather_sources.yaml") -> list[AuditFinding]:
    resolved = resolve_experiment_config(path)
    views = {str(view["name"]): view["data"] for view in resolved.get("weather_source_views", [])}
    findings: list[AuditFinding] = []
    if set(views) != EXPECTED_WEATHER_VIEWS:
        findings.append(AuditFinding("weather_views", "fail", f"Expected views {sorted(EXPECTED_WEATHER_VIEWS)}, got {sorted(views)}.", str(path)))
        return findings
    for view_name, data_config in views.items():
        declared = data_config.get("weather_source_view")
        if declared != view_name:
            findings.append(AuditFinding("weather_views", "fail", f"View {view_name} declares {declared}.", str(path)))
            continue
        features = set(data_config.get("numeric_features", []))
        if view_name == "openmeteo" and any(feature.startswith(("outside_nasa_", "outside_fused_")) for feature in features):
            findings.append(AuditFinding("weather_views", "fail", "Open-Meteo view contains NASA or fused features.", str(path)))
        elif view_name == "nasa" and any(feature.startswith(("outside_openmeteo_", "outside_fused_")) for feature in features):
            findings.append(AuditFinding("weather_views", "fail", "NASA view contains Open-Meteo or fused features.", str(path)))
        elif view_name == "fused" and not any(feature.startswith("outside_fused_") for feature in features):
            findings.append(AuditFinding("weather_views", "fail", "Fused view lacks fused features.", str(path)))
        else:
            findings.append(AuditFinding("weather_views", "pass", f"{view_name} view is source-separated.", str(path)))
    return findings


def _validate_prepared_dataset(prepared: PreparedDataset) -> list[str]:
    errors: list[str] = []
    frame = prepared.frame.sort_values("timestamp").reset_index(drop=True)
    label = frame[prepared.label_column].dropna()
    if label.empty:
        errors.append("label has no non-null values")
    elif float(label.mean()) <= 0.0:
        errors.append("positive rate is zero")
    for split_name in ("train", "val", "test"):
        split_frame = frame[frame["split"] == split_name]
        if split_frame.empty:
            errors.append(f"{split_name} split is empty")
            continue
        timestamps = pd.to_datetime(split_frame["timestamp"])
        if not timestamps.is_monotonic_increasing:
            errors.append(f"{split_name} split is not chronological")
        if not (pd.to_datetime(split_frame["label_end"]) > pd.to_datetime(split_frame["observation_end"])).all():
            errors.append(f"{split_name} has label_end not in the future")
    split_bounds = []
    for split_name in ("train", "val", "test"):
        split_frame = frame[frame["split"] == split_name]
        if not split_frame.empty:
            split_bounds.append((split_name, pd.to_datetime(split_frame["timestamp"]).min(), pd.to_datetime(split_frame["timestamp"]).max()))
    for (_, _, left_end), (right_name, right_start, _) in zip(split_bounds, split_bounds[1:]):
        if left_end >= right_start:
            errors.append(f"split order overlaps before {right_name}")
    return errors


def audit_prepared_datasets(path: str | Path) -> list[AuditFinding]:
    resolved = resolve_experiment_config(path)
    findings: list[AuditFinding] = []
    view_configs: list[tuple[str, dict[str, Any]]]
    if "weather_source_views" in resolved:
        view_configs = []
        for view in resolved["weather_source_views"]:
            view_config = dict(resolved)
            view_config.pop("weather_source_views", None)
            view_config["name"] = f"{resolved['name']}__{view['name']}"
            view_config["weather_source_view"] = view["name"]
            view_config["data"] = view["data"]
            view_config["tasks"] = view["tasks"]
            view_configs.append((str(view["name"]), view_config))
    else:
        view_configs = [(str(resolved.get("weather_source_view", "default")), resolved)]
    for view_name, view_config in view_configs:
        _, prepared_datasets = prepare_all_datasets(view_config)
        for prepared in prepared_datasets:
            errors = _validate_prepared_dataset(prepared)
            findings.append(
                AuditFinding(
                    "prepared_dataset",
                    "fail" if errors else "pass",
                    f"{view_name}/{prepared.task_name}: " + ("; ".join(errors) if errors else "valid"),
                    str(path),
                )
            )
    return findings


def audit_report_dependencies() -> list[AuditFinding]:
    report_path = project_root() / "src" / "evaluation" / "reports.py"
    text = report_path.read_text(encoding="utf-8")
    forbidden = [name for name in ("pyarrow", "fastparquet", "tabulate") if name in text]
    if forbidden:
        return [AuditFinding("report_dependencies", "fail", f"reports.py references optional dependencies: {forbidden}.", str(report_path))]
    return [AuditFinding("report_dependencies", "pass", "reports.py does not import pyarrow, fastparquet or tabulate.", str(report_path))]


def run_pipeline_audit(
    *,
    dataset_experiments: list[str | Path] | None = None,
) -> dict[str, Any]:
    findings = []
    findings.extend(audit_config_resolution())
    findings.extend(audit_q1_paper_sources())
    findings.extend(audit_weather_views("configs/experiments/exp17_q1_enriched_weather_sources.yaml"))
    findings.extend(audit_weather_views("configs/experiments/exp18_juxtapose_v2_relation_ablation.yaml"))
    exp19_path = project_root() / "configs" / "experiments" / "exp19_q1_all_models_all_alerts_weather_leaderboard.yaml"
    if exp19_path.exists():
        findings.extend(audit_weather_views(exp19_path))
    findings.extend(audit_report_dependencies())
    for experiment_path in dataset_experiments or []:
        findings.extend(audit_prepared_datasets(experiment_path))
    failed = [finding for finding in findings if finding.status == "fail"]
    return {
        "ok": not failed,
        "failures": len(failed),
        "findings": [finding.to_dict() for finding in findings],
    }
