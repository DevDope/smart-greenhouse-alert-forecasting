from __future__ import annotations

import hashlib
from pathlib import Path

from src.utils.config import resolve_experiment_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "exp19_q1_all_models_all_alerts_weather_leaderboard.yaml"
ARCHIVE = ROOT / "data" / "raw" / "texcoco" / "texcoco.rar"
EXPECTED_SHA256 = "51E224DF48EAF259D285FADEBC07F6CC90958E8B4FD7A5D7442EB1DC81ACD0B1"


def _run_units(config: dict) -> list[tuple[str, str, str]]:
    return [
        (view["name"], task["name"], model_name)
        for view in config["weather_source_views"]
        for task in view["tasks"]
        for model_name in config["models"]
    ]


def test_exp19_resolves_expected_benchmark_shape() -> None:
    resolved = resolve_experiment_config(CONFIG)

    views = {view["name"]: view for view in resolved["weather_source_views"]}
    assert set(views) == {"openmeteo", "nasa", "fused"}
    assert len(resolved["models"]) == 13
    assert all(len(view["tasks"]) == 7 for view in views.values())

    units = _run_units(resolved)
    assert len(units) == 273


def test_smoke_unit_is_available() -> None:
    resolved = resolve_experiment_config(CONFIG)
    units = [
        unit
        for unit in _run_units(resolved)
        if unit == ("openmeteo", "heat_water_stress_60m", "xgboost")
    ]

    assert len(units) == 1
    assert units[0] == ("openmeteo", "heat_water_stress_60m", "xgboost")


def test_dataset_archive_hash_matches_manifest() -> None:
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    assert digest.hexdigest().upper() == EXPECTED_SHA256
