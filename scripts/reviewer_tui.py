from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/exp19_q1_all_models_all_alerts_weather_leaderboard.yaml"
ARCHIVE = ROOT / "data" / "raw" / "texcoco" / "texcoco.rar"
EXTRACTED = ROOT / "data" / "raw" / "texcoco" / "proto_enriched_outside_weather.csv"
EXPECTED_SHA256 = "51E224DF48EAF259D285FADEBC07F6CC90958E8B4FD7A5D7442EB1DC81ACD0B1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_dataset() -> bool:
    print("\nDataset check")
    print("-------------")
    if not ARCHIVE.exists():
        print(f"Missing archive: {ARCHIVE.relative_to(ROOT)}")
        return False

    archive_hash = _sha256(ARCHIVE)
    print(f"Archive: {ARCHIVE.relative_to(ROOT)}")
    print(f"SHA256 : {archive_hash}")
    if archive_hash != EXPECTED_SHA256:
        print("Hash mismatch. Please confirm the archive was not modified.")
        return False

    if not EXTRACTED.exists():
        print(f"Missing extracted CSV: {EXTRACTED.relative_to(ROOT)}")
        print("Extract texcoco.rar in data/raw/texcoco/ before running experiments.")
        return False

    print(f"Extracted CSV found: {EXTRACTED.relative_to(ROOT)}")
    return True


def run_command(args: list[str]) -> int:
    print("\nRunning:")
    print(" ".join(args))
    print()
    return subprocess.call(args, cwd=ROOT)


def run_smoke() -> int:
    if not verify_dataset():
        return 1
    return run_command(
        [
            sys.executable,
            "-m",
            "src.cli",
            "run-resumable",
            "--config",
            CONFIG,
            "--run-id",
            "reviewer_smoke",
            "--view",
            "openmeteo",
            "--task",
            "heat_water_stress_60m",
            "--model",
            "xgboost",
            "--fail-fast",
        ]
    )


def run_full() -> int:
    if not verify_dataset():
        return 1
    return run_command(
        [
            sys.executable,
            "-m",
            "src.cli",
            "run-resumable",
            "--config",
            CONFIG,
            "--run-id",
            "reviewer_exp19",
        ]
    )


def generate_report() -> int:
    return run_command(
        [
            sys.executable,
            "-m",
            "src.cli",
            "report",
            "--config",
            CONFIG,
            "--run-id",
            "reviewer_exp19",
        ]
    )


def clean_reviewer_outputs() -> None:
    targets = [
        ROOT / "results" / "metrics" / "reviewer_smoke",
        ROOT / "results" / "metrics" / "reviewer_smoke__openmeteo",
        ROOT / "results" / "metrics" / "reviewer_exp19",
        ROOT / "results" / "metrics" / "reviewer_exp19__openmeteo",
        ROOT / "results" / "metrics" / "reviewer_exp19__nasa",
        ROOT / "results" / "metrics" / "reviewer_exp19__fused",
        ROOT / "results" / "tables" / "reviewer_exp19",
        ROOT / "results" / "figures" / "reviewer_exp19",
    ]
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed {target.relative_to(ROOT)}")


def main() -> int:
    menu = """
Smart Greenhouse Alert Forecasting
----------------------------------
1. Verify dataset
2. Run smoke test
3. Run full Exp19 benchmark
4. Generate Exp19 report tables
5. Clean reviewer outputs
0. Exit
"""
    actions = {
        "1": lambda: 0 if verify_dataset() else 1,
        "2": run_smoke,
        "3": run_full,
        "4": generate_report,
        "5": lambda: clean_reviewer_outputs() or 0,
    }
    while True:
        print(menu)
        choice = input("Select an option: ").strip()
        if choice == "0":
            return 0
        action = actions.get(choice)
        if action is None:
            print("Unknown option.")
            continue
        code = action()
        print(f"\nDone with exit code {code}.")


if __name__ == "__main__":
    raise SystemExit(main())
