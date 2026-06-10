# Smart Greenhouse Alert Forecasting

Reproducible code and data package for the Texcoco greenhouse alert-forecasting benchmark.

This repository contains the reviewer-facing implementation used to reproduce the multi-source weather-aware alert benchmark described in the manuscript **"Multi-Source Weather-Aware Forecasting for Greenhouse Alerts"**. The benchmark evaluates operational proxy alerts for greenhouse decision support using internal greenhouse sensors, Open-Meteo weather variables, NASA POWER weather variables, and a fused weather-source view.

The default reproduction path is:

1. run a short smoke test to verify the environment and dataset;
2. run the full Exp19 benchmark when full reproduction is needed;
3. generate the report tables from the resulting metrics.

## What Is Included

```text
configs/                 Experiment, data, task, and model YAML files
data/raw/texcoco/         Compressed Texcoco dataset and data dictionaries
scripts/reviewer_tui.py   Small command-line menu for reviewers
src/                      Benchmark pipeline, models, evaluation, and reporting
tests/                    Minimal checks for configuration and dataset readiness
results/                  Generated metrics, reports, and figures (gitignored)
```

The full dataset is stored as:

```text
data/raw/texcoco/texcoco.rar
```

Extract it before running experiments. The expected extracted CSV is:

```text
data/raw/texcoco/proto_enriched_outside_weather.csv
```

The archive hash is recorded in `data/raw/texcoco/MANIFEST.json`.

## Dataset

The dataset contains greenhouse sensor readings from Texcoco, State of Mexico, Mexico, combined with external weather-source variables and astronomical context used by the benchmark. It is distributed as a compressed archive so reviewers can reproduce the exact input used by the pipeline.

The main metric dictionary is provided in:

```text
data/raw/texcoco/weather_metric_dictionary.xlsx
data/raw/texcoco/DATA_DICTIONARY.md
```

Dataset summary:

| Property | Value |
| --- | --- |
| Main CSV after extraction | `proto_enriched_outside_weather.csv` |
| Rows | `14,316` |
| Columns | `111` |
| Site | Greenhouse in Texcoco, State of Mexico, Mexico |
| Crop context | Tomato greenhouse crop canopy |
| Coordinates for external context | latitude `19.496304`, longitude `-98.865113` |
| Time zone | `America/Mexico_City` |
| Documented period | `2025-09-11 06:37:14` to `2025-10-30 23:59:59` |
| Internal sensor frequency | approximately every `5 min`; not resampled to hourly |
| External weather frequency | hourly |
| Weather merge rule | local hourly block assignment; no interpolation to 5 min |

The 111 columns are grouped as follows:

| Source | Columns | What they contain |
| --- | ---: | --- |
| Internal greenhouse sensors | 30 | Timestamp fields, air temperature, relative humidity, dew point, VPD, absolute humidity, vertical air measurements, light, PPFD, UV, soil, solution, and irrigation-related fields |
| Open-Meteo | 36 | Hourly outside temperature, humidity, dew point, precipitation, pressure, radiation, wind, evapotranspiration, VPD, daylight flag, lags, rolling sums, daily summaries, and inside-outside contrasts |
| NASA POWER | 37 | Hourly outside temperature, humidity, dew point, precipitation, pressure, wind, radiation/PAR, VPD, daylight flag, lags, rolling sums, daily summaries, and inside-outside contrasts |
| Calculated astronomy | 8 | Solar elevation, solar azimuth, day length, minutes since sunrise, minutes until sunset, moon phase, moon illumination, and moon age |

The internal greenhouse measurements keep their original approximate 5 min cadence. External variables with the `outside_` prefix are hourly or derived from hourly weather records. They describe outside conditions, not direct internal greenhouse measurements. Several internal observations can therefore share the same outside-weather value within one local hour.

The fused weather view is not a fourth data source. It is a benchmark view constructed from selected Open-Meteo and NASA POWER columns, including means, maxima, absolute gaps, and comparable rolling summaries.

Important interpretation notes:

- `lux` is visible illuminance and is not equivalent to solar radiation or PPFD.
- `ppfd` is the photosynthetic photon flux density proxy used as the most crop-relevant internal light variable.
- VPD-related variables describe atmospheric evaporative demand.
- Dew point variables help interpret condensation risk.
- Precipitation and wind variables are external weather context; they do not measure rainfall entering the greenhouse, crop wetting, structural damage, disease, or verified management action.
- CO2 and ventilation variables are not used by this benchmark and are not candidate features for Exp19.

## Installation

Python 3.11 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

If you prefer installing from the pinned requirements file:

```powershell
pip install -r requirements.txt
pip install -e .
```

Notes:

- The full benchmark includes tree-based and neural models, so `lightgbm`, `xgboost`, and `torch` are required for complete reproduction.
- The dataset is provided as a `.rar` archive. Extract it with 7-Zip, WinRAR, or another RAR-compatible tool.
- Generated artifacts are written under `data/processed/` and `results/`.

## Quick Reviewer Path

Open the small reviewer menu:

```powershell
python scripts\reviewer_tui.py
```

The menu can:

- verify the dataset archive and extracted CSV;
- run a one-model smoke test;
- run the full Exp19 benchmark;
- generate report tables;
- clean generated reviewer outputs.

## Direct Commands

Smoke test:

```powershell
greenhouse run-resumable --config configs/experiments/exp19_q1_all_models_all_alerts_weather_leaderboard.yaml --run-id reviewer_smoke --view openmeteo --task heat_water_stress_60m --model xgboost --fail-fast
```

Full Exp19 benchmark:

```powershell
greenhouse run-resumable --config configs/experiments/exp19_q1_all_models_all_alerts_weather_leaderboard.yaml --run-id reviewer_exp19
```

Generate report tables:

```powershell
greenhouse report --config configs/experiments/exp19_q1_all_models_all_alerts_weather_leaderboard.yaml --run-id reviewer_exp19
```

Run minimal tests:

```powershell
python -m pytest -q
```

## Benchmark Scope

The full benchmark uses:

- 13 model configurations;
- 3 weather-source views: `openmeteo`, `nasa`, and `fused`;
- 7 operational alert families;
- 273 expected source-alert-model runs.

The chronological split is 60% training, 20% validation, and 20% testing. PR-AUC is the primary ranking metric. F1, Brier score, expected calibration error, training time, inference time, and artifact size are complementary metrics.

The alert labels are operational decision-support proxies. They should not be interpreted as direct biological damage, yield loss, verified grower action, or crop outcome measurements.

## Model Naming Note

The report layer maps compatibility model IDs to the manuscript-facing names used in the paper. In particular, wrapper-style historical compatibility IDs are reported as `discipline_wrapper` and `juxtapose_wrapper`, while native structured variants are reported as `discipline_native` and `juxtapose_native_v*` where applicable.

## Expected Outputs

After a run, the main files are written under:

```text
results/metrics/<run_id>/
results/tables/<run_id>/
results/figures/<run_id>/
```

For resumable runs, progress is tracked at:

```text
results/metrics/<run_id>/progress_manifest.json
```

The smoke test should complete one source-alert-model unit. The full benchmark should report 273 total units in the progress manifest.

## Reproducibility Notes

- The experiment seed is fixed at `42`.
- The pipeline uses chronological splitting, not random train-test mixing.
- Lagged and rolling features are backward-looking and prediction-time available.
- The exact dataset archive is included and hashed.
- Results may vary slightly across hardware and library builds, especially for neural models and GPU-enabled environments.

## Citation

If this repository is used for review or later reproduction, cite the accompanying manuscript:

> Multi-Source Weather-Aware Forecasting for Greenhouse Alerts.

Repository and dataset links will be added after journal-facing links are finalized.
