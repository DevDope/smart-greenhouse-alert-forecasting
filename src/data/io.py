"""Data loading, canonical adaptation and persistence utilities."""

from __future__ import annotations

import glob
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.paths import ensure_dir, project_root


def _load_files(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        target = Path(path)
        if target.suffix.lower() == ".csv":
            frames.append(pd.read_csv(target))
        elif target.suffix.lower() == ".parquet":
            frames.append(pd.read_parquet(target))
        else:
            raise ValueError(f"Unsupported file format: {target.suffix}")
    if not frames:
        raise FileNotFoundError("No data files found.")
    return pd.concat(frames, ignore_index=True)


def _generate_synthetic_texcoco(data_config: dict[str, Any]) -> pd.DataFrame:
    source_cfg = data_config["source"]["synthetic_fallback"]
    frequency = source_cfg["frequency"]
    periods = int(source_cfg["periods"])
    rng = np.random.default_rng(int(source_cfg["seed"]))
    timestamps = pd.date_range(source_cfg["start"], periods=periods, freq=frequency)

    step = np.arange(periods)
    hour = (step % 288) / 12.0
    radiation_proxy = np.clip(1_200 * np.sin((hour - 6.0) * np.pi / 12.0) + rng.normal(0, 30, periods), 0, None)
    air_temperature = 20 + 8 * np.sin((hour - 8.0) * np.pi / 12.0) + rng.normal(0, 0.8, periods)
    relative_humidity = 75 - 15 * np.sin((hour - 8.0) * np.pi / 12.0) + rng.normal(0, 2.0, periods)
    light_lux = radiation_proxy * 54.0 + rng.normal(0, 20, periods)
    soil_temperature = 18 + 4 * np.sin((hour - 10.0) * np.pi / 12.0) + rng.normal(0, 0.5, periods)
    vpd = np.clip(0.4 + (air_temperature - 18.0) * 0.12 + rng.normal(0, 0.05, periods), 0, None)

    irrigation_lph = np.zeros(periods)
    schedule = {72, 108, 144, 180, 216}
    for index in range(periods):
        if (index % 288) in schedule and radiation_proxy[index] > 100:
            irrigation_lph[index : index + 4] = 6.0 + rng.uniform(0.0, 1.5)

    soil_moisture = np.zeros(periods)
    soil_moisture[0] = 58.0
    for index in range(1, periods):
        evap = 0.02 * air_temperature[index] + 0.0008 * radiation_proxy[index]
        recovery = 0.6 * irrigation_lph[index]
        soil_moisture[index] = soil_moisture[index - 1] - evap + recovery + rng.normal(0, 0.25)
        soil_moisture[index] = float(np.clip(soil_moisture[index], 20.0, 90.0))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "soil_moisture": soil_moisture,
            "air_temperature": air_temperature,
            "relative_humidity": relative_humidity,
            "radiation_proxy": radiation_proxy,
            "soil_temperature": soil_temperature,
            "vpd": vpd,
            "light_lux": light_lux,
            "ec25": np.nan,
            "ph": np.nan,
            "irrigation_event_observed": (irrigation_lph > 0).astype(float),
            "irrigation_lph": irrigation_lph,
        }
    )


def _coerce_series(series: pd.Series, dtype_name: str) -> pd.Series:
    if dtype_name == "float":
        if series.dtype == object:
            normalized = (
                series.astype(str)
                .str.strip()
                .replace({"": np.nan, "nan": np.nan, "None": np.nan, "True": 1.0, "False": 0.0})
            )
            return pd.to_numeric(normalized, errors="coerce")
        return pd.to_numeric(series, errors="coerce")
    if dtype_name == "string":
        return series.astype("string")
    raise ValueError(f"Unsupported adapter dtype: {dtype_name}")


def _transform_source_series(
    raw_df: pd.DataFrame,
    mapping: dict[str, Any],
    dtype_name: str,
) -> pd.Series | None:
    source_names = mapping.get("sources")
    if not source_names:
        return None
    if not isinstance(source_names, list) or len(source_names) < 2:
        raise ValueError("Adapter mappings with 'sources' require at least two source columns.")
    missing = [source_name for source_name in source_names if source_name not in raw_df.columns]
    if missing:
        return None

    source_frame = pd.concat(
        [_coerce_series(raw_df[source_name], dtype_name) for source_name in source_names],
        axis=1,
    )
    transform = str(mapping.get("transform", "mean")).lower()
    if transform == "mean":
        return source_frame.mean(axis=1)
    if transform == "max":
        return source_frame.max(axis=1)
    if transform == "min":
        return source_frame.min(axis=1)
    if transform == "sum":
        return source_frame.sum(axis=1, min_count=1)
    if transform == "difference":
        return source_frame.iloc[:, 0] - source_frame.iloc[:, 1]
    if transform == "abs_difference":
        return (source_frame.iloc[:, 0] - source_frame.iloc[:, 1]).abs()
    raise ValueError(f"Unsupported adapter transform: {transform}")


def _build_timestamp(raw_df: pd.DataFrame, timestamp_cfg: dict[str, Any]) -> pd.Series:
    if "from_columns" in timestamp_cfg:
        separator = timestamp_cfg.get("separator", " ")
        raw_timestamp = raw_df[timestamp_cfg["from_columns"]].astype(str).agg(separator.join, axis=1)
    elif "source_column" in timestamp_cfg:
        raw_timestamp = raw_df[timestamp_cfg["source_column"]]
    else:
        raise ValueError("Timestamp adapter requires 'source_column' or 'from_columns'.")
    return pd.to_datetime(
        raw_timestamp,
        dayfirst=bool(timestamp_cfg.get("dayfirst", False)),
        errors=timestamp_cfg.get("errors", "raise"),
    )


def _map_columns(
    raw_df: pd.DataFrame,
    mappings: dict[str, dict[str, Any]],
    *,
    required: bool,
    create_optional_columns: bool,
) -> tuple[dict[str, pd.Series], list[str], list[str]]:
    canonical_map: dict[str, pd.Series] = {}
    missing_columns: list[str] = []
    empty_columns: list[str] = []

    for canonical_name, mapping in mappings.items():
        source_name = mapping.get("source", canonical_name)
        dtype_name = mapping.get("dtype", "float")
        transformed = _transform_source_series(raw_df, mapping, dtype_name)
        if transformed is not None:
            canonical_map[canonical_name] = transformed
            if transformed.notna().sum() == 0:
                empty_columns.append(canonical_name)
            continue

        if source_name not in raw_df.columns:
            if required:
                missing_columns.append(canonical_name)
            elif create_optional_columns:
                canonical_map[canonical_name] = pd.Series(np.nan, index=raw_df.index, dtype="float64")
                missing_columns.append(canonical_name)
            continue

        coerced = _coerce_series(raw_df[source_name], dtype_name)
        canonical_map[canonical_name] = coerced
        if coerced.notna().sum() == 0:
            empty_columns.append(canonical_name)

    return canonical_map, missing_columns, empty_columns


def _adapt_raw_to_canonical(raw_df: pd.DataFrame, data_config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    adapter_cfg = data_config["source"].get("adapter", {})
    timestamp_cfg = adapter_cfg.get("timestamp", {"source_column": data_config["timestamp_column"]})
    mappings_cfg = adapter_cfg.get("mappings", {})
    create_optional_columns = bool(adapter_cfg.get("create_optional_columns", True))

    canonical_columns: dict[str, pd.Series] = {data_config["timestamp_column"]: _build_timestamp(raw_df, timestamp_cfg)}
    required_map, missing_required, empty_required = _map_columns(
        raw_df,
        mappings_cfg.get("required", {}),
        required=True,
        create_optional_columns=create_optional_columns,
    )
    optional_map, missing_optional, empty_optional = _map_columns(
        raw_df,
        mappings_cfg.get("optional", {}),
        required=False,
        create_optional_columns=create_optional_columns,
    )
    canonical_columns.update(required_map)
    canonical_columns.update(optional_map)

    if missing_required:
        raise ValueError(
            "Missing required Texcoco columns after mapping: "
            f"{missing_required}. Check configs/data/texcoco.yaml."
        )

    adapted = pd.DataFrame(canonical_columns)
    invalid_timestamps = int(adapted[data_config["timestamp_column"]].isna().sum())
    if invalid_timestamps:
        message = f"Dropping {invalid_timestamps} rows with invalid timestamps during canonical adaptation."
        if bool(adapter_cfg.get("drop_invalid_timestamps", True)):
            warnings.warn(message)
            adapted = adapted.loc[adapted[data_config["timestamp_column"]].notna()].reset_index(drop=True)
        else:
            raise ValueError(message)

    if empty_required:
        raise ValueError(
            f"Required Texcoco columns mapped correctly but contain no usable values: {empty_required}."
        )

    if missing_optional:
        warnings.warn(f"Optional Texcoco columns not found and filled with NaN: {missing_optional}.")
    if empty_optional:
        warnings.warn(
            "Optional Texcoco columns are present but empty after coercion: "
            f"{empty_optional}."
        )

    metadata = {
        "adapter": adapter_cfg,
        "raw_columns": raw_df.columns.tolist(),
        "missing_optional_columns": missing_optional,
        "empty_optional_columns": empty_optional,
        "preserved_raw_columns": {
            column: column for column in adapter_cfg.get("preserve_raw_columns", []) if column in raw_df.columns
        },
    }
    return adapted, metadata


def load_raw_data(data_config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = data_config["source"]
    source_type = source.get("type", "auto")
    patterns = source.get("file_patterns", [])
    matched_paths: list[str] = []
    for pattern in patterns:
        matched_paths.extend(glob.glob(str(project_root() / pattern)))
    matched_paths = sorted(set(matched_paths))

    if matched_paths and source_type in {"auto", "files"}:
        raw = _load_files(matched_paths)
        adapted, metadata = _adapt_raw_to_canonical(raw, data_config)
        metadata["source_type"] = "files"
        metadata["source_files"] = matched_paths
        return adapted, metadata

    fallback_enabled = source.get("synthetic_fallback", {}).get("enabled", False)
    if source_type == "synthetic" or fallback_enabled:
        synthetic = _generate_synthetic_texcoco(data_config)
        return synthetic, {
            "source_type": "synthetic",
            "source_files": [],
            "missing_optional_columns": [],
            "empty_optional_columns": [],
            "raw_columns": synthetic.columns.tolist(),
            "adapter": source.get("adapter", {}),
            "preserved_raw_columns": {},
        }

    raise FileNotFoundError("No input files found and synthetic fallback is disabled.")


def write_dataframe(df: pd.DataFrame, path_without_suffix: str | Path) -> Path:
    target_base = Path(path_without_suffix)
    ensure_dir(target_base.parent)
    parquet_path = target_base.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = target_base.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def write_dataframe_bundle(df: pd.DataFrame, directory: str | Path, stem: str) -> dict[str, Path]:
    target_dir = ensure_dir(directory)
    parquet_path = target_dir / f"{stem}.parquet"
    csv_path = target_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
        primary_path = parquet_path
    except Exception:
        primary_path = csv_path
    return {"parquet": primary_path, "csv": csv_path}


def read_dataframe(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(target)
        except Exception:
            csv_fallback = target.with_suffix(".csv")
            if csv_fallback.exists():
                return read_dataframe(csv_fallback)
            raise
    if target.suffix.lower() == ".csv":
        frame = pd.read_csv(target)
        for column in ["timestamp", "observation_start", "observation_end", "label_end"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame
    raise ValueError(f"Unsupported tabular file format: {target.suffix}")
