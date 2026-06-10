"""Dataset orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.reproducibility import save_json, stable_hash
from .clean import clean_data
from .features import build_features
from .io import load_raw_data, read_dataframe, write_dataframe_bundle
from .labels import generate_labels
from .resample import resample_data
from .schema import validate_schema
from .splits import apply_chronological_split, compute_embargo_steps
from .windows import build_supervised_frame


@dataclass
class CanonicalDataset:
    site: str
    canonical_id: str
    data_path: Path
    csv_path: Path
    frame: pd.DataFrame
    metadata: dict[str, Any]


@dataclass
class PreparedDataset:
    task_name: str
    label_column: str
    dataset_id: str
    data_path: Path
    frame: pd.DataFrame
    feature_columns: list[str]
    class_names: list[str]
    metadata: dict[str, Any]


def _time_summary(frame: pd.DataFrame, timestamp_column: str) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame[timestamp_column]).sort_values()
    diffs = timestamps.diff().dropna()
    top_diff = diffs.mode().iloc[0] if not diffs.empty else None
    return {
        "row_count": int(len(frame)),
        "start": timestamps.min().isoformat() if not timestamps.empty else None,
        "end": timestamps.max().isoformat() if not timestamps.empty else None,
        "inferred_step": str(top_diff) if top_diff is not None else None,
    }


def prepare_canonical_dataset(resolved_config: dict[str, Any]) -> CanonicalDataset:
    data_config = resolved_config["data"]
    raw, load_metadata = load_raw_data(data_config)
    validated = validate_schema(raw, data_config)
    cleaned = clean_data(validated, data_config)
    resampled = resample_data(cleaned, data_config)
    optional_columns = data_config["schema"].get("optional_columns", [])
    available_columns = [column for column in resampled.columns if column != data_config["timestamp_column"]]
    missing_optional_columns = [
        column
        for column in optional_columns
        if column not in resampled.columns or resampled[column].notna().sum() == 0
    ]
    canonical_metadata = {
        "site": data_config["site"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timestamp_column": data_config["timestamp_column"],
        "source": load_metadata,
        "data_config": data_config,
        "columns_final": available_columns,
        "missing_optional_columns": missing_optional_columns,
        "time_summary": _time_summary(resampled, data_config["timestamp_column"]),
        "null_fraction": {
            column: float(resampled[column].isna().mean())
            for column in resampled.columns
            if column != data_config["timestamp_column"]
        },
    }
    canonical_signature = {
        "site": canonical_metadata["site"],
        "timestamp_column": canonical_metadata["timestamp_column"],
        "source": canonical_metadata["source"],
        "data_config": canonical_metadata["data_config"],
        "columns_final": canonical_metadata["columns_final"],
        "missing_optional_columns": canonical_metadata["missing_optional_columns"],
        "time_summary": canonical_metadata["time_summary"],
        "null_fraction": canonical_metadata["null_fraction"],
    }
    canonical_id = f"{data_config['site']}__canonical__{stable_hash(canonical_signature)}"
    processed_root = Path(resolved_config["artifacts"]["processed_dir"]) / data_config["site"] / "canonical" / canonical_id
    canonical_parquet = processed_root / "canonical.parquet"
    canonical_csv = processed_root / "canonical.csv"
    if canonical_parquet.exists():
        resampled = read_dataframe(canonical_parquet)
        canonical_artifacts = {"parquet": canonical_parquet, "csv": canonical_csv}
    else:
        canonical_artifacts = write_dataframe_bundle(resampled, processed_root, "canonical")
        save_json(canonical_metadata, processed_root / "metadata.json")

    return CanonicalDataset(
        site=data_config["site"],
        canonical_id=canonical_id,
        data_path=canonical_artifacts["parquet"],
        csv_path=canonical_artifacts["csv"],
        frame=resampled,
        metadata=canonical_metadata,
    )


def prepare_task_dataset(
    resolved_config: dict[str, Any],
    task_config: dict[str, Any],
    canonical_dataset: CanonicalDataset,
) -> PreparedDataset:
    data_config = resolved_config["data"]
    resampled = canonical_dataset.frame
    features, feature_columns, feature_metadata = build_features(resampled, data_config)
    labels, label_metadata = generate_labels(resampled, task_config, data_config)
    supervised = build_supervised_frame(
        feature_frame=features,
        labels=labels,
        feature_columns=feature_columns,
        task_config=task_config,
        data_config=data_config,
        feature_metadata=feature_metadata,
    )
    embargo_steps = compute_embargo_steps(task_config, feature_metadata, data_config)
    split_frame, split_metadata = apply_chronological_split(
        supervised=supervised,
        split_config=resolved_config["split"],
        embargo_steps=embargo_steps,
    )

    metadata = {
        "task": task_config,
        "data": data_config,
        "canonical_dataset": {
            "canonical_id": canonical_dataset.canonical_id,
            "path": str(canonical_dataset.data_path),
        },
        "split": split_metadata,
        "feature_metadata": feature_metadata,
        "label_metadata": label_metadata,
        "feature_columns": feature_columns,
    }
    dataset_id = f"{task_config['name']}__{stable_hash(metadata)}"
    processed_dir = Path(resolved_config["artifacts"]["processed_dir"]) / data_config["site"] / "tasks" / dataset_id
    dataset_parquet = processed_dir / "dataset.parquet"
    dataset_csv = processed_dir / "dataset.csv"
    if dataset_parquet.exists():
        split_frame = read_dataframe(dataset_parquet)
        dataset_artifacts = {"parquet": dataset_parquet, "csv": dataset_csv}
    else:
        dataset_artifacts = write_dataframe_bundle(split_frame, processed_dir, "dataset")
        save_json(metadata, processed_dir / "metadata.json")

    return PreparedDataset(
        task_name=task_config["name"],
        label_column=task_config["label_column"],
        dataset_id=dataset_id,
        data_path=dataset_artifacts["parquet"],
        frame=split_frame,
        feature_columns=feature_columns,
        class_names=label_metadata["class_names"],
        metadata=metadata,
    )
