"""Reusable sequence adapters and PyTorch training utilities."""

from __future__ import annotations

import copy
import re
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..base import BaseModel
from ...data.io import read_dataframe

LAG_PATTERN = re.compile(r"^(?P<base>.+)__lag_(?P<lag>\d+)$")


@dataclass
class SequenceArrayBundle:
    sequence: np.ndarray
    sequence_mask: np.ndarray
    static: np.ndarray
    static_mask: np.ndarray


class SequenceTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Simple tensor-backed dataset."""

    def __init__(self, sequence: np.ndarray, static: np.ndarray, target: np.ndarray) -> None:
        self.sequence = torch.from_numpy(sequence.astype(np.float32))
        self.static = torch.from_numpy(static.astype(np.float32))
        self.target = torch.from_numpy(target.astype(np.int64))

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequence[index], self.static[index], self.target[index]


class SequenceFeatureAdapter:
    """Reconstructs sequence tensors from the prepared tabular artifact."""

    def __init__(
        self,
        *,
        representation: str = "lag_tabular",
        context_steps: int | None = None,
        static_mode: str = "engineered",
    ) -> None:
        self.representation = representation
        self.context_steps = context_steps
        self.static_mode = static_mode
        self.columns_: list[str] = []
        self.sequence_features_: list[str] = []
        self.static_features_: list[str] = []
        self.lag_steps_: list[int] = []
        self._lag_lookup: dict[str, dict[int, str]] = {}
        self._sequence_means: np.ndarray | None = None
        self._sequence_stds: np.ndarray | None = None
        self._static_means: np.ndarray | None = None
        self._static_stds: np.ndarray | None = None
        self._canonical_frame: pd.DataFrame | None = None
        self._timestamp_to_index: dict[pd.Timestamp, int] = {}
        self._timing_columns: list[str] = []

    def fit(self, X: pd.DataFrame) -> "SequenceFeatureAdapter":
        frame = X.copy()
        frame.attrs = dict(X.attrs)
        self.columns_ = frame.columns.tolist()
        if self.representation == "sequence_first":
            self._fit_sequence_first(frame)
        else:
            self._fit_lag_tabular(frame)

        bundle = self._build_bundle(frame)
        self._sequence_means = np.nanmean(bundle.sequence, axis=(0, 1))
        self._sequence_stds = np.nanstd(bundle.sequence, axis=(0, 1))
        self._sequence_means = np.where(np.isfinite(self._sequence_means), self._sequence_means, 0.0)
        self._sequence_stds = np.where(
            np.isfinite(self._sequence_stds) & (self._sequence_stds > 1e-6),
            self._sequence_stds,
            1.0,
        )

        if bundle.static.shape[1]:
            self._static_means = np.nanmean(bundle.static, axis=0)
            self._static_stds = np.nanstd(bundle.static, axis=0)
            self._static_means = np.where(np.isfinite(self._static_means), self._static_means, 0.0)
            self._static_stds = np.where(
                np.isfinite(self._static_stds) & (self._static_stds > 1e-6),
                self._static_stds,
                1.0,
            )
        else:
            self._static_means = np.zeros(0, dtype=np.float32)
            self._static_stds = np.ones(0, dtype=np.float32)
        return self

    def fit_transform(self, X: pd.DataFrame) -> SequenceArrayBundle:
        return self.fit(X).transform(X)

    def transform(self, X: pd.DataFrame) -> SequenceArrayBundle:
        if not self.columns_:
            raise RuntimeError("SequenceFeatureAdapter must be fitted before transform().")

        frame = X.reindex(columns=self.columns_, fill_value=np.nan).copy()
        frame.attrs = dict(X.attrs)
        bundle = self._build_bundle(frame)

        sequence = (bundle.sequence - self._sequence_means.reshape(1, 1, -1)) / self._sequence_stds.reshape(1, 1, -1)
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if bundle.static.shape[1]:
            static = (bundle.static - self._static_means.reshape(1, -1)) / self._static_stds.reshape(1, -1)
            static = np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            static = bundle.static.astype(np.float32)

        return SequenceArrayBundle(
            sequence=sequence,
            sequence_mask=bundle.sequence_mask.astype(np.float32),
            static=static,
            static_mask=bundle.static_mask.astype(np.float32),
        )

    def _fit_lag_tabular(self, frame: pd.DataFrame) -> None:
        lag_lookup: dict[str, dict[int, str]] = {}
        lag_steps: set[int] = set()
        base_candidates: list[str] = []
        static_features: list[str] = []

        for column in self.columns_:
            match = LAG_PATTERN.match(column)
            if match:
                base_name = match.group("base")
                lag = int(match.group("lag"))
                lag_lookup.setdefault(base_name, {})[lag] = column
                lag_steps.add(lag)
                continue
            if "__" not in column:
                base_candidates.append(column)
            else:
                static_features.append(column)

        if lag_lookup:
            sequence_features = [column for column in base_candidates if column in lag_lookup]
        else:
            sequence_features = list(base_candidates)
        static_features.extend(column for column in base_candidates if column not in set(sequence_features))

        if not sequence_features:
            raise ValueError(
                "Cannot build sequence tensors because no raw or lagged sequence features were found "
                "in the prepared dataset."
            )

        self.sequence_features_ = sequence_features
        self.static_features_ = static_features
        self.lag_steps_ = [0, *sorted(lag_steps)]
        self._lag_lookup = lag_lookup

    def _fit_sequence_first(self, frame: pd.DataFrame) -> None:
        attrs = self._extract_attrs(frame)
        prepared_metadata = attrs["prepared_metadata"]
        feature_metadata = prepared_metadata["feature_metadata"]
        available_features = list(feature_metadata.get("available_numeric_features", []))
        if not available_features:
            raise ValueError("Sequence-first representation requires available numeric features in prepared metadata.")
        self.sequence_features_ = available_features
        self.lag_steps_ = list(range(self.context_steps or feature_metadata.get("window_size_steps", 1)))
        if self.static_mode == "calendar":
            self.static_features_ = [column for column in ("hour_sin", "hour_cos", "dow_sin", "dow_cos") if column in frame.columns]
        elif self.static_mode == "none":
            self.static_features_ = []
        else:
            self.static_features_ = [column for column in frame.columns if column.startswith("hour_") or column.startswith("dow_")]

        canonical_path = Path(prepared_metadata["canonical_dataset"]["path"])
        canonical_frame = read_dataframe(canonical_path)
        timestamp_column = prepared_metadata["data"]["timestamp_column"]
        canonical_frame[timestamp_column] = pd.to_datetime(canonical_frame[timestamp_column])
        self._canonical_frame = canonical_frame
        self._timestamp_to_index = {
            pd.Timestamp(timestamp): index
            for index, timestamp in enumerate(canonical_frame[timestamp_column])
        }
        self._timing_columns = attrs["timing_frame"].columns.tolist()

    def _extract_attrs(self, frame: pd.DataFrame) -> dict[str, Any]:
        timing_frame = frame.attrs.get("timing_frame")
        prepared_metadata = frame.attrs.get("prepared_metadata")
        if timing_frame is None or prepared_metadata is None:
            raise ValueError(
                "Deep sequence adapters require timing_frame and prepared_metadata attached to the feature frame."
            )
        return {"timing_frame": timing_frame.copy(), "prepared_metadata": prepared_metadata}

    def describe(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "static_mode": self.static_mode,
            "sequence_length": len(self.lag_steps_),
            "lag_steps": list(self.lag_steps_),
            "sequence_features": list(self.sequence_features_),
            "static_features": list(self.static_features_),
            "sequence_feature_count": len(self.sequence_features_),
            "static_feature_count": len(self.static_features_),
        }

    def _build_bundle(self, frame: pd.DataFrame) -> SequenceArrayBundle:
        if self.representation == "sequence_first":
            return self._build_sequence_first_bundle(frame)

        row_count = len(frame)
        seq_len = len(self.lag_steps_)
        seq_width = len(self.sequence_features_)
        sequence = np.full((row_count, seq_len, seq_width), np.nan, dtype=np.float32)

        for feature_index, feature_name in enumerate(self.sequence_features_):
            sequence[:, 0, feature_index] = pd.to_numeric(
                frame[feature_name],
                errors="coerce",
            ).to_numpy(dtype=np.float32)
            for lag_index, lag in enumerate(self.lag_steps_[1:], start=1):
                lag_column = self._lag_lookup.get(feature_name, {}).get(lag)
                if lag_column is None:
                    continue
                sequence[:, lag_index, feature_index] = pd.to_numeric(
                    frame[lag_column],
                    errors="coerce",
                ).to_numpy(dtype=np.float32)

        sequence_mask = np.isfinite(sequence).astype(np.float32)
        if self.static_features_:
            static = (
                frame[self.static_features_]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float32)
            )
            static_mask = np.isfinite(static).astype(np.float32)
        else:
            static = np.zeros((row_count, 0), dtype=np.float32)
            static_mask = np.zeros((row_count, 0), dtype=np.float32)

        return SequenceArrayBundle(
            sequence=sequence,
            sequence_mask=sequence_mask,
            static=static,
            static_mask=static_mask,
        )

    def _build_sequence_first_bundle(self, frame: pd.DataFrame) -> SequenceArrayBundle:
        if self._canonical_frame is None:
            raise RuntimeError("Sequence-first adapter requires a loaded canonical frame.")
        attrs = self._extract_attrs(frame)
        timing_frame = attrs["timing_frame"].reset_index(drop=True)
        prepared_metadata = attrs["prepared_metadata"]
        timestamp_column = prepared_metadata["data"]["timestamp_column"]
        timestamps = pd.to_datetime(timing_frame.get("observation_end", timing_frame["timestamp"]))

        row_count = len(frame)
        seq_len = len(self.lag_steps_)
        seq_width = len(self.sequence_features_)
        sequence = np.full((row_count, seq_len, seq_width), np.nan, dtype=np.float32)

        canonical_numeric = (
            self._canonical_frame[self.sequence_features_]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        for row_index, timestamp in enumerate(timestamps):
            canonical_index = self._timestamp_to_index.get(pd.Timestamp(timestamp))
            if canonical_index is None:
                continue
            start_index = canonical_index - seq_len + 1
            source_start = max(start_index, 0)
            source_end = canonical_index + 1
            source_values = canonical_numeric[source_start:source_end]
            if source_values.size == 0:
                continue
            target_start = seq_len - source_values.shape[0]
            sequence[row_index, target_start:, :] = source_values

        sequence_mask = np.isfinite(sequence).astype(np.float32)
        if self.static_features_:
            static = (
                frame[self.static_features_]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float32)
            )
            static_mask = np.isfinite(static).astype(np.float32)
        else:
            static = np.zeros((row_count, 0), dtype=np.float32)
            static_mask = np.zeros((row_count, 0), dtype=np.float32)

        return SequenceArrayBundle(
            sequence=sequence,
            sequence_mask=sequence_mask,
            static=static,
            static_mask=static_mask,
        )


class TorchSequenceClassifier(BaseModel):
    """Common training loop for deep sequence classifiers."""

    model_name = "torch_sequence"

    def __init__(
        self,
        *,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        max_epochs: int = 8,
        patience: int = 3,
        val_fraction: float = 0.1,
        weight_decay: float = 1e-4,
        class_weight: str | None = "balanced",
        random_state: int = 42,
        device: str = "cpu",
        use_missingness: bool = True,
        grad_clip: float = 1.0,
        representation: str = "lag_tabular",
        context_steps: int | None = None,
        static_mode: str = "engineered",
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        self.random_state = random_state
        self.device = device
        self.use_missingness = use_missingness
        self.grad_clip = grad_clip
        self.representation = representation
        self.context_steps = context_steps
        self.static_mode = static_mode

        self.adapter_: SequenceFeatureAdapter | None = None
        self.network_: nn.Module | None = None
        self.classes_: list[int] = []
        self.constant_probabilities_: np.ndarray | None = None
        self.training_metadata: dict[str, Any] = {}

    @abstractmethod
    def build_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
        num_classes: int,
    ) -> nn.Module:
        raise NotImplementedError

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
    ) -> "TorchSequenceClassifier":
        target = pd.Series(y).astype(int)
        if target.empty:
            raise ValueError("Cannot fit a deep model on an empty training split.")

        unique_classes = sorted(target.unique().tolist())
        self.classes_ = list(range(max(unique_classes) + 1))
        if len(unique_classes) == 1:
            self.constant_probabilities_ = np.zeros(len(self.classes_), dtype=np.float32)
            self.constant_probabilities_[unique_classes[0]] = 1.0
            self.training_metadata = {
                "architecture": self.model_name,
                "epochs_trained": 0,
                "best_epoch": 0,
                "best_monitor_loss": None,
                "training_seconds": 0.0,
                "note": "single-class training split, using deterministic probabilities",
            }
            return self

        self.adapter_ = SequenceFeatureAdapter(
            representation=self.representation,
            context_steps=self.context_steps,
            static_mode=self.static_mode,
        )
        bundle = self.adapter_.fit_transform(X)
        sequence_input = self._compose_sequence_input(bundle)
        static_input = self._compose_static_input(bundle)

        train_indices, val_indices = self._validation_indices(len(target))
        train_dataset = SequenceTensorDataset(
            sequence_input[train_indices],
            static_input[train_indices],
            target.iloc[train_indices].to_numpy(dtype=np.int64),
        )
        val_dataset = (
            SequenceTensorDataset(
                sequence_input[val_indices],
                static_input[val_indices],
                target.iloc[val_indices].to_numpy(dtype=np.int64),
            )
            if val_indices.size
            else None
        )
        loader_generator = torch.Generator().manual_seed(self.random_state)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False) if val_dataset else None

        device = self._resolve_device()
        self.network_ = self.build_network(
            sequence_dim=int(sequence_input.shape[2]),
            static_dim=int(static_input.shape[1]),
            sequence_length=int(sequence_input.shape[1]),
            num_classes=len(self.classes_),
        ).to(device)

        weights = self._build_class_weights(target.iloc[train_indices], len(self.classes_))
        criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_state: dict[str, torch.Tensor] | None = None
        best_monitor = float("inf")
        best_epoch = 0
        stale_epochs = 0
        epochs_ran = 0
        start_time = time.perf_counter()

        for epoch in range(1, self.max_epochs + 1):
            epochs_ran = epoch
            train_loss = self._run_epoch(train_loader, criterion, optimizer, device)
            monitor_loss = self._evaluate_loss(val_loader, criterion, device) if val_loader else train_loss
            if monitor_loss + 1e-4 < best_monitor:
                best_monitor = monitor_loss
                best_epoch = epoch
                stale_epochs = 0
                best_state = copy.deepcopy(self.network_.state_dict())
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is not None:
            self.network_.load_state_dict(best_state)

        self.network_.to("cpu")
        adapter_description = self.adapter_.describe()
        parameter_count = int(sum(parameter.numel() for parameter in self.network_.parameters()))
        self.training_metadata = {
            "architecture": self.model_name,
            "epochs_trained": epochs_ran,
            "best_epoch": best_epoch,
            "best_monitor_loss": float(best_monitor) if np.isfinite(best_monitor) else None,
            "training_seconds": round(time.perf_counter() - start_time, 4),
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device_requested": self.device,
            "device_used": device,
            "sequence_input_shape": [
                int(sequence_input.shape[0]),
                int(sequence_input.shape[1]),
                int(sequence_input.shape[2]),
            ],
            "static_input_shape": [
                int(static_input.shape[0]),
                int(static_input.shape[1]),
            ],
            "parameter_count": parameter_count,
            **adapter_description,
        }
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(X)
        predictions = np.asarray(self.classes_)[np.argmax(probabilities.to_numpy(), axis=1)]
        return pd.Series(predictions, index=X.index, name="prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.constant_probabilities_ is not None:
            repeated = np.tile(self.constant_probabilities_, (len(X), 1))
            return pd.DataFrame(repeated, index=X.index, columns=self._probability_columns())
        if self.adapter_ is None or self.network_ is None:
            raise RuntimeError("Deep model must be fitted before prediction.")

        bundle = self.adapter_.transform(X)
        dataset = SequenceTensorDataset(
            self._compose_sequence_input(bundle),
            self._compose_static_input(bundle),
            np.zeros(len(X), dtype=np.int64),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.network_.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for sequence_batch, static_batch, _ in loader:
                logits = self.network_(sequence_batch, static_batch)
                outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probabilities = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, len(self.classes_)))
        return pd.DataFrame(probabilities, index=X.index, columns=self._probability_columns())

    def _compose_sequence_input(self, bundle: SequenceArrayBundle) -> np.ndarray:
        if not self.use_missingness:
            return bundle.sequence.astype(np.float32)
        return np.concatenate([bundle.sequence, bundle.sequence_mask], axis=2).astype(np.float32)

    def _compose_static_input(self, bundle: SequenceArrayBundle) -> np.ndarray:
        if bundle.static.shape[1] == 0:
            return bundle.static.astype(np.float32)
        if not self.use_missingness:
            return bundle.static.astype(np.float32)
        return np.concatenate([bundle.static, bundle.static_mask], axis=1).astype(np.float32)

    def _build_class_weights(self, target: pd.Series, num_classes: int) -> torch.Tensor | None:
        if self.class_weight != "balanced":
            return None
        counts = np.bincount(target.to_numpy(dtype=np.int64), minlength=num_classes).astype(np.float32)
        counts = np.where(counts > 0, counts, 1.0)
        weights = counts.sum() / (num_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)

    def _run_epoch(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
    ) -> float:
        if self.network_ is None:
            raise RuntimeError("Network is not initialized.")
        self.network_.train()
        running_loss = 0.0
        total = 0
        for sequence_batch, static_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = self.network_(sequence_batch.to(device), static_batch.to(device))
            loss = criterion(logits, target_batch.to(device))
            loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.network_.parameters(), self.grad_clip)
            optimizer.step()
            batch_size = int(target_batch.shape[0])
            running_loss += float(loss.item()) * batch_size
            total += batch_size
        return running_loss / max(total, 1)

    def _evaluate_loss(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        device: str,
    ) -> float:
        if self.network_ is None:
            raise RuntimeError("Network is not initialized.")
        self.network_.eval()
        running_loss = 0.0
        total = 0
        with torch.no_grad():
            for sequence_batch, static_batch, target_batch in loader:
                logits = self.network_(sequence_batch.to(device), static_batch.to(device))
                loss = criterion(logits, target_batch.to(device))
                batch_size = int(target_batch.shape[0])
                running_loss += float(loss.item()) * batch_size
                total += batch_size
        return running_loss / max(total, 1)

    def _probability_columns(self) -> list[str]:
        return [f"class_{int(class_value)}" for class_value in self.classes_]

    def _validation_indices(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        if length < 32 or self.val_fraction <= 0:
            return np.arange(length), np.array([], dtype=int)
        val_size = max(1, int(round(length * self.val_fraction)))
        val_size = min(val_size, max(1, length // 4))
        train_size = max(length - val_size, 1)
        return np.arange(train_size), np.arange(train_size, length)

    def _resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device
