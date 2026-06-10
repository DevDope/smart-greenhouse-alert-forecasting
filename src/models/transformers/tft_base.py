"""Reusable TFT family base model and classifier."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ..deep.common import SequenceArrayBundle, SequenceFeatureAdapter, TorchSequenceClassifier
from .modules import (
    CausalInterpretableAttention,
    ContinuousVariableEmbedding,
    GRN,
    LocalLSTMEncoder,
    MultiTaskHeads,
    MultiTaskLoss,
    ReadoutBlock,
    StaticContextEncoder,
    VariableSelectionNetwork,
)

PROJECT_TASK_TO_HEAD = {
    "event_next_60m": "event_60m",
    "event_next_120m": "event_120m",
    "deficit_next_60m": "deficit_60m",
    "regime_forecast": "regime_future",
    "event_60m": "event_60m",
    "event_120m": "event_120m",
    "deficit_60m": "deficit_60m",
    "regime_future": "regime_future",
}


def normalize_task_name(task_name: str) -> str:
    return PROJECT_TASK_TO_HEAD.get(task_name, task_name)


@dataclass
class TFTPreparedBatch:
    x_seq: np.ndarray
    seq_mask: np.ndarray
    x_static: np.ndarray
    targets: dict[str, np.ndarray]
    target_masks: dict[str, np.ndarray]


class TFTTensorDataset(Dataset[tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]]):
    def __init__(self, batch: TFTPreparedBatch) -> None:
        self.x_seq = torch.from_numpy(batch.x_seq.astype(np.float32))
        self.seq_mask = torch.from_numpy(batch.seq_mask.astype(np.bool_))
        self.x_static = torch.from_numpy(batch.x_static.astype(np.float32))
        self.targets = {
            task_name: torch.from_numpy(values)
            for task_name, values in batch.targets.items()
        }
        self.target_masks = {
            task_name: torch.from_numpy(values.astype(np.bool_))
            for task_name, values in batch.target_masks.items()
        }

    def __len__(self) -> int:
        return int(self.x_seq.shape[0])

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        inputs = {
            "x_seq": self.x_seq[index],
            "seq_mask": self.seq_mask[index],
            "x_static": self.x_static[index],
        }
        targets = {
            task_name: {
                "values": values[index],
                "mask": self.target_masks[task_name][index],
            }
            for task_name, values in self.targets.items()
        }
        return inputs, targets


class BaseTFTNetwork(nn.Module):
    """Clean TFT-style baseline for the project batch contract."""

    def __init__(
        self,
        *,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
        enabled_tasks: list[str],
        d_model: int,
        hidden_size: int,
        attention_heads: int,
        num_layers_local: int,
        dropout: float,
        readout_mode: str = "last_valid",
        use_static: bool = False,
        n_regimes: int = 3,
    ) -> None:
        super().__init__()
        self.enabled_tasks = enabled_tasks
        self.sequence_dim = sequence_dim
        self.static_dim = static_dim
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.use_static = use_static and static_dim > 0
        self.n_regimes = n_regimes

        self.seq_embed = ContinuousVariableEmbedding(sequence_dim, d_model=d_model, dropout=dropout)
        self.static_encoder = (
            StaticContextEncoder(static_dim, d_model=d_model, hidden_dim=hidden_size, dropout=dropout)
            if self.use_static
            else None
        )
        self.seq_vsn = VariableSelectionNetwork(
            n_vars=sequence_dim,
            d_model=d_model,
            hidden_dim=hidden_size,
            dropout=dropout,
            use_static_context=self.use_static,
        )
        self.local_encoder = LocalLSTMEncoder(
            d_model=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers_local,
            dropout=dropout,
        )
        self.attn_block = CausalInterpretableAttention(d_model=d_model, n_heads=attention_heads, dropout=dropout)
        self.post_grn = GRN(d_model, hidden_size, d_out=d_model, dropout=dropout, context_dim=d_model if self.use_static else None)
        self.readout = ReadoutBlock(d_model=d_model, mode=readout_mode)
        self.heads = MultiTaskHeads(d_model=d_model, n_regimes=n_regimes, enabled_tasks=enabled_tasks)

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled_tasks": list(self.enabled_tasks),
            "architecture_class": self.__class__.__name__,
        }

    def build_static_context(self, x_static: Tensor | None) -> Tensor | None:
        if self.static_encoder is None:
            return None
        return self.static_encoder(x_static)

    def embed_sequence(self, x_seq: Tensor, batch: dict[str, Tensor], static_context: Tensor | None) -> tuple[Tensor, Tensor]:
        x_emb = self.seq_embed(x_seq)
        selected, var_weights = self.seq_vsn(x_emb, static_context=static_context)
        return selected, var_weights

    def encode_local(self, z_seq: Tensor, seq_mask: Tensor | None) -> Tensor:
        return self.local_encoder(z_seq, seq_mask)

    def attend(self, z_local: Tensor, seq_mask: Tensor | None) -> tuple[Tensor, Tensor]:
        return self.attn_block(z_local, seq_mask)

    def post_process(self, z_attn: Tensor, static_context: Tensor | None) -> Tensor:
        if static_context is not None:
            context = static_context.unsqueeze(1).expand(-1, z_attn.shape[1], -1)
            return self.post_grn(z_attn, context=context)
        return self.post_grn(z_attn)

    def readout_state(self, z_post: Tensor, seq_mask: Tensor | None) -> Tensor:
        return self.readout(z_post, seq_mask)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x_seq = batch["x_seq"]
        seq_mask = batch.get("seq_mask")
        x_static = batch.get("x_static")

        static_context = self.build_static_context(x_static)
        z_seq, var_weights = self.embed_sequence(x_seq, batch, static_context)
        z_local = self.encode_local(z_seq, seq_mask)
        z_attn, attn_weights = self.attend(z_local, seq_mask)
        z_post = self.post_process(z_attn, static_context)
        z_final = self.readout_state(z_post, seq_mask)

        outputs = self.heads(z_final)
        outputs["var_weights"] = var_weights
        outputs["attn_weights"] = attn_weights
        outputs["shared_representation"] = z_final
        return outputs


class TFTFamilyClassifier(TorchSequenceClassifier):
    """Project-compatible TFT classifier with optional multitask supervision."""

    model_name = "tft_family"
    network_class = BaseTFTNetwork

    def __init__(
        self,
        *,
        d_model: int = 64,
        hidden_size: int = 64,
        attention_heads: int = 4,
        num_layers_local: int = 1,
        dropout: float = 0.1,
        readout_mode: str = "last_valid",
        enabled_tasks: list[str] | None = None,
        active_task: str | None = None,
        use_static: bool = False,
        use_time_features: bool = True,
        variable_selection: bool = True,
        use_missingness_channels: bool = True,
        task_weights: dict[str, float] | None = None,
        n_regimes: int = 3,
        class_names: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.num_layers_local = num_layers_local
        self.dropout = dropout
        self.readout_mode = readout_mode
        self.enabled_tasks = [normalize_task_name(task_name) for task_name in (enabled_tasks or [])]
        self.active_task = normalize_task_name(active_task) if active_task else None
        self.use_static = use_static
        self.use_time_features = use_time_features
        self.variable_selection_enabled = variable_selection
        self.use_missingness_channels = use_missingness_channels
        self.task_weights = {normalize_task_name(task_name): weight for task_name, weight in (task_weights or {}).items()}
        self.n_regimes = n_regimes
        self.class_names_config = class_names or []
        self.task_output_dims_: dict[str, int] = {}
        self.sequence_feature_names_: list[str] = []
        self.static_feature_names_: list[str] = []
        self.active_head_: str | None = None

    def build_tft_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
    ) -> BaseTFTNetwork:
        return self.network_class(
            sequence_dim=sequence_dim,
            static_dim=static_dim,
            sequence_length=sequence_length,
            enabled_tasks=list(self.enabled_tasks),
            d_model=self.d_model,
            hidden_size=self.hidden_size,
            attention_heads=self.attention_heads,
            num_layers_local=self.num_layers_local,
            dropout=self.dropout,
            readout_mode=self.readout_mode,
            use_static=self.use_static,
            n_regimes=self.n_regimes,
        )

    def build_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
        num_classes: int,
    ) -> nn.Module:
        return self.build_tft_network(sequence_dim, static_dim, sequence_length)

    def _resolve_enabled_tasks(self, task_name: str) -> None:
        active_head = normalize_task_name(task_name)
        self.active_head_ = active_head
        if not self.enabled_tasks:
            self.enabled_tasks = [active_head]
        elif active_head not in self.enabled_tasks:
            self.enabled_tasks.append(active_head)
        if self.task_output_dims_.get(active_head) is None:
            self.task_output_dims_[active_head] = self.n_regimes if active_head == "regime_future" else 1

    def _feature_names_with_missingness(self, base_names: list[str]) -> list[str]:
        if not self.use_missingness_channels:
            return list(base_names)
        return [*base_names, *[f"{name}__missing" for name in base_names]]

    def _compose_tft_inputs(self, bundle: SequenceArrayBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequence = np.nan_to_num(bundle.sequence.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        sequence_mask = (bundle.sequence_mask.max(axis=2) > 0).astype(np.bool_)
        if self.use_missingness_channels:
            sequence = np.concatenate([sequence, bundle.sequence_mask.astype(np.float32)], axis=2)

        static = np.nan_to_num(bundle.static.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if static.shape[1] and self.use_missingness_channels:
            static = np.concatenate([static, bundle.static_mask.astype(np.float32)], axis=1)
        return sequence, sequence_mask, static

    def _build_targets(self, X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        active_head = self.active_head_ or normalize_task_name(X.attrs.get("task_name", "event_next_60m"))
        targets: dict[str, np.ndarray] = {active_head: pd.Series(y).astype(int).to_numpy()}
        target_masks: dict[str, np.ndarray] = {active_head: np.ones(len(y), dtype=np.bool_)}

        auxiliary = X.attrs.get("auxiliary_targets")
        if isinstance(auxiliary, pd.DataFrame) and not auxiliary.empty:
            for column in auxiliary.columns:
                head_name = normalize_task_name(column)
                if head_name not in self.enabled_tasks:
                    continue
                values = auxiliary[column]
                mask = values.notna().to_numpy(dtype=np.bool_)
                targets[head_name] = values.fillna(0).astype(int).to_numpy()
                target_masks[head_name] = mask
                self.task_output_dims_.setdefault(head_name, self.n_regimes if head_name == "regime_future" else 1)
        return targets, target_masks

    def _subset_prepared_batch(self, batch: TFTPreparedBatch, indices: np.ndarray) -> TFTPreparedBatch:
        return TFTPreparedBatch(
            x_seq=batch.x_seq[indices],
            seq_mask=batch.seq_mask[indices],
            x_static=batch.x_static[indices],
            targets={task_name: values[indices] for task_name, values in batch.targets.items()},
            target_masks={task_name: values[indices] for task_name, values in batch.target_masks.items()},
        )

    def _build_multitask_class_weights(self, batch: TFTPreparedBatch) -> dict[str, Tensor]:
        weights: dict[str, Tensor] = {}
        for task_name, values in batch.targets.items():
            mask = batch.target_masks[task_name]
            valid = values[mask]
            if valid.size == 0:
                continue
            if self.task_output_dims_.get(task_name, 1) == 1:
                positives = float(valid.sum())
                negatives = float(len(valid) - positives)
                if positives > 0 and negatives > 0:
                    weights[task_name] = torch.tensor(negatives / positives, dtype=torch.float32)
            else:
                bincount = np.bincount(valid.astype(np.int64), minlength=self.task_output_dims_[task_name]).astype(np.float32)
                bincount = np.where(bincount > 0, bincount, 1.0)
                class_weights = bincount.sum() / (len(bincount) * bincount)
                weights[task_name] = torch.tensor(class_weights, dtype=torch.float32)
        return weights

    def _consistency_loss(self, outputs: dict[str, Tensor]) -> Tensor:
        return torch.tensor(0.0, device=next(iter(outputs.values())).device)

    def _prepare_batch(self, X: pd.DataFrame, y: pd.Series) -> TFTPreparedBatch:
        self._resolve_enabled_tasks(X.attrs.get("task_name", self.active_task or "event_next_60m"))
        bundle = self.adapter_.fit_transform(X) if self.adapter_ is not None and not self.adapter_.columns_ else None
        raise RuntimeError("Internal misuse: _prepare_batch expects adapter to be fitted explicitly.")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: pd.Series | None = None,
    ) -> "TFTFamilyClassifier":
        target = pd.Series(y).astype(int)
        if target.empty:
            raise ValueError("Cannot fit a TFT model on an empty training split.")

        runtime_task = X.attrs.get("task_name", self.active_task or "event_next_60m")
        self._resolve_enabled_tasks(runtime_task)

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
                "enabled_tasks": list(self.enabled_tasks),
                "active_task": self.active_head_,
            }
            return self

        self.adapter_ = SequenceFeatureAdapter(
            representation=self.representation,
            context_steps=self.context_steps,
            static_mode=self.static_mode,
        )
        bundle = self.adapter_.fit_transform(X)
        adapter_description = self.adapter_.describe()
        self.sequence_feature_names_ = self._feature_names_with_missingness(adapter_description["sequence_features"])
        self.static_feature_names_ = list(adapter_description["static_features"])
        sequence_input, sequence_mask, static_input = self._compose_tft_inputs(bundle)
        targets, target_masks = self._build_targets(X, target)
        for task_name in list(targets):
            self.task_output_dims_.setdefault(task_name, self.n_regimes if task_name == "regime_future" else 1)

        full_batch = TFTPreparedBatch(
            x_seq=sequence_input,
            seq_mask=sequence_mask,
            x_static=static_input,
            targets=targets,
            target_masks=target_masks,
        )
        train_indices, val_indices = self._validation_indices(len(target))
        train_batch = self._subset_prepared_batch(full_batch, train_indices)
        val_batch = self._subset_prepared_batch(full_batch, val_indices) if len(val_indices) else None

        train_dataset = TFTTensorDataset(train_batch)
        val_dataset = TFTTensorDataset(val_batch) if val_batch is not None else None
        loader_generator = torch.Generator().manual_seed(self.random_state)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=loader_generator)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False) if val_dataset is not None else None

        device = self._resolve_device()
        self.network_ = self.build_network(
            sequence_dim=int(sequence_input.shape[2]),
            static_dim=int(static_input.shape[1]),
            sequence_length=int(sequence_input.shape[1]),
            num_classes=len(self.classes_),
        ).to(device)

        class_weights = self._build_multitask_class_weights(train_batch)
        task_weights = {task_name: self.task_weights.get(task_name, 1.0) for task_name in self.enabled_tasks}
        criterion = MultiTaskLoss(task_weights=task_weights, class_weights=class_weights)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_state: dict[str, Tensor] | None = None
        best_monitor = float("inf")
        best_epoch = 0
        stale_epochs = 0
        epochs_ran = 0
        start_time = time.perf_counter()

        for epoch in range(1, self.max_epochs + 1):
            epochs_ran = epoch
            train_loss = self._run_tft_epoch(train_loader, criterion, optimizer, device)
            monitor_loss = self._evaluate_tft_loss(val_loader, criterion, device) if val_loader else train_loss
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
        parameter_count = int(sum(parameter.numel() for parameter in self.network_.parameters()))
        network_metadata = self.network_.metadata() if hasattr(self.network_, "metadata") else {}
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
            "sequence_input_shape": [int(sequence_input.shape[0]), int(sequence_input.shape[1]), int(sequence_input.shape[2])],
            "static_input_shape": [int(static_input.shape[0]), int(static_input.shape[1])],
            "parameter_count": parameter_count,
            "enabled_tasks": list(self.enabled_tasks),
            "active_task": self.active_head_,
            "auxiliary_supervision_tasks": sorted(task_name for task_name in targets if task_name != self.active_head_),
            "sequence_variable_names": list(self.sequence_feature_names_),
            "static_feature_names": list(self.static_feature_names_),
            **adapter_description,
            **network_metadata,
        }
        return self

    def _run_tft_epoch(
        self,
        loader: DataLoader[tuple[dict[str, Tensor], dict[str, dict[str, Tensor]]]],
        criterion: MultiTaskLoss,
        optimizer: torch.optim.Optimizer,
        device: str,
    ) -> float:
        if self.network_ is None:
            raise RuntimeError("Network is not initialized.")
        self.network_.train()
        running_loss = 0.0
        total = 0
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            batch_inputs = {name: value.to(device) for name, value in batch_inputs.items()}
            targets = {
                task_name: {
                    "values": values["values"].to(device),
                    "mask": values["mask"].to(device),
                }
                for task_name, values in batch_targets.items()
            }
            outputs = self.network_(batch_inputs)
            loss, _ = criterion(outputs, targets)
            loss = loss + self._consistency_loss(outputs)
            loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.network_.parameters(), self.grad_clip)
            optimizer.step()
            batch_size = int(batch_inputs["x_seq"].shape[0])
            running_loss += float(loss.item()) * batch_size
            total += batch_size
        return running_loss / max(total, 1)

    def _evaluate_tft_loss(
        self,
        loader: DataLoader[tuple[dict[str, Tensor], dict[str, dict[str, Tensor]]]] | None,
        criterion: MultiTaskLoss,
        device: str,
    ) -> float:
        if loader is None:
            return 0.0
        if self.network_ is None:
            raise RuntimeError("Network is not initialized.")
        self.network_.eval()
        running_loss = 0.0
        total = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in loader:
                batch_inputs = {name: value.to(device) for name, value in batch_inputs.items()}
                targets = {
                    task_name: {
                        "values": values["values"].to(device),
                        "mask": values["mask"].to(device),
                    }
                    for task_name, values in batch_targets.items()
                }
                outputs = self.network_(batch_inputs)
                loss, _ = criterion(outputs, targets)
                loss = loss + self._consistency_loss(outputs)
                batch_size = int(batch_inputs["x_seq"].shape[0])
                running_loss += float(loss.item()) * batch_size
                total += batch_size
        return running_loss / max(total, 1)

    def _iter_prediction_batches(self, sequence_input: np.ndarray, sequence_mask: np.ndarray, static_input: np.ndarray) -> DataLoader[dict[str, Tensor]]:
        class _PredictionDataset(Dataset[dict[str, Tensor]]):
            def __init__(self, x_seq: np.ndarray, seq_mask: np.ndarray, x_static: np.ndarray) -> None:
                self.x_seq = torch.from_numpy(x_seq.astype(np.float32))
                self.seq_mask = torch.from_numpy(seq_mask.astype(np.bool_))
                self.x_static = torch.from_numpy(x_static.astype(np.float32))

            def __len__(self) -> int:
                return int(self.x_seq.shape[0])

            def __getitem__(self, index: int) -> dict[str, Tensor]:
                return {
                    "x_seq": self.x_seq[index],
                    "seq_mask": self.seq_mask[index],
                    "x_static": self.x_static[index],
                }

        return DataLoader(_PredictionDataset(sequence_input, sequence_mask, static_input), batch_size=self.batch_size, shuffle=False)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.constant_probabilities_ is not None:
            repeated = np.tile(self.constant_probabilities_, (len(X), 1))
            return pd.DataFrame(repeated, index=X.index, columns=self._probability_columns())
        if self.adapter_ is None or self.network_ is None or self.active_head_ is None:
            raise RuntimeError("TFT model must be fitted before prediction.")

        bundle = self.adapter_.transform(X)
        sequence_input, sequence_mask, static_input = self._compose_tft_inputs(bundle)
        loader = self._iter_prediction_batches(sequence_input, sequence_mask, static_input)

        self.network_.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_inputs in loader:
                network_output = self.network_(batch_inputs)
                logits = network_output[self.active_head_]
                if logits.shape[-1] == 1:
                    positive = torch.sigmoid(logits.squeeze(-1))
                    probability = torch.stack([1.0 - positive, positive], dim=-1)
                else:
                    probability = torch.softmax(logits, dim=-1)
                outputs.append(probability.cpu().numpy())
        probabilities = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, len(self.classes_)))
        columns = [f"class_{int(class_value)}" for class_value in range(probabilities.shape[1])]
        return pd.DataFrame(probabilities, index=X.index, columns=columns)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities.to_numpy(), axis=1)
        return pd.Series(predictions, index=X.index, name="prediction")


class TFTBaseClassifier(TFTFamilyClassifier):
    model_name = "tft_base"
