"""Zappa TFT variant: task-aware routing and tokenized multitask adaptation."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .modules import TaskRoutingGate, TaskSpecificAdapter, TaskTokenCrossAttention
from .tft_base import BaseTFTNetwork, TFTFamilyClassifier


class TFTZappaNetwork(BaseTFTNetwork):
    def __init__(
        self,
        *,
        adapter_hidden_size: int = 64,
        routing_hidden_size: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.task_tokens = TaskTokenCrossAttention(self.d_model, task_names=list(self.enabled_tasks), dropout=self.dropout)
        self.task_adapters = nn.ModuleDict(
            {task_name: TaskSpecificAdapter(self.d_model, hidden_dim=adapter_hidden_size, dropout=self.dropout) for task_name in self.enabled_tasks}
        )
        self.routing_gates = nn.ModuleDict(
            {task_name: TaskRoutingGate(self.d_model) for task_name in self.enabled_tasks}
        )
        self.task_heads = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, routing_hidden_size),
                    nn.GELU(),
                    nn.Linear(routing_hidden_size, self.n_regimes if task_name == "regime_future" else 1),
                )
                for task_name in self.enabled_tasks
            }
        )
        self.adapter_hidden_size = adapter_hidden_size
        self.routing_hidden_size = routing_hidden_size

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x_seq = batch["x_seq"]
        seq_mask = batch.get("seq_mask")
        x_static = batch.get("x_static")
        static_context = self.build_static_context(x_static)

        z_seq, var_weights = self.embed_sequence(x_seq, batch, static_context)
        z_local = self.encode_local(z_seq, seq_mask)
        z_attn, attn_weights = self.attend(z_local, seq_mask)
        z_post = self.post_process(z_attn, static_context)
        shared_state = self.readout_state(z_post, seq_mask)

        task_contexts = self.task_tokens(z_post, seq_mask)
        outputs: dict[str, Tensor] = {}
        routed_states: dict[str, Tensor] = {}
        for task_name in self.enabled_tasks:
            adapted = self.task_adapters[task_name](task_contexts[task_name])
            routed = self.routing_gates[task_name](shared_state, adapted)
            routed_states[task_name] = routed
            outputs[task_name] = self.task_heads[task_name](routed)

        outputs["var_weights"] = var_weights
        outputs["attn_weights"] = attn_weights
        outputs["task_representations"] = torch.stack([routed_states[task_name] for task_name in self.enabled_tasks], dim=1)
        return outputs

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "adapter_hidden_size": self.adapter_hidden_size,
                "routing_hidden_size": self.routing_hidden_size,
            }
        )
        return metadata


class TFTZappaClassifier(TFTFamilyClassifier):
    model_name = "tft_zappa"

    def __init__(
        self,
        *,
        adapter_hidden_size: int = 64,
        routing_hidden_size: int = 64,
        consistency_loss_weight: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.adapter_hidden_size = adapter_hidden_size
        self.routing_hidden_size = routing_hidden_size
        self.consistency_loss_weight = consistency_loss_weight

    def build_tft_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
    ) -> BaseTFTNetwork:
        return TFTZappaNetwork(
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
            adapter_hidden_size=self.adapter_hidden_size,
            routing_hidden_size=self.routing_hidden_size,
        )

    def _consistency_loss(self, outputs: dict[str, Tensor]) -> Tensor:
        if self.consistency_loss_weight <= 0:
            return super()._consistency_loss(outputs)
        if "event_60m" not in outputs or "event_120m" not in outputs:
            return super()._consistency_loss(outputs)
        prob_60 = torch.sigmoid(outputs["event_60m"].squeeze(-1))
        prob_120 = torch.sigmoid(outputs["event_120m"].squeeze(-1))
        monotonicity_penalty = torch.relu(prob_60 - prob_120)
        return self.consistency_loss_weight * (monotonicity_penalty ** 2).mean()
