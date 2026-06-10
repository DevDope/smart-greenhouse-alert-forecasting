"""Experimental Stage 3 Zappa: multitask MMoE transformer/TFT hybrid."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ...transformers.modules import (
    GRN,
    LocalDilatedTCN,
    MMoEBlock,
    RevIN,
    SparseVariableSelectionNetwork,
)
from ...transformers.tft_base import BaseTFTNetwork, TFTFamilyClassifier


class ZappaMMoENetwork(BaseTFTNetwork):
    def __init__(
        self,
        *,
        num_experts: int = 4,
        expert_hidden_size: int = 64,
        gate_mode: str = "softmax",
        gate_top_k: int | None = None,
        tcn_kernel_size: int = 3,
        dilation_schedule: list[int] | None = None,
        use_revin: bool = True,
        has_missingness_channels: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.has_missingness_channels = has_missingness_channels
        observed_dim = self.sequence_dim // 2 if self.has_missingness_channels and self.sequence_dim % 2 == 0 else self.sequence_dim
        self.revin = RevIN(observed_dim) if use_revin else None
        self.seq_vsn = SparseVariableSelectionNetwork(
            n_vars=self.sequence_dim,
            d_model=self.d_model,
            hidden_dim=self.hidden_size,
            dropout=self.dropout,
            use_static_context=self.use_static,
            sparse_mode=gate_mode,
            top_k=gate_top_k,
        )
        self.local_encoder = LocalDilatedTCN(
            input_dim=self.d_model,
            hidden_dim=self.hidden_size,
            output_dim=self.d_model,
            kernel_size=tcn_kernel_size,
            dilation_schedule=list(dilation_schedule or [1, 2, 4]),
            dropout=self.dropout,
        )
        self.mmoe = MMoEBlock(
            input_dim=self.d_model,
            output_dim=self.d_model,
            hidden_dim=expert_hidden_size,
            num_experts=num_experts,
            task_names=list(self.enabled_tasks),
            dropout=self.dropout,
            gate_mode=gate_mode,
            gate_top_k=gate_top_k,
        )
        self.task_adapters = nn.ModuleDict(
            {task_name: GRN(self.d_model, expert_hidden_size, d_out=self.d_model, dropout=self.dropout) for task_name in self.enabled_tasks}
        )
        self.task_heads = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.d_model, self.n_regimes if task_name == "regime_future" else 1),
                )
                for task_name in self.enabled_tasks
            }
        )
        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size
        self.gate_mode = gate_mode
        self.gate_top_k = gate_top_k
        self.tcn_kernel_size = tcn_kernel_size
        self.dilation_schedule = list(dilation_schedule or [1, 2, 4])
        self.use_revin = use_revin

    def _maybe_normalize(self, x_seq: Tensor) -> Tensor:
        if self.revin is None:
            return x_seq
        observed_dim = self.revin.num_features
        observed = x_seq[..., :observed_dim]
        normalized, _ = self.revin(observed, mode="norm")
        if observed_dim == x_seq.shape[-1]:
            return normalized
        return torch.cat([normalized, x_seq[..., observed_dim:]], dim=-1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x_seq = self._maybe_normalize(batch["x_seq"])
        seq_mask = batch.get("seq_mask")
        x_static = batch.get("x_static")
        static_context = self.build_static_context(x_static)

        z_seq, var_weights = self.embed_sequence(x_seq, batch, static_context)
        z_local = self.local_encoder(z_seq, seq_mask)
        z_attn, attn_weights = self.attend(z_local, seq_mask)
        z_post = self.post_process(z_attn, static_context)
        shared_state = self.readout_state(z_post, seq_mask)

        task_states, gate_weights = self.mmoe(shared_state)
        outputs: dict[str, Tensor] = {}
        routed_states: list[Tensor] = []
        task_gate_rows: list[Tensor] = []
        for task_name in self.enabled_tasks:
            adapted = self.task_adapters[task_name](task_states[task_name])
            routed_states.append(adapted)
            task_gate_rows.append(gate_weights[task_name])
            outputs[task_name] = self.task_heads[task_name](adapted)

        outputs["var_weights"] = var_weights
        outputs["attn_weights"] = attn_weights
        outputs["mmoe_gate_weights"] = torch.stack(task_gate_rows, dim=1)
        outputs["task_representations"] = torch.stack(routed_states, dim=1)
        return outputs

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "num_experts": self.num_experts,
                "expert_hidden_size": self.expert_hidden_size,
                "gate_mode": self.gate_mode,
                "gate_top_k": self.gate_top_k,
                "tcn_kernel_size": self.tcn_kernel_size,
                "dilation_schedule": list(self.dilation_schedule),
                "use_revin": self.use_revin,
            }
        )
        return metadata


class ZappaMMoEClassifier(TFTFamilyClassifier):
    model_name = "zappa_mmoe"

    def __init__(
        self,
        *,
        num_experts: int = 4,
        expert_hidden_size: int = 64,
        gate_mode: str = "softmax",
        gate_top_k: int | None = None,
        tcn_kernel_size: int = 3,
        dilation_schedule: list[int] | None = None,
        use_revin: bool = True,
        consistency_loss_weight: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size
        self.gate_mode = gate_mode
        self.gate_top_k = gate_top_k
        self.tcn_kernel_size = tcn_kernel_size
        self.dilation_schedule = list(dilation_schedule or [1, 2, 4])
        self.use_revin = use_revin
        self.consistency_loss_weight = consistency_loss_weight

    def build_tft_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
    ) -> BaseTFTNetwork:
        return ZappaMMoENetwork(
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
            num_experts=self.num_experts,
            expert_hidden_size=self.expert_hidden_size,
            gate_mode=self.gate_mode,
            gate_top_k=self.gate_top_k,
            tcn_kernel_size=self.tcn_kernel_size,
            dilation_schedule=self.dilation_schedule,
            use_revin=self.use_revin,
            has_missingness_channels=self.use_missingness_channels,
        )

    def _consistency_loss(self, outputs: dict[str, Tensor]) -> Tensor:
        if self.consistency_loss_weight <= 0 or "event_60m" not in outputs or "event_120m" not in outputs:
            return super()._consistency_loss(outputs)
        prob_60 = torch.sigmoid(outputs["event_60m"].squeeze(-1))
        prob_120 = torch.sigmoid(outputs["event_120m"].squeeze(-1))
        penalty = torch.relu(prob_60 - prob_120)
        return self.consistency_loss_weight * (penalty**2).mean()
