"""Archived Stage 3 Wilson: simple multiscale temporal model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ...transformers.modules import GRN, LocalDilatedTCN, MultiScaleSequenceBuilder, RevIN
from ...transformers.tft_base import BaseTFTNetwork, TFTFamilyClassifier


class WilsonTCNMultiscaleNetwork(BaseTFTNetwork):
    def __init__(
        self,
        *,
        scale_kernel_sizes: list[int] | None = None,
        tcn_kernel_size: int = 3,
        dilation_schedule: list[int] | None = None,
        fusion_hidden_size: int = 96,
        use_revin: bool = True,
        has_missingness_channels: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.has_missingness_channels = has_missingness_channels
        observed_dim = self.sequence_dim // 2 if self.has_missingness_channels and self.sequence_dim % 2 == 0 else self.sequence_dim
        self.revin = RevIN(observed_dim) if use_revin else None
        self.scale_kernel_sizes = list(scale_kernel_sizes or [3, 6])
        self.multiscale = MultiScaleSequenceBuilder(self.scale_kernel_sizes)
        self.local_encoder = LocalDilatedTCN(
            input_dim=self.d_model,
            hidden_dim=self.hidden_size,
            output_dim=self.d_model,
            kernel_size=tcn_kernel_size,
            dilation_schedule=list(dilation_schedule or [1, 2, 4]),
            dropout=self.dropout,
        )
        self.fusion = GRN(self.d_model * 3, fusion_hidden_size, d_out=self.d_model, dropout=self.dropout)
        self.tcn_kernel_size = tcn_kernel_size
        self.dilation_schedule = list(dilation_schedule or [1, 2, 4])
        self.fusion_hidden_size = fusion_hidden_size
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

        scales = self.multiscale(x_seq)
        ordered_names = ["fine", "mid", "coarse"]
        summaries: list[Tensor] = []
        branch_summaries: list[Tensor] = []
        fine_var_weights: Tensor | None = None
        attn_weights: Tensor | None = None

        for branch_name in ordered_names:
            branch_input = scales.get(branch_name, scales["fine"])
            z_seq, var_weights = self.embed_sequence(branch_input, batch, static_context)
            z_local = self.local_encoder(z_seq, seq_mask)
            if branch_name == "fine":
                fine_var_weights = var_weights
                z_attn, attn_weights = self.attend(z_local, seq_mask)
                z_local = self.post_process(z_attn, static_context)
            summary = self.readout_state(z_local, seq_mask)
            summaries.append(summary)
            branch_summaries.append(summary)

        fused = self.fusion(torch.cat(summaries, dim=-1))
        outputs = self.heads(fused)
        outputs["var_weights"] = fine_var_weights if fine_var_weights is not None else torch.empty(0)
        outputs["attn_weights"] = attn_weights if attn_weights is not None else torch.empty(0)
        outputs["multiscale_summaries"] = torch.stack(branch_summaries, dim=1)
        return outputs

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "scale_kernel_sizes": list(self.scale_kernel_sizes),
                "tcn_kernel_size": self.tcn_kernel_size,
                "dilation_schedule": list(self.dilation_schedule),
                "fusion_hidden_size": self.fusion_hidden_size,
                "use_revin": self.use_revin,
            }
        )
        return metadata


class WilsonTCNMultiscaleClassifier(TFTFamilyClassifier):
    model_name = "wilson_tcn_multiscale"

    def __init__(
        self,
        *,
        scale_kernel_sizes: list[int] | None = None,
        tcn_kernel_size: int = 3,
        dilation_schedule: list[int] | None = None,
        fusion_hidden_size: int = 96,
        use_revin: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.scale_kernel_sizes = list(scale_kernel_sizes or [3, 6])
        self.tcn_kernel_size = tcn_kernel_size
        self.dilation_schedule = list(dilation_schedule or [1, 2, 4])
        self.fusion_hidden_size = fusion_hidden_size
        self.use_revin = use_revin

    def build_tft_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
    ) -> BaseTFTNetwork:
        return WilsonTCNMultiscaleNetwork(
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
            scale_kernel_sizes=self.scale_kernel_sizes,
            tcn_kernel_size=self.tcn_kernel_size,
            dilation_schedule=self.dilation_schedule,
            fusion_hidden_size=self.fusion_hidden_size,
            use_revin=self.use_revin,
            has_missingness_channels=self.use_missingness_channels,
        )
