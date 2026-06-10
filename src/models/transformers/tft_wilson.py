"""Wilson TFT variant: multiscale long-memory and trend-aware."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .modules import (
    LocalGRUEncoder,
    LongContextFusionBlock,
    LongContextSummaryBlock,
    MultiScaleSequenceBuilder,
    TrendResidualDecomposer,
)
from .tft_base import BaseTFTNetwork, TFTFamilyClassifier


class TFTWilsonNetwork(BaseTFTNetwork):
    def __init__(
        self,
        *,
        trend_kernel_size: int = 5,
        pool_kernel_sizes: list[int] | None = None,
        fusion_hidden_size: int = 96,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.trend_residual = TrendResidualDecomposer(kernel_size=trend_kernel_size)
        self.multiscale = MultiScaleSequenceBuilder(pool_kernel_sizes or [3, 6])
        self.local_encoder = LocalGRUEncoder(
            d_model=self.d_model,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=self.dropout,
        )
        self.summary_block = LongContextSummaryBlock(self.d_model, hidden_dim=fusion_hidden_size, dropout=self.dropout)
        self.fusion_block = LongContextFusionBlock(self.d_model, hidden_dim=fusion_hidden_size, dropout=self.dropout)
        self.trend_kernel_size = trend_kernel_size
        self.pool_kernel_sizes = list(pool_kernel_sizes or [3, 6])
        self.fusion_hidden_size = fusion_hidden_size

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x_seq = batch["x_seq"]
        seq_mask = batch.get("seq_mask")
        x_static = batch.get("x_static")
        static_context = self.build_static_context(x_static)

        trend, residual = self.trend_residual(x_seq)
        scales = self.multiscale(residual)
        branch_inputs = {
            "fine": scales["fine"],
            "mid": scales.get("mid", scales["fine"]),
            "coarse": scales.get("coarse", scales["fine"]),
            "trend": trend,
            "residual": residual,
        }

        branch_states: dict[str, Tensor] = {}
        branch_weights: dict[str, Tensor] = {}
        for name, branch in branch_inputs.items():
            embedded, weights = self.embed_sequence(branch, batch, static_context)
            branch_states[name] = self.encode_local(embedded, seq_mask)
            branch_weights[name] = weights

        z_attn, attn_weights = self.attend(branch_states["fine"], seq_mask)
        z_post = self.post_process(z_attn, static_context)
        local_state = self.readout_state(z_post, seq_mask)

        summaries = self.summary_block(branch_states, seq_mask)
        fused = self.fusion_block(local_state, summaries)
        outputs = self.heads(fused)
        outputs["var_weights"] = branch_weights["fine"]
        outputs["attn_weights"] = attn_weights
        outputs["multiscale_summaries"] = torch.stack(list(summaries.values()), dim=1)
        return outputs

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "trend_kernel_size": self.trend_kernel_size,
                "pool_kernel_sizes": list(self.pool_kernel_sizes),
                "fusion_hidden_size": self.fusion_hidden_size,
            }
        )
        return metadata


class TFTWilsonClassifier(TFTFamilyClassifier):
    model_name = "tft_wilson"

    def __init__(
        self,
        *,
        trend_kernel_size: int = 5,
        pool_kernel_sizes: list[int] | None = None,
        fusion_hidden_size: int = 96,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.trend_kernel_size = trend_kernel_size
        self.pool_kernel_sizes = list(pool_kernel_sizes or [3, 6])
        self.fusion_hidden_size = fusion_hidden_size

    def build_tft_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
    ) -> BaseTFTNetwork:
        return TFTWilsonNetwork(
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
            trend_kernel_size=self.trend_kernel_size,
            pool_kernel_sizes=self.pool_kernel_sizes,
            fusion_hidden_size=self.fusion_hidden_size,
        )
