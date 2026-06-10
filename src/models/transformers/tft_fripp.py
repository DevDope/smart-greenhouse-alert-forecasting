"""Fripp TFT variant: sparse, robust, lower-complexity."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .modules import (
    LowRankAttention,
    ReadoutBlock,
    SparseVariableSelectionNetwork,
    VariableSubsetController,
)
from .tft_base import BaseTFTNetwork, TFTFamilyClassifier


class TFTFrippNetwork(BaseTFTNetwork):
    def __init__(
        self,
        *,
        variable_names: list[str],
        variable_subset: list[str] | None = None,
        sparse_mode: str = "topk",
        sparse_top_k: int | None = None,
        low_rank_dim: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.variable_controller = VariableSubsetController(variable_names=variable_names, subset=variable_subset)
        subset_dim = len(self.variable_controller.subset)
        self.seq_embed = self.seq_embed.__class__(subset_dim, d_model=self.d_model, dropout=self.dropout)
        self.seq_vsn = SparseVariableSelectionNetwork(
            n_vars=subset_dim,
            d_model=self.d_model,
            hidden_dim=max(16, self.hidden_size // 2),
            dropout=self.dropout,
            use_static_context=self.use_static,
            sparse_mode=sparse_mode,
            top_k=sparse_top_k,
        )
        self.low_rank_attention = LowRankAttention(
            d_model=self.d_model,
            n_heads=max(1, self.attention_heads // 2),
            rank_dim=low_rank_dim,
            dropout=self.dropout,
        )
        self.readout = ReadoutBlock(d_model=self.d_model, mode="robust_pool")
        self.sparse_mode = sparse_mode
        self.sparse_top_k = sparse_top_k or max(1, subset_dim // 2)
        self.low_rank_dim = low_rank_dim

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x_seq = self.variable_controller(batch["x_seq"])
        seq_mask = batch.get("seq_mask")
        x_static = batch.get("x_static")
        static_context = self.build_static_context(x_static)

        x_emb = self.seq_embed(x_seq)
        z_seq, var_weights = self.seq_vsn(x_emb, static_context=static_context)
        z_local = self.encode_local(z_seq, seq_mask)
        z_attn, attn_weights = self.low_rank_attention(z_local, seq_mask)
        z_post = self.post_process(z_attn, static_context)
        z_final = self.readout_state(z_post, seq_mask)

        outputs = self.heads(z_final)
        outputs["var_weights"] = var_weights
        outputs["attn_weights"] = attn_weights
        return outputs

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                **self.variable_controller.metadata(),
                "sparse_mode": self.sparse_mode,
                "sparse_top_k": self.sparse_top_k,
                "low_rank_dim": self.low_rank_dim,
            }
        )
        return metadata


class TFTFrippClassifier(TFTFamilyClassifier):
    model_name = "tft_fripp"

    def __init__(
        self,
        *,
        variable_subset: list[str] | None = None,
        sparse_mode: str = "topk",
        sparse_top_k: int | None = None,
        low_rank_dim: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.variable_subset = list(variable_subset or [])
        self.sparse_mode = sparse_mode
        self.sparse_top_k = sparse_top_k
        self.low_rank_dim = low_rank_dim

    def build_tft_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
    ) -> BaseTFTNetwork:
        return TFTFrippNetwork(
            sequence_dim=sequence_dim,
            static_dim=static_dim,
            sequence_length=sequence_length,
            enabled_tasks=list(self.enabled_tasks),
            d_model=self.d_model,
            hidden_size=max(16, self.hidden_size // 2),
            attention_heads=max(1, self.attention_heads // 2),
            num_layers_local=1,
            dropout=self.dropout,
            readout_mode="robust_pool",
            use_static=self.use_static,
            n_regimes=self.n_regimes,
            variable_names=list(self.sequence_feature_names_),
            variable_subset=self.variable_subset or list(self.sequence_feature_names_),
            sparse_mode=self.sparse_mode,
            sparse_top_k=self.sparse_top_k,
            low_rank_dim=self.low_rank_dim,
        )
