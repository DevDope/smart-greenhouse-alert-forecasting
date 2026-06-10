"""Experimental Stage 3 Fripp: PatchTST-lite austere transformer."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ...transformers.modules import (
    CausalInterpretableAttention,
    GRN,
    MultiTaskHeads,
    PatchEmbedding1D,
    ReadoutBlock,
    VariableSubsetController,
)
from ...transformers.tft_base import TFTFamilyClassifier


class FrippPatchTSTLiteNetwork(nn.Module):
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
        dropout: float,
        variable_names: list[str],
        variable_subset: list[str] | None = None,
        patch_length: int = 8,
        patch_stride: int = 4,
        channel_independent: bool = True,
        n_regimes: int = 3,
    ) -> None:
        super().__init__()
        self.variable_controller = VariableSubsetController(variable_names=variable_names, subset=variable_subset)
        self.patch_embed = PatchEmbedding1D(
            input_dim=len(self.variable_controller.subset),
            d_model=d_model,
            patch_length=patch_length,
            stride=patch_stride,
            channel_independent=channel_independent,
        )
        max_patches = max(1, ((max(sequence_length, patch_length) - patch_length) // patch_stride) + 1)
        self.position = nn.Parameter(torch.zeros(1, max_patches, d_model))
        self.attention = CausalInterpretableAttention(d_model=d_model, n_heads=attention_heads, dropout=dropout)
        self.post = GRN(d_model, hidden_size, d_out=d_model, dropout=dropout)
        self.readout = ReadoutBlock(d_model=d_model, mode="robust_pool")
        self.static_projection = (
            nn.Sequential(
                nn.LayerNorm(static_dim),
                nn.Linear(static_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if static_dim
            else None
        )
        head_input_dim = d_model + (d_model if static_dim else 0)
        self.heads = MultiTaskHeads(d_model=head_input_dim, n_regimes=n_regimes, enabled_tasks=enabled_tasks)
        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.channel_independent = channel_independent
        self.enabled_tasks = list(enabled_tasks)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x_seq = self.variable_controller(batch["x_seq"])
        x_static = batch.get("x_static")
        tokens, channel_tokens = self.patch_embed(x_seq)
        tokens = tokens + self.position[:, : tokens.shape[1], :]
        encoded, attn_weights = self.attention(tokens, None)
        encoded = self.post(encoded)
        pooled = self.readout(encoded, None)
        if self.static_projection is not None and x_static is not None and x_static.numel() > 0:
            pooled = torch.cat([pooled, self.static_projection(x_static)], dim=-1)

        outputs = self.heads(pooled)
        outputs["attn_weights"] = attn_weights
        outputs["var_weights"] = channel_tokens.abs().mean(dim=-1).transpose(1, 2)
        return outputs

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled_tasks": list(self.enabled_tasks),
            "architecture_class": self.__class__.__name__,
            **self.variable_controller.metadata(),
            "patch_length": self.patch_length,
            "patch_stride": self.patch_stride,
            "channel_independent": self.channel_independent,
        }


class FrippPatchTSTLiteClassifier(TFTFamilyClassifier):
    model_name = "fripp_patchtst_lite"

    def __init__(
        self,
        *,
        variable_subset: list[str] | None = None,
        patch_length: int = 8,
        patch_stride: int = 4,
        channel_independent: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.variable_subset = list(variable_subset or [])
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.channel_independent = channel_independent

    def _resolved_subset(self) -> list[str]:
        if not self.variable_subset:
            return list(self.sequence_feature_names_)
        if not self.use_missingness_channels:
            return list(self.variable_subset)
        expanded = list(self.variable_subset)
        expanded.extend(f"{name}__missing" for name in self.variable_subset)
        return [name for name in self.sequence_feature_names_ if name in set(expanded)]

    def build_tft_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
    ) -> nn.Module:
        return FrippPatchTSTLiteNetwork(
            sequence_dim=sequence_dim,
            static_dim=static_dim,
            sequence_length=sequence_length,
            enabled_tasks=list(self.enabled_tasks),
            d_model=self.d_model,
            hidden_size=max(16, self.hidden_size // 2),
            attention_heads=max(1, self.attention_heads // 2),
            dropout=self.dropout,
            variable_names=list(self.sequence_feature_names_),
            variable_subset=self._resolved_subset(),
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            channel_independent=self.channel_independent,
            n_regimes=self.n_regimes,
        )
