"""Carey TFT variant: rhythm-aware and phase-sensitive."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .modules import (
    FourierTimeEmbedding,
    LocalCausalTCNEncoder,
    RhythmAwareReadout,
    TemporalPhaseGate,
    TimeFeatureProjector,
)
from .tft_base import BaseTFTNetwork, TFTFamilyClassifier


class TFTCareyNetwork(BaseTFTNetwork):
    def __init__(
        self,
        *,
        n_harmonics: int = 3,
        phase_hidden_size: int = 64,
        dilation_schedule: list[int] | None = None,
        kernel_size: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.fourier_time = FourierTimeEmbedding(n_harmonics=n_harmonics, d_model=self.d_model)
        self.static_time = TimeFeatureProjector(input_dim=max(self.static_dim, 1), d_model=self.d_model, dropout=self.dropout)
        self.local_rhythm = LocalCausalTCNEncoder(
            d_model=self.d_model,
            kernel_size=kernel_size,
            dilation_schedule=list(dilation_schedule or [1, 2, 4, 8]),
            dropout=self.dropout,
        )
        self.phase_gate = TemporalPhaseGate(d_model=self.d_model, hidden_dim=phase_hidden_size, dropout=self.dropout)
        self.rhythm_readout = RhythmAwareReadout(d_model=self.d_model)
        self.n_harmonics = n_harmonics
        self.phase_hidden_size = phase_hidden_size
        self.dilation_schedule = list(dilation_schedule or [1, 2, 4, 8])
        self.kernel_size = kernel_size

    def _phase_context(self, x_static: Tensor | None, batch_size: int, sequence_length: int, device: torch.device) -> Tensor:
        positions = torch.linspace(0.0, 1.0, steps=sequence_length, device=device).view(1, sequence_length, 1)
        positions = positions.expand(batch_size, -1, -1)
        phase_context = self.fourier_time(positions)
        if x_static is not None and x_static.numel() > 0:
            projected = self.static_time(x_static).unsqueeze(1)
            phase_context = phase_context + projected
        return phase_context

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x_seq = batch["x_seq"]
        seq_mask = batch.get("seq_mask")
        x_static = batch.get("x_static")
        static_context = self.build_static_context(x_static)

        z_seq, var_weights = self.embed_sequence(x_seq, batch, static_context)
        phase_context = self._phase_context(x_static, x_seq.shape[0], x_seq.shape[1], x_seq.device)
        z_local = self.local_rhythm(z_seq, seq_mask)
        z_phase = self.phase_gate(z_local, phase_context)
        z_attn, attn_weights = self.attend(z_phase, seq_mask)
        z_post = self.post_process(z_attn, static_context)
        z_final = self.rhythm_readout(z_post, seq_mask, phase_context=phase_context)

        outputs = self.heads(z_final)
        outputs["var_weights"] = var_weights
        outputs["attn_weights"] = attn_weights
        outputs["phase_context"] = phase_context
        return outputs

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "n_harmonics": self.n_harmonics,
                "phase_hidden_size": self.phase_hidden_size,
                "dilation_schedule": list(self.dilation_schedule),
                "kernel_size": self.kernel_size,
            }
        )
        return metadata


class TFTCareyClassifier(TFTFamilyClassifier):
    model_name = "tft_carey"

    def __init__(
        self,
        *,
        n_harmonics: int = 3,
        phase_hidden_size: int = 64,
        dilation_schedule: list[int] | None = None,
        kernel_size: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_harmonics = n_harmonics
        self.phase_hidden_size = phase_hidden_size
        self.dilation_schedule = list(dilation_schedule or [1, 2, 4, 8])
        self.kernel_size = kernel_size

    def build_tft_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
    ) -> BaseTFTNetwork:
        return TFTCareyNetwork(
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
            n_harmonics=self.n_harmonics,
            phase_hidden_size=self.phase_hidden_size,
            dilation_schedule=self.dilation_schedule,
            kernel_size=self.kernel_size,
        )
