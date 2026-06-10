"""Archived Stage 3 Carey: lightweight TCN plus temporal embeddings."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ...transformers.modules import FourierTimeEmbedding, GRN, LocalDilatedTCN, ReadoutBlock, RevIN, Time2Vec
from ...transformers.tft_base import BaseTFTNetwork, TFTFamilyClassifier


class CareyTCNFourierNetwork(BaseTFTNetwork):
    def __init__(
        self,
        *,
        n_harmonics: int = 3,
        time2vec_dim: int = 16,
        phase_hidden_size: int = 64,
        kernel_size: int = 3,
        dilation_schedule: list[int] | None = None,
        use_revin: bool = True,
        has_missingness_channels: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.has_missingness_channels = has_missingness_channels
        observed_dim = self.sequence_dim // 2 if self.has_missingness_channels and self.sequence_dim % 2 == 0 else self.sequence_dim
        self.revin = RevIN(observed_dim) if use_revin else None
        self.time2vec = Time2Vec(time2vec_dim)
        self.fourier = FourierTimeEmbedding(n_harmonics=n_harmonics, d_model=self.d_model)
        self.phase_projection = nn.Linear(self.d_model + time2vec_dim, self.d_model)
        self.local_encoder = LocalDilatedTCN(
            input_dim=self.d_model,
            hidden_dim=self.hidden_size,
            output_dim=self.d_model,
            kernel_size=kernel_size,
            dilation_schedule=list(dilation_schedule or [1, 2, 4, 8]),
            dropout=self.dropout,
        )
        self.phase_gate = GRN(self.d_model * 2, phase_hidden_size, d_out=self.d_model, dropout=self.dropout)
        self.readout = ReadoutBlock(d_model=self.d_model, mode="gated_pool")
        self.n_harmonics = n_harmonics
        self.time2vec_dim = time2vec_dim
        self.phase_hidden_size = phase_hidden_size
        self.kernel_size = kernel_size
        self.dilation_schedule = list(dilation_schedule or [1, 2, 4, 8])
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

    def _time_context(self, batch_size: int, sequence_length: int, device: torch.device) -> Tensor:
        positions = torch.linspace(0.0, 1.0, steps=sequence_length, device=device).view(1, sequence_length, 1)
        positions = positions.expand(batch_size, -1, -1)
        fourier = self.fourier(positions)
        time2vec = self.time2vec(positions)
        return self.phase_projection(torch.cat([fourier, time2vec], dim=-1))

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x_seq = self._maybe_normalize(batch["x_seq"])
        seq_mask = batch.get("seq_mask")
        x_static = batch.get("x_static")
        static_context = self.build_static_context(x_static)

        z_seq, var_weights = self.embed_sequence(x_seq, batch, static_context)
        phase_context = self._time_context(x_seq.shape[0], x_seq.shape[1], x_seq.device)
        local_input = z_seq + phase_context
        z_local = self.local_encoder(local_input, seq_mask)
        gated = self.phase_gate(torch.cat([z_local, phase_context], dim=-1))
        z_attn, attn_weights = self.attend(z_local + gated, seq_mask)
        z_post = self.post_process(z_attn, static_context)
        z_final = self.readout(z_post, seq_mask)

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
                "time2vec_dim": self.time2vec_dim,
                "phase_hidden_size": self.phase_hidden_size,
                "kernel_size": self.kernel_size,
                "dilation_schedule": list(self.dilation_schedule),
                "use_revin": self.use_revin,
            }
        )
        return metadata


class CareyTCNFourierClassifier(TFTFamilyClassifier):
    model_name = "carey_tcn_fourier"

    def __init__(
        self,
        *,
        n_harmonics: int = 3,
        time2vec_dim: int = 16,
        phase_hidden_size: int = 64,
        kernel_size: int = 3,
        dilation_schedule: list[int] | None = None,
        use_revin: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_harmonics = n_harmonics
        self.time2vec_dim = time2vec_dim
        self.phase_hidden_size = phase_hidden_size
        self.kernel_size = kernel_size
        self.dilation_schedule = list(dilation_schedule or [1, 2, 4, 8])
        self.use_revin = use_revin

    def build_tft_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
    ) -> BaseTFTNetwork:
        return CareyTCNFourierNetwork(
            sequence_dim=sequence_dim,
            static_dim=static_dim,
            sequence_length=sequence_length,
            enabled_tasks=list(self.enabled_tasks),
            d_model=self.d_model,
            hidden_size=self.hidden_size,
            attention_heads=self.attention_heads,
            num_layers_local=self.num_layers_local,
            dropout=self.dropout,
            readout_mode="gated_pool",
            use_static=self.use_static,
            n_regimes=self.n_regimes,
            n_harmonics=self.n_harmonics,
            time2vec_dim=self.time2vec_dim,
            phase_hidden_size=self.phase_hidden_size,
            kernel_size=self.kernel_size,
            dilation_schedule=self.dilation_schedule,
            use_revin=self.use_revin,
            has_missingness_channels=self.use_missingness_channels,
        )
