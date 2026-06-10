"""Vanilla transformer encoder classifier."""

from __future__ import annotations

import torch
from torch import nn

from ..deep.common import TorchSequenceClassifier


class _VanillaTransformerNetwork(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
        num_classes: int,
        *,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(sequence_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, sequence_length + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.static_projection = (
            nn.Sequential(
                nn.Linear(static_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if static_dim
            else None
        )
        head_input = d_model + (d_model if static_dim else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        batch_size = sequence.shape[0]
        encoded = self.input_projection(sequence)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        encoded = torch.cat([cls_token, encoded], dim=1)
        encoded = self.dropout(encoded + self.position[:, : encoded.shape[1], :])
        encoded = self.encoder(encoded)
        pooled = encoded[:, 0, :]
        if self.static_projection is not None:
            pooled = torch.cat([pooled, self.static_projection(static)], dim=1)
        return self.head(pooled)


class VanillaTransformerClassifier(TorchSequenceClassifier):
    model_name = "vanilla_transformer"

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

    def build_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
        num_classes: int,
    ) -> nn.Module:
        return _VanillaTransformerNetwork(
            sequence_dim=sequence_dim,
            static_dim=static_dim,
            sequence_length=sequence_length,
            num_classes=num_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
