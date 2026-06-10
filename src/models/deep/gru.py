"""GRU sequence classifier."""

from __future__ import annotations

import torch
from torch import nn

from .common import TorchSequenceClassifier


class _GRUNetwork(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        static_dim: int,
        num_classes: int,
        *,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=sequence_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.static_projection = (
            nn.Sequential(
                nn.Linear(static_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if static_dim
            else None
        )
        head_input = hidden_size + (hidden_size if static_dim else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(sequence)
        encoded = hidden[-1]
        if self.static_projection is not None:
            encoded = torch.cat([encoded, self.static_projection(static)], dim=1)
        return self.head(encoded)


class GRUClassifier(TorchSequenceClassifier):
    model_name = "gru"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def build_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
        num_classes: int,
    ) -> nn.Module:
        return _GRUNetwork(
            sequence_dim=sequence_dim,
            static_dim=static_dim,
            num_classes=num_classes,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

