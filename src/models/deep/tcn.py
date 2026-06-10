"""Temporal convolutional classifier."""

from __future__ import annotations

import torch
from torch import nn

from .common import TorchSequenceClassifier


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(self.net(x) + residual)


class _TCNNetwork(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        static_dim: int,
        num_classes: int,
        *,
        channels: list[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = sequence_dim
        for level, out_channels in enumerate(channels):
            blocks.append(
                _TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*blocks)
        self.static_projection = (
            nn.Sequential(
                nn.Linear(static_dim, channels[-1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if static_dim
            else None
        )
        head_input = channels[-1] + (channels[-1] if static_dim else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        encoded = self.network(sequence.transpose(1, 2))[:, :, -1]
        if self.static_projection is not None:
            encoded = torch.cat([encoded, self.static_projection(static)], dim=1)
        return self.head(encoded)


class TCNClassifier(TorchSequenceClassifier):
    model_name = "tcn"

    def __init__(
        self,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.channels = channels or [64, 64, 64]
        self.kernel_size = kernel_size
        self.dropout = dropout

    def build_network(
        self,
        sequence_dim: int,
        static_dim: int,
        sequence_length: int,
        num_classes: int,
    ) -> nn.Module:
        return _TCNNetwork(
            sequence_dim=sequence_dim,
            static_dim=static_dim,
            num_classes=num_classes,
            channels=self.channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

