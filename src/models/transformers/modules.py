"""Shared TFT-style modules for Stage 3 variants."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _ensure_mask(mask: Tensor | None, x: Tensor) -> Tensor:
    if mask is None:
        return torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
    return mask.to(dtype=torch.bool, device=x.device)


def _masked_mean(x: Tensor, mask: Tensor | None) -> Tensor:
    valid_mask = _ensure_mask(mask, x).unsqueeze(-1).to(dtype=x.dtype)
    denom = valid_mask.sum(dim=1).clamp_min(1.0)
    return (x * valid_mask).sum(dim=1) / denom


def _last_valid(x: Tensor, mask: Tensor | None) -> Tensor:
    valid_mask = _ensure_mask(mask, x)
    lengths = valid_mask.sum(dim=1).clamp_min(1) - 1
    return x[torch.arange(x.shape[0], device=x.device), lengths]


def _topk_softmax(logits: Tensor, top_k: int) -> Tensor:
    if top_k >= logits.shape[-1]:
        return torch.softmax(logits, dim=-1)
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(-1, indices, values)
    return torch.softmax(masked, dim=-1)


def sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits, _ = torch.sort(shifted, descending=True, dim=dim)
    cumsum = sorted_logits.cumsum(dim) - 1.0
    range_values = torch.arange(
        1,
        sorted_logits.shape[dim] + 1,
        device=logits.device,
        dtype=logits.dtype,
    )
    view_shape = [1] * logits.dim()
    view_shape[dim] = -1
    range_values = range_values.view(view_shape)
    support = sorted_logits > cumsum / range_values
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = cumsum.gather(dim, support_size.long() - 1) / support_size.to(dtype=logits.dtype)
    return torch.clamp(shifted - tau, min=0.0)


def entmax15(logits: Tensor, dim: int = -1, n_iter: int = 25) -> Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    tau_lo = shifted.min(dim=dim, keepdim=True).values - 2.0
    tau_hi = shifted.max(dim=dim, keepdim=True).values
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2.0
        probs = torch.clamp(0.5 * (shifted - tau), min=0.0) ** 2
        normalizer = probs.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(normalizer > 1.0, tau, tau_lo)
        tau_hi = torch.where(normalizer <= 1.0, tau, tau_hi)
    probs = torch.clamp(0.5 * (shifted - tau_hi), min=0.0) ** 2
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class ContinuousVariableEmbedding(nn.Module):
    """Projects each continuous variable independently into model space."""

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_vars, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_vars, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq: Tensor) -> Tensor:
        embedded = x_seq.unsqueeze(-1) * self.weight.unsqueeze(0).unsqueeze(0) + self.bias.unsqueeze(0).unsqueeze(0)
        return self.dropout(embedded)


class StaticContextEncoder(nn.Module):
    """Encodes optional static features safely."""

    def __init__(self, static_dim: int, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(static_dim),
            nn.Linear(static_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_static: Tensor | None) -> Tensor | None:
        if x_static is None or x_static.numel() == 0:
            return None
        return self.norm(self.network(x_static))


class GRN(nn.Module):
    """Core gated residual network block."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int | None = None,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        d_out = d_out or d_in
        self.linear_in = nn.Linear(d_in, d_hidden)
        self.context_projection = nn.Linear(context_dim, d_hidden) if context_dim is not None else None
        self.hidden = nn.Linear(d_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_out, d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        hidden = self.linear_in(x)
        if context is not None and self.context_projection is not None:
            hidden = hidden + self.context_projection(context)
        hidden = torch.nn.functional.elu(hidden)
        hidden = self.dropout(torch.nn.functional.elu(self.hidden(hidden)))
        candidate = self.out(hidden)
        gated = torch.sigmoid(self.gate(candidate)) * candidate
        return self.norm(self.skip(x) + gated)


class VariableSelectionNetwork(nn.Module):
    """Per-variable transformation plus interpretable weighting."""

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
        use_static_context: bool = True,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        context_dim = d_model if use_static_context else None
        self.variable_grns = nn.ModuleList(
            [GRN(d_model, hidden_dim, d_out=d_model, dropout=dropout) for _ in range(n_vars)]
        )
        self.weight_grn = GRN(
            d_in=n_vars * d_model,
            d_hidden=hidden_dim,
            d_out=n_vars,
            dropout=dropout,
            context_dim=context_dim,
        )

    def _weight_activation(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=-1)

    def forward(self, x_emb: Tensor, static_context: Tensor | None = None) -> tuple[Tensor, Tensor]:
        transformed = torch.stack(
            [module(x_emb[:, :, index, :]) for index, module in enumerate(self.variable_grns)],
            dim=2,
        )
        flattened = x_emb.reshape(x_emb.shape[0], x_emb.shape[1], -1)
        if static_context is not None:
            context = static_context.unsqueeze(1).expand(-1, x_emb.shape[1], -1)
            logits = self.weight_grn(flattened, context=context)
        else:
            logits = self.weight_grn(flattened)
        weights = self._weight_activation(logits)
        selected = (weights.unsqueeze(-1) * transformed).sum(dim=2)
        return selected, weights


class SparseVariableSelectionNetwork(VariableSelectionNetwork):
    """Stronger sparsity over variables."""

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
        use_static_context: bool = True,
        sparse_mode: str = "topk",
        top_k: int | None = None,
    ) -> None:
        super().__init__(n_vars, d_model, hidden_dim, dropout, use_static_context=use_static_context)
        self.sparse_mode = sparse_mode
        self.top_k = top_k or max(1, n_vars // 2)

    def _weight_activation(self, logits: Tensor) -> Tensor:
        if self.sparse_mode == "softmax":
            return torch.softmax(logits, dim=-1)
        if self.sparse_mode == "sparsemax":
            return sparsemax(logits, dim=-1)
        if self.sparse_mode == "entmax15":
            return entmax15(logits, dim=-1)
        if self.sparse_mode == "topk":
            return _topk_softmax(logits, self.top_k)
        if self.sparse_mode == "sparse_sigmoid":
            weights = torch.sigmoid(logits)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return weights
        raise ValueError(f"Unsupported sparse selection mode: {self.sparse_mode}")


class SparseGate(nn.Module):
    """Applies a selectable sparse activation over routing logits."""

    def __init__(self, mode: str = "softmax", top_k: int | None = None) -> None:
        super().__init__()
        self.mode = mode
        self.top_k = top_k

    def forward(self, logits: Tensor) -> Tensor:
        if self.mode == "softmax":
            return torch.softmax(logits, dim=-1)
        if self.mode == "sparsemax":
            return sparsemax(logits, dim=-1)
        if self.mode == "entmax15":
            return entmax15(logits, dim=-1)
        if self.mode == "topk":
            return _topk_softmax(logits, self.top_k or max(1, logits.shape[-1] // 2))
        raise ValueError(f"Unsupported sparse gate mode: {self.mode}")


class RevIN(nn.Module):
    """Reversible instance normalization over the temporal axis."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(
        self,
        x: Tensor,
        mode: str = "norm",
        stats: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        if mode == "norm":
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
            normalized = (x - mean) / std
            if self.affine and self.weight is not None and self.bias is not None:
                normalized = normalized * self.weight + self.bias
            return normalized, (mean, std)
        if mode == "denorm":
            if stats is None:
                raise ValueError("RevIN denorm requires cached statistics.")
            mean, std = stats
            restored = x
            if self.affine and self.weight is not None and self.bias is not None:
                restored = (restored - self.bias) / self.weight.clamp_min(self.eps)
            return restored * std + mean
        raise ValueError(f"Unsupported RevIN mode: {mode}")


class LocalLSTMEncoder(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_size, d_model) if hidden_size != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        valid_mask = _ensure_mask(mask, x)
        lengths = valid_mask.sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        projected = self.projection(encoded)
        return self.norm(projected)


class LocalGRUEncoder(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_size, d_model) if hidden_size != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        valid_mask = _ensure_mask(mask, x)
        lengths = valid_mask.sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
        projected = self.projection(encoded)
        return self.norm(projected)


class _CausalConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _causal_crop(self, x: Tensor) -> Tensor:
        return x[:, :, :-self.padding] if self.padding else x

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        y = self._causal_crop(self.conv1(x.transpose(1, 2))).transpose(1, 2)
        y = self.dropout(torch.nn.functional.gelu(y))
        y = self._causal_crop(self.conv2(y.transpose(1, 2))).transpose(1, 2)
        y = self.dropout(torch.nn.functional.gelu(y))
        return self.norm(residual + y)


class LocalCausalTCNEncoder(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dilation_schedule: list[int], dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_CausalConvBlock(d_model=d_model, kernel_size=kernel_size, dilation=dilation, dropout=dropout) for dilation in dilation_schedule]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        valid_mask = _ensure_mask(mask, x).unsqueeze(-1).to(dtype=x.dtype)
        hidden = x * valid_mask
        for block in self.blocks:
            hidden = block(hidden) * valid_mask
        return hidden


class LocalDilatedTCN(nn.Module):
    """Causal dilated temporal block with optional input/output projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        kernel_size: int,
        dilation_schedule: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.encoder = LocalCausalTCNEncoder(
            d_model=hidden_dim,
            kernel_size=kernel_size,
            dilation_schedule=dilation_schedule,
            dropout=dropout,
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        hidden = self.input_projection(x)
        hidden = self.encoder(hidden, mask)
        return self.norm(self.output_projection(hidden))


class CausalInterpretableAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, seq_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch_size, time_steps, d_model = x.shape
        valid_mask = _ensure_mask(seq_mask, x)
        q = self.q_proj(x).view(batch_size, time_steps, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, time_steps, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, time_steps, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(time_steps, time_steps, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
        key_mask = (~valid_mask).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(key_mask, -1e4)

        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        context = torch.matmul(self.dropout(weights), v)
        context = context.transpose(1, 2).reshape(batch_size, time_steps, d_model)
        context = self.out_proj(context)
        context = context * valid_mask.unsqueeze(-1).to(dtype=context.dtype)
        return self.norm(x + self.dropout(context)), weights


class MMoEBlock(nn.Module):
    """Mixture-of-experts block with one gate per task."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_experts: int,
        task_names: list[str],
        dropout: float,
        gate_mode: str = "softmax",
        gate_top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.task_names = list(task_names)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.gates = nn.ModuleDict({task_name: nn.Linear(input_dim, num_experts) for task_name in self.task_names})
        self.activations = nn.ModuleDict(
            {task_name: SparseGate(mode=gate_mode, top_k=gate_top_k) for task_name in self.task_names}
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        experts = torch.stack([expert(x) for expert in self.experts], dim=1)
        outputs: dict[str, Tensor] = {}
        weights: dict[str, Tensor] = {}
        for task_name in self.task_names:
            logits = self.gates[task_name](x)
            task_weights = self.activations[task_name](logits)
            weights[task_name] = task_weights
            task_output = torch.einsum("be,bed->bd", task_weights, experts)
            outputs[task_name] = self.output_norm(task_output)
        return outputs, weights


class LowRankAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rank_dim: int, dropout: float) -> None:
        super().__init__()
        self.pre = nn.Linear(d_model, rank_dim)
        self.attention = CausalInterpretableAttention(rank_dim, n_heads=n_heads, dropout=dropout)
        self.post = nn.Linear(rank_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, seq_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        projected = self.pre(x)
        attended, weights = self.attention(projected, seq_mask)
        return self.norm(x + self.post(attended)), weights


class Time2Vec(nn.Module):
    """Time2Vec encoding for scalar temporal positions."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        if out_dim < 2:
            raise ValueError("Time2Vec requires out_dim >= 2.")
        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))
        self.periodic_weight = nn.Parameter(torch.randn(out_dim - 1))
        self.periodic_bias = nn.Parameter(torch.zeros(out_dim - 1))

    def forward(self, t: Tensor) -> Tensor:
        linear = self.linear_weight * t + self.linear_bias
        periodic = torch.sin(t * self.periodic_weight.view(1, 1, -1) + self.periodic_bias.view(1, 1, -1))
        return torch.cat([linear, periodic], dim=-1)


class ReadoutBlock(nn.Module):
    def __init__(self, d_model: int, mode: str = "last_valid") -> None:
        super().__init__()
        self.mode = mode
        self.score = nn.Linear(d_model, 1)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, z: Tensor, seq_mask: Tensor | None = None) -> Tensor:
        valid_mask = _ensure_mask(seq_mask, z)
        if self.mode == "last_valid":
            return _last_valid(z, valid_mask)
        if self.mode == "attn_pool":
            scores = self.score(z).squeeze(-1).masked_fill(~valid_mask, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (weights * z).sum(dim=1)
        if self.mode == "gated_pool":
            gates = torch.sigmoid(self.gate(z))
            pooled = _masked_mean(gates * z, valid_mask)
            return pooled
        if self.mode == "robust_pool":
            pooled = _masked_mean(z, valid_mask)
            tail = _last_valid(z, valid_mask)
            return 0.5 * pooled + 0.5 * tail
        raise ValueError(f"Unsupported readout mode: {self.mode}")


class RhythmAwareReadout(ReadoutBlock):
    def __init__(self, d_model: int) -> None:
        super().__init__(d_model=d_model, mode="attn_pool")
        self.phase_projection = nn.Linear(d_model, 1)

    def forward(self, z: Tensor, seq_mask: Tensor | None = None, phase_context: Tensor | None = None) -> Tensor:
        valid_mask = _ensure_mask(seq_mask, z)
        scores = self.score(z).squeeze(-1)
        if phase_context is not None:
            scores = scores + self.phase_projection(phase_context).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (weights * z).sum(dim=1)


class MultiTaskHeads(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_regimes: int | None = None,
        enabled_tasks: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.enabled_tasks = enabled_tasks or ["event_60m", "event_120m"]
        self.heads = nn.ModuleDict()
        for task_name in self.enabled_tasks:
            output_dim = n_regimes if task_name == "regime_future" else 1
            self.heads[task_name] = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, output_dim),
            )

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        return {task_name: head(z) for task_name, head in self.heads.items()}


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        task_weights: dict[str, float],
        class_weights: dict[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.task_weights = task_weights
        self.class_weights = class_weights or {}

    def forward(self, outputs: dict[str, Tensor], targets: dict[str, dict[str, Tensor]]) -> tuple[Tensor, dict[str, float]]:
        total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        details: dict[str, float] = {}
        for task_name, target_pack in targets.items():
            if task_name not in outputs:
                continue
            values = target_pack["values"]
            mask = target_pack["mask"]
            if mask.sum() == 0:
                continue
            logits = outputs[task_name][mask]
            target_values = values[mask]
            if logits.shape[-1] == 1:
                pos_weight = self.class_weights.get(task_name)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1),
                    target_values.float(),
                    pos_weight=pos_weight.to(logits.device) if pos_weight is not None else None,
                )
            else:
                weight = self.class_weights.get(task_name)
                loss = nn.functional.cross_entropy(
                    logits,
                    target_values.long(),
                    weight=weight.to(logits.device) if weight is not None else None,
                )
            weighted = self.task_weights.get(task_name, 1.0) * loss
            total_loss = total_loss + weighted
            details[f"loss_{task_name}"] = float(loss.detach().cpu().item())
        return total_loss, details


class TrendResidualDecomposer(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        padded = nn.functional.pad(x.transpose(1, 2), (self.kernel_size - 1, 0), mode="replicate")
        trend = nn.functional.avg_pool1d(
            padded,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            count_include_pad=False,
        ).transpose(1, 2)
        residual = x - trend
        return trend, residual


class MultiScaleSequenceBuilder(nn.Module):
    def __init__(self, pool_kernel_sizes: list[int]) -> None:
        super().__init__()
        self.pool_kernel_sizes = pool_kernel_sizes

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        scales = {"fine": x}
        for index, kernel in enumerate(self.pool_kernel_sizes, start=1):
            padded = nn.functional.pad(x.transpose(1, 2), (kernel - 1, 0), mode="replicate")
            pooled = nn.functional.avg_pool1d(
                padded,
                kernel_size=kernel,
                stride=1,
                padding=0,
                count_include_pad=False,
            ).transpose(1, 2)
            scales["mid" if index == 1 else "coarse"] = pooled
        return scales


class LongContextSummaryBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.summary_grn = GRN(d_model, hidden_dim, d_out=d_model, dropout=dropout)

    def forward(self, streams: dict[str, Tensor], seq_mask: Tensor | None = None) -> dict[str, Tensor]:
        return {name: self.summary_grn(_masked_mean(stream, seq_mask)) for name, stream in streams.items()}


class LongContextFusionBlock(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fusion = GRN(d_model * 2, hidden_dim, d_out=d_model, dropout=dropout)

    def forward(self, local_state: Tensor, summaries: dict[str, Tensor]) -> Tensor:
        stacked = torch.stack(list(summaries.values()), dim=1).mean(dim=1)
        fused = torch.cat([local_state, stacked], dim=-1)
        return self.fusion(fused)


class TaskTokenCrossAttention(nn.Module):
    def __init__(self, d_model: int, task_names: list[str], dropout: float) -> None:
        super().__init__()
        self.task_names = task_names
        self.tokens = nn.ParameterDict({task_name: nn.Parameter(torch.randn(1, 1, d_model) * 0.02) for task_name in task_names})
        self.attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True, dropout=dropout)

    def forward(self, sequence_repr: Tensor, seq_mask: Tensor | None = None) -> dict[str, Tensor]:
        key_padding_mask = ~_ensure_mask(seq_mask, sequence_repr)
        outputs: dict[str, Tensor] = {}
        for task_name in self.task_names:
            token = self.tokens[task_name].expand(sequence_repr.shape[0], -1, -1)
            attended, _ = self.attention(token, sequence_repr, sequence_repr, key_padding_mask=key_padding_mask, need_weights=False)
            outputs[task_name] = attended.squeeze(1)
        return outputs


class TaskSpecificAdapter(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.adapter = GRN(d_model, hidden_dim, d_out=d_model, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.adapter(x)


class TaskRoutingGate(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, shared_repr: Tensor, task_repr: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([shared_repr, task_repr], dim=-1)))
        return gate * task_repr + (1.0 - gate) * shared_repr


class FourierTimeEmbedding(nn.Module):
    def __init__(self, n_harmonics: int, d_model: int) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics
        self.projection = nn.Linear(n_harmonics * 4, d_model)

    def forward(self, positions: Tensor) -> Tensor:
        features: list[Tensor] = []
        for harmonic in range(1, self.n_harmonics + 1):
            angle = 2 * math.pi * harmonic * positions
            features.extend([torch.sin(angle), torch.cos(angle), torch.sin(angle / 7.0), torch.cos(angle / 7.0)])
        stacked = torch.cat(features, dim=-1)
        return self.projection(stacked)


class PatchEmbedding1D(nn.Module):
    """PatchTST-style temporal patch embedding with optional channel independence."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        patch_length: int,
        stride: int,
        channel_independent: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.patch_length = patch_length
        self.stride = stride
        self.channel_independent = channel_independent
        projection_in = patch_length if channel_independent else patch_length * input_dim
        self.projection = nn.Linear(projection_in, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _pad(self, x: Tensor) -> Tensor:
        if x.shape[1] >= self.patch_length:
            remainder = (x.shape[1] - self.patch_length) % self.stride
            if remainder == 0:
                return x
            pad_needed = self.stride - remainder
        else:
            pad_needed = self.patch_length - x.shape[1]
        padding = x[:, -1:, :].expand(-1, pad_needed, -1)
        return torch.cat([x, padding], dim=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self._pad(x)
        if self.channel_independent:
            patches = x.transpose(1, 2).unfold(-1, self.patch_length, self.stride)
            patch_tokens = self.projection(patches)
            channel_tokens = self.norm(patch_tokens)
            return channel_tokens.mean(dim=1), channel_tokens
        patches = x.unfold(1, self.patch_length, self.stride).reshape(x.shape[0], -1, self.patch_length * self.input_dim)
        tokens = self.norm(self.projection(patches))
        return tokens, tokens.unsqueeze(1)


class TimeFeatureProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class TemporalPhaseGate(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate = GRN(d_model * 2, hidden_dim, d_out=d_model, dropout=dropout)

    def forward(self, sequence_repr: Tensor, phase_repr: Tensor) -> Tensor:
        gate_input = torch.cat([sequence_repr, phase_repr], dim=-1)
        gated = torch.sigmoid(self.gate(gate_input))
        return sequence_repr * (1.0 + gated)


class VariableSubsetController(nn.Module):
    def __init__(self, variable_names: list[str], subset: list[str] | None = None) -> None:
        super().__init__()
        self.variable_names = list(variable_names)
        self.subset = list(subset or variable_names)
        self.indices = [index for index, name in enumerate(variable_names) if name in self.subset]
        if not self.indices:
            self.indices = list(range(len(variable_names)))
            self.subset = list(variable_names)

    def forward(self, x: Tensor) -> Tensor:
        index = torch.tensor(self.indices, dtype=torch.long, device=x.device)
        return x.index_select(-1, index)

    def metadata(self) -> dict[str, Any]:
        return {
            "variable_subset": list(self.subset),
            "variable_subset_size": len(self.subset),
            "variable_subset_indices": list(self.indices),
        }
