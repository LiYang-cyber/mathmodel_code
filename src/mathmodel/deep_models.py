"""Portable PyTorch adaptations of four Mamba-family time-series models.

The implementations retain each model's main data flow while replacing custom CUDA
kernels with a differentiable PyTorch selective state-space scan.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelectiveStateSpace(nn.Module):
    """A portable input-selective state-space layer with a Mamba-style gate."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        super().__init__()
        if d_model < 1 or d_state < 1 or d_conv < 1 or expand < 1:
            raise ValueError("d_model、d_state、d_conv 和 expand 必须为正整数")
        inner = d_model * expand
        self.d_state = d_state
        self.inner = inner
        self.in_proj = nn.Linear(d_model, inner * 2)
        self.conv = nn.Conv1d(inner, inner, d_conv, padding=d_conv - 1, groups=inner)
        self.delta_proj = nn.Linear(inner, inner)
        self.bc_proj = nn.Linear(inner, d_state * 2)
        self.log_a = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(inner, 1))
        self.skip = nn.Parameter(torch.ones(inner))
        self.norm = nn.LayerNorm(inner)
        self.out_proj = nn.Linear(inner, d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("SelectiveStateSpace 输入形状必须为 [batch, length, channels]")
        values, gate = self.in_proj(inputs).chunk(2, dim=-1)
        length = values.shape[1]
        values = self.conv(values.transpose(1, 2))[..., :length].transpose(1, 2)
        values = F.silu(values)
        delta = F.softplus(self.delta_proj(values))
        b_coef, c_coef = self.bc_proj(values).chunk(2, dim=-1)
        decay_rate = torch.exp(self.log_a).to(dtype=values.dtype)
        state = values.new_zeros(values.shape[0], self.inner, self.d_state)
        outputs = []
        for step in range(length):
            dt = delta[:, step].unsqueeze(-1)
            decay = torch.exp(-dt * decay_rate.unsqueeze(0))
            state = decay * state + dt * values[:, step].unsqueeze(-1) * b_coef[:, step].unsqueeze(
                1
            )
            selected = (state * c_coef[:, step].unsqueeze(1)).sum(dim=-1)
            outputs.append(selected + self.skip * values[:, step])
        output = torch.stack(outputs, dim=1) * F.silu(gate)
        return self.out_proj(self.norm(output))


class SSMBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.forward_ssm = SelectiveStateSpace(d_model, d_state, d_conv, expand)
        self.backward_ssm = (
            SelectiveStateSpace(d_model, d_state, d_conv, expand) if bidirectional else None
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.forward_ssm(self.norm1(inputs))
        if self.backward_ssm is not None:
            backward = self.backward_ssm(self.norm1(inputs).flip(1)).flip(1)
            hidden = 0.5 * (hidden + backward)
        hidden = inputs + hidden
        return hidden + self.ffn(self.norm2(hidden))


def _normalise(inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    mean = inputs.mean(dim=1, keepdim=True).detach()
    scale = inputs.var(dim=1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    return (inputs - mean) / scale, mean, scale


class SMamba(nn.Module):
    """Bidirectional inter-variate Mamba adaptation for multivariate forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 128,
        d_state: int = 16,
        e_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embedding = nn.Linear(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [SSMBlock(d_model, d_state, 3, 1, dropout, bidirectional=True) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != self.seq_len:
            raise ValueError(f"S-Mamba 需要 [batch, {self.seq_len}, variables] 输入")
        values, mean, scale = _normalise(inputs)
        hidden = self.embedding(values.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        output = self.head(self.norm(hidden)).transpose(1, 2)
        return output * scale[:, :1] + mean[:, :1]


class TimePro(nn.Module):
    """Variable- and time-aware hyper-state model for long-term forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 96,
        d_state: int = 8,
        e_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len 不能大于 seq_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.patch_count = 1 + (seq_len - patch_len) // stride
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.variable_embedding = nn.Parameter(torch.empty(1, n_vars, 1, d_model))
        nn.init.normal_(self.variable_embedding, std=0.02)
        self.hyper_gate = nn.Sequential(nn.Linear(d_model + 2, d_model), nn.Sigmoid())
        self.blocks = nn.ModuleList(
            [SSMBlock(d_model, d_state, 5, 1, dropout, bidirectional=True) for _ in range(e_layers)]
        )
        self.head = nn.Linear(self.patch_count * d_model, pred_len)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (self.seq_len, self.n_vars):
            raise ValueError(f"TimePro 需要 [batch, {self.seq_len}, {self.n_vars}] 输入")
        values, mean, scale = _normalise(inputs)
        patches = values.transpose(1, 2).unfold(-1, self.patch_len, self.stride)
        hidden = self.patch_embedding(patches) + self.variable_embedding
        stats = torch.stack([patches.mean(-1), patches.std(-1, unbiased=False)], dim=-1)
        gate = self.hyper_gate(torch.cat([hidden, stats], dim=-1))
        hidden = hidden * gate
        batch, variables, patches_count, channels = hidden.shape
        hidden = hidden.reshape(batch * variables, patches_count, channels)
        for block in self.blocks:
            hidden = block(hidden)
        output = self.head(hidden.reshape(batch, variables, -1)).transpose(1, 2)
        return output * scale[:, :1] + mean[:, :1]


class _GraphExpert(nn.Module):
    def __init__(self, d_model: int, d_state: int, scales: tuple[int, ...], dropout: float):
        super().__init__()
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(d_model, d_model, scale, padding=scale - 1, groups=d_model)
                for scale in scales
            ]
        )
        self.mix = nn.Linear(d_model * len(scales), d_model)
        self.ssm = SSMBlock(d_model, d_state, 3, 1, dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        length = inputs.shape[1]
        channels_first = inputs.transpose(1, 2)
        multiscale = [conv(channels_first)[..., :length] for conv in self.convolutions]
        hidden = self.mix(torch.cat(multiscale, dim=1).transpose(1, 2))
        return self.ssm(hidden)


class STM3(nn.Module):
    """Multiscale temporal experts with adaptive graph fusion for spatio-temporal data."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        input_dim: int = 1,
        output_dim: int = 1,
        d_model: int = 64,
        node_emb_dim: int = 16,
        d_state: int = 8,
        num_experts: int = 3,
        scales: tuple[int, ...] = (1, 3, 5, 7),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.node_embedding)
        self.router = nn.Linear(node_emb_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_GraphExpert(d_model, d_state, scales, dropout) for _ in range(num_experts)]
        )
        self.graph_proj = nn.Linear(d_model, d_model)
        self.head = nn.Linear(seq_len * d_model, pred_len * output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        expected = (self.seq_len, self.num_nodes, self.input_dim)
        if inputs.ndim != 4 or inputs.shape[1:] != expected:
            raise ValueError(f"STM3 需要 [batch, {', '.join(map(str, expected))}] 输入")
        hidden = self.input_proj(inputs)
        batch, length, nodes, channels = hidden.shape
        flattened = hidden.permute(0, 2, 1, 3).reshape(batch * nodes, length, channels)
        expert_outputs = torch.stack([expert(flattened) for expert in self.experts], dim=2)
        weights = F.softmax(self.router(self.node_embedding), dim=-1)
        weights = weights.repeat(batch, 1).reshape(batch * nodes, 1, len(self.experts), 1)
        hidden = (expert_outputs * weights).sum(dim=2).reshape(batch, nodes, length, channels)
        adjacency = F.softmax(F.relu(self.node_embedding @ self.node_embedding.T), dim=-1)
        graph_hidden = torch.einsum("nm,bmld->bnld", adjacency, hidden)
        hidden = hidden + self.graph_proj(graph_hidden)
        output = self.head(hidden.reshape(batch, nodes, -1))
        return output.view(batch, nodes, self.pred_len, self.output_dim).permute(0, 2, 1, 3)


class MambaSL(nn.Module):
    """Single-layer Mamba classifier with multi-head adaptive temporal pooling."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 5,
        expand: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Conv1d(
            n_features, d_model, d_conv, padding="same", padding_mode="replicate", bias=False
        )
        self.mamba = SelectiveStateSpace(d_model, d_state, d_conv, expand)
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, n_heads)
        self.classifier = nn.Linear(d_model, n_classes, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("MambaSL 输入形状必须为 [batch, length, features]")
        hidden = self.embedding(inputs.transpose(1, 2)).transpose(1, 2)
        hidden = F.silu(self.norm(self.mamba(hidden)))
        scores = self.score(hidden)
        if mask is not None:
            if mask.shape != inputs.shape[:2]:
                raise ValueError("mask 形状必须为 [batch, length]")
            scores = scores.masked_fill(~mask.bool().unsqueeze(-1), torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1).mean(dim=-1, keepdim=True)
        pooled = (hidden * weights).sum(dim=1)
        return self.classifier(self.dropout(pooled))


class MambaForecaster(nn.Module):
    """Direct Mamba-style multivariate sequence forecaster."""

    def __init__(self, seq_len: int, pred_len: int, n_vars: int, d_model: int = 64,
                 d_state: int = 16, e_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.seq_len, self.pred_len, self.n_vars = seq_len, pred_len, n_vars
        self.embedding = nn.Linear(n_vars, d_model)
        self.blocks = nn.ModuleList([
            SSMBlock(d_model, d_state, 3, 2, dropout) for _ in range(e_layers)
        ])
        self.head = nn.Linear(seq_len * d_model, pred_len * n_vars)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (self.seq_len, self.n_vars):
            raise ValueError(f"Mamba 需要 [batch, {self.seq_len}, {self.n_vars}] 输入")
        values, mean, scale = _normalise(inputs)
        hidden = self.embedding(values)
        for block in self.blocks:
            hidden = block(hidden)
        output = self.head(hidden.flatten(1)).view(-1, self.pred_len, self.n_vars)
        return output * scale[:, :1] + mean[:, :1]


class LSTMForecaster(nn.Module):
    """Multivariate LSTM baseline using the same input/output contract as Mamba."""

    def __init__(self, seq_len: int, pred_len: int, n_vars: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.seq_len, self.pred_len, self.n_vars = seq_len, pred_len, n_vars
        self.lstm = nn.LSTM(n_vars, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, pred_len * n_vars)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (self.seq_len, self.n_vars):
            raise ValueError(f"LSTM 需要 [batch, {self.seq_len}, {self.n_vars}] 输入")
        values, mean, scale = _normalise(inputs)
        hidden, _ = self.lstm(values)
        output = self.head(hidden[:, -1]).view(-1, self.pred_len, self.n_vars)
        return output * scale[:, :1] + mean[:, :1]


def build_deep_timeseries_model(name: str, params: Mapping[str, Any]) -> nn.Module:
    """Construct a registered model from a name and keyword parameters."""
    registry: dict[str, type[nn.Module]] = {
        "mamba": MambaForecaster,
        "s_mamba": SMamba,
        "timepro": TimePro,
        "stm3": STM3,
        "mambasl": MambaSL,
        "lstm": LSTMForecaster,
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(f"未知深度时序模型: {name!r}；可选值: {', '.join(registry)}")
    return registry[key](**dict(params))
