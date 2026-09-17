"""General-purpose PyTorch models inspired by D2L 1.0.3 examples."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Any

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 hidden_sizes: tuple[int, ...] = (256, 128), dropout: float = 0.1):
        super().__init__()
        dimensions = (input_size, *hidden_sizes, output_size)
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(pairwise(dimensions)):
            layers.append(nn.Linear(left, right))
            if index < len(dimensions) - 2:
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs.flatten(1))


class SequenceClassifier(nn.Module):
    """RNN, GRU or LSTM classifier for dense sequence features."""

    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128,
                 num_layers: int = 1, cell: str = "gru", bidirectional: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        recurrent = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
        if cell not in recurrent:
            raise ValueError(f"未知循环单元: {cell!r}；可选值: {', '.join(recurrent)}")
        self.encoder = recurrent[cell](input_size, hidden_size, num_layers=num_layers,
                                       batch_first=True, bidirectional=bidirectional,
                                       dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded, _ = self.encoder(inputs)
        return self.head(encoded[:, -1])


class TransformerClassifier(nn.Module):
    """Transformer encoder classifier with sinusoidal positional encoding."""

    def __init__(self, input_size: int, num_classes: int, d_model: int = 128,
                 n_heads: int = 4, layers: int = 2, feedforward: int = 256,
                 dropout: float = 0.1, max_length: int = 2048):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.projection = nn.Linear(input_size, d_model)
        self.register_buffer("position", self._positions(max_length, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, feedforward, dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, num_classes)

    @staticmethod
    def _positions(length: int, dimensions: int) -> Tensor:
        positions = torch.arange(length).float().unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, dimensions, 2).float()
                                * (-np_log(10000.0) / dimensions))
        encoding = torch.zeros(1, length, dimensions)
        encoding[0, :, 0::2] = torch.sin(positions * frequencies)
        encoding[0, :, 1::2] = torch.cos(positions * frequencies[:dimensions // 2])
        return encoding

    def forward(self, inputs: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if inputs.shape[1] > self.position.shape[1]:
            raise ValueError("序列长度超过 max_length")
        hidden = self.projection(inputs) + self.position[:, :inputs.shape[1]]
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            valid = (~padding_mask).unsqueeze(-1)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.head(pooled)


def np_log(value: float) -> float:
    """Small scalar helper that avoids importing NumPy in the PyTorch-only module."""
    import math

    return math.log(value)


def build_neural_model(name: str, params: Mapping[str, Any]) -> nn.Module:
    key = name.lower()
    if key == "mlp":
        return MLP(**dict(params))
    if key in {"rnn", "gru", "lstm"}:
        return SequenceClassifier(cell=key, **dict(params))
    if key == "transformer":
        return TransformerClassifier(**dict(params))
    raise ValueError("未知通用深度模型；可选值: mlp, rnn, gru, lstm, transformer")
