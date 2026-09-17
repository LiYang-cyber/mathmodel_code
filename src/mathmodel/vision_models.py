"""Classic PyTorch vision models adapted from D2L 1.0.3 architectures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class LeNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, padding=2), nn.Sigmoid(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.Sigmoid(), nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(120), nn.Sigmoid(), nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


class AlexNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, 11, stride=4, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 25, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


def _vgg_block(convolutions: int, in_channels: int, out_channels: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(convolutions):
        layers.extend([nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU()])
        in_channels = out_channels
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10,
                 architecture: tuple[tuple[int, int], ...] = ((1, 64), (1, 128), (2, 256),
                                                              (2, 512), (2, 512))):
        super().__init__()
        blocks = []
        current_channels = in_channels
        for convolutions, channels in architecture:
            blocks.append(_vgg_block(convolutions, current_channels, channels))
            current_channels = channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((7, 7)))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(current_channels * 49, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


def _nin_block(in_channels: int, out_channels: int, kernel_size: int, stride: int,
               padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 1), nn.ReLU(),
    )


class NiN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            _nin_block(in_channels, 96, 11, 4, 0), nn.MaxPool2d(3, 2),
            _nin_block(96, 256, 5, 1, 2), nn.MaxPool2d(3, 2),
            _nin_block(256, 384, 3, 1, 1), nn.MaxPool2d(3, 2), nn.Dropout(0.5),
            _nin_block(384, num_classes, 3, 1, 1), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class Inception(nn.Module):
    def __init__(self, in_channels: int, channels: tuple[int, int, int, int, int, int]):
        super().__init__()
        c1, c2_reduce, c2, c3_reduce, c3, c4 = channels
        self.path1 = nn.Conv2d(in_channels, c1, 1)
        self.path2 = nn.Sequential(nn.Conv2d(in_channels, c2_reduce, 1), nn.ReLU(),
                                   nn.Conv2d(c2_reduce, c2, 3, padding=1))
        self.path3 = nn.Sequential(nn.Conv2d(in_channels, c3_reduce, 1), nn.ReLU(),
                                   nn.Conv2d(c3_reduce, c3, 5, padding=2))
        self.path4 = nn.Sequential(nn.MaxPool2d(3, 1, padding=1), nn.Conv2d(in_channels, c4, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        paths = [self.path1(inputs), self.path2(inputs), self.path3(inputs), self.path4(inputs)]
        return torch.cat([torch.relu(path) for path in paths], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 64, 1), nn.ReLU(), nn.Conv2d(64, 192, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            Inception(192, (64, 96, 128, 16, 32, 32)),
            Inception(256, (128, 128, 192, 32, 96, 64)), nn.MaxPool2d(3, 2, 1),
            Inception(480, (192, 96, 208, 16, 48, 64)),
            Inception(512, (160, 112, 224, 24, 64, 64)),
            Inception(512, (128, 128, 256, 24, 64, 64)),
            Inception(512, (112, 144, 288, 32, 64, 64)),
            Inception(528, (256, 160, 320, 32, 128, 128)), nn.MaxPool2d(3, 2, 1),
            Inception(832, (256, 160, 320, 32, 128, 128)),
            Inception(832, (384, 192, 384, 48, 128, 128)),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(1024, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class Residual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.projection = (nn.Identity() if stride == 1 and in_channels == out_channels else
                           nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                         nn.BatchNorm2d(out_channels)))

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.relu(self.body(inputs) + self.projection(inputs))


class ResNet18(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        self.features = nn.Sequential(
            self._stage(64, 64, 2, 1), self._stage(64, 128, 2, 2),
            self._stage(128, 256, 2, 2), self._stage(256, 512, 2, 2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                  nn.Linear(512, num_classes))

    @staticmethod
    def _stage(in_channels: int, out_channels: int, blocks: int, stride: int):
        layers = [Residual(in_channels, out_channels, stride)]
        layers.extend(Residual(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.features(self.stem(inputs)))


class _DenseBlock(nn.Module):
    def __init__(self, layers: int, in_channels: int, growth_rate: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for index in range(layers):
            channels = in_channels + index * growth_rate
            self.layers.append(nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(),
                                              nn.Conv2d(channels, growth_rate, 3, padding=1,
                                                        bias=False)))

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for layer in self.layers:
            output = torch.cat([output, layer(output)], dim=1)
        return output


class DenseNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, growth_rate: int = 32,
                 block_layers: tuple[int, ...] = (6, 12, 24, 16)):
        super().__init__()
        channels = 64
        features: list[nn.Module] = [nn.Conv2d(in_channels, channels, 7, 2, 3, bias=False),
                                     nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(3, 2, 1)]
        for index, layers in enumerate(block_layers):
            features.append(_DenseBlock(layers, channels, growth_rate))
            channels += layers * growth_rate
            if index < len(block_layers) - 1:
                next_channels = channels // 2
                features.extend([nn.BatchNorm2d(channels), nn.ReLU(),
                                 nn.Conv2d(channels, next_channels, 1, bias=False),
                                 nn.AvgPool2d(2, 2)])
                channels = next_channels
        self.features = nn.Sequential(*features)
        self.head = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                  nn.Linear(channels, num_classes))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.features(inputs))


def build_vision_model(name: str, params: Mapping[str, Any] | None = None) -> nn.Module:
    registry: dict[str, type[nn.Module]] = {
        "alexnet": AlexNet,
        "densenet": DenseNet,
        "googlenet": GoogLeNet,
        "lenet": LeNet,
        "nin": NiN,
        "resnet18": ResNet18,
        "vgg": VGG,
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(f"未知视觉模型: {name!r}；可选值: {', '.join(registry)}")
    return registry[key](**dict(params or {}))
