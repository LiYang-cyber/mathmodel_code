"""Reusable plotting helpers adapted from D2L 1.0.3 interfaces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def set_axes(
    axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim=None,
    ylim=None,
    xscale: str = "linear",
    yscale: str = "linear",
    legend: Sequence[str] | None = None,
    grid: bool = True,
):
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid(grid)
    return axes


def plot(
    x,
    y=None,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: Sequence[str] | None = None,
    formats: Sequence[str] = ("-", "m--", "g-.", "r:"),
    figsize: tuple[float, float] = (5.5, 3.5),
    axes=None,
    **axes_options,
):
    """Plot one or more series and return ``(figure, axes)``."""
    if axes is None:
        figure, axes = plt.subplots(figsize=figsize)
    else:
        figure = axes.figure
    x_values = _series_list(x)
    if y is None:
        y_values = x_values
        x_values = [None] * len(y_values)
    else:
        y_values = _series_list(y)
        if len(x_values) == 1 and len(y_values) > 1:
            x_values *= len(y_values)
        if len(x_values) != len(y_values):
            raise ValueError("x 与 y 的序列数量不一致")
    for index, (x_item, y_item) in enumerate(zip(x_values, y_values, strict=True)):
        style = formats[index % len(formats)]
        axes.plot(y_item, style) if x_item is None else axes.plot(x_item, y_item, style)
    set_axes(axes, xlabel=xlabel, ylabel=ylabel, legend=legend, **axes_options)
    return figure, axes


def show_images(
    images: Sequence,
    rows: int,
    columns: int,
    *,
    titles: Sequence[str] | None = None,
    scale: float = 1.5,
    cmap: str = "gray",
):
    """Render an image grid and return ``(figure, axes)``."""
    if rows < 1 or columns < 1 or len(images) > rows * columns:
        raise ValueError("图像数量必须能放入正数行列组成的网格")
    figure, axes = plt.subplots(rows, columns, figsize=(columns * scale, rows * scale),
                                squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.set_axis_off()
        if index >= len(images):
            continue
        image = _to_numpy(images[index])
        if image.ndim == 3 and image.shape[0] in {1, 3, 4}:
            image = np.moveaxis(image, 0, -1)
        axis.imshow(np.squeeze(image), cmap=cmap if image.ndim == 2 else None)
        if titles is not None and index < len(titles):
            axis.set_title(titles[index])
    return figure, axes


def show_heatmaps(
    matrices,
    *,
    xlabel: str = "Keys",
    ylabel: str = "Queries",
    titles: Sequence[str] | None = None,
    figsize: tuple[float, float] = (6, 4),
    cmap: str = "Reds",
):
    """Render matrices shaped ``[rows, columns, height, width]``."""
    values = _to_numpy(matrices)
    if values.ndim != 4:
        raise ValueError("matrices 形状必须为 [rows, columns, height, width]")
    rows, columns = values.shape[:2]
    figure, axes = plt.subplots(rows, columns, figsize=figsize, sharex=True, sharey=True,
                                squeeze=False)
    image = None
    for row in range(rows):
        for column in range(columns):
            image = axes[row, column].imshow(values[row, column], cmap=cmap)
            if row == rows - 1:
                axes[row, column].set_xlabel(xlabel)
            if column == 0:
                axes[row, column].set_ylabel(ylabel)
            if titles and column < len(titles):
                axes[row, column].set_title(titles[column])
    figure.colorbar(image, ax=axes, shrink=0.6)
    return figure, axes


class TrainingBoard:
    """Collect and plot training metrics without notebook-only display dependencies."""

    def __init__(self):
        self.history: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def add(self, name: str, step: float, value: float) -> None:
        self.history[name].append((float(step), float(value)))

    def figure(self, *, xlabel: str = "step", ylabel: str = "value",
               figsize: tuple[float, float] = (6, 4)):
        figure, axes = plt.subplots(figsize=figsize)
        for name, points in self.history.items():
            values = np.asarray(points)
            axes.plot(values[:, 0], values[:, 1], label=name)
        set_axes(axes, xlabel=xlabel, ylabel=ylabel, legend=list(self.history))
        return figure, axes


def _series_list(values) -> list:
    if isinstance(values, np.ndarray) and values.ndim == 1:
        return [values]
    if hasattr(values, "ndim") and values.ndim == 1:
        return [values]
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        items = list(values)
        if not items or np.isscalar(items[0]):
            return [items]
        return items
    return [[values]]


def _to_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    return np.asarray(values)
