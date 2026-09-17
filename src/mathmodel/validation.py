"""Time-aware model validation, metrics and selection helpers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def chronological_split(n_samples: int, test_size: float = 0.2,
                        gap: int = 0) -> tuple[np.ndarray, np.ndarray]:
    test_count = int(np.ceil(n_samples * test_size)) if isinstance(test_size, float) else test_size
    split = n_samples - test_count
    if split - gap <= 0 or test_count <= 0:
        raise ValueError("训练集、gap 和测试集大小不合法")
    return np.arange(split - gap), np.arange(split, n_samples)


def time_series_splits(n_samples: int, *, initial: int, horizon: int, step: int | None = None,
                       window: str = "expanding", gap: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding or rolling-origin train/test indices."""
    if window not in {"expanding", "rolling"}:
        raise ValueError("window 必须为 expanding 或 rolling")
    if min(initial, horizon) < 1 or gap < 0:
        raise ValueError("initial/horizon 必须为正，gap 必须非负")
    step = horizon if step is None else step
    train_end = initial
    while train_end + gap + horizon <= n_samples:
        train_start = 0 if window == "expanding" else train_end - initial
        yield np.arange(train_start, train_end), np.arange(train_end + gap, train_end + gap + horizon)
        train_end += step


def rmse(actual, predicted) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def aic(log_likelihood: float, n_parameters: int) -> float:
    if n_parameters < 0:
        raise ValueError("n_parameters 不能为负")
    return float(2 * n_parameters - 2 * log_likelihood)


def gaussian_prediction_interval(prediction, residuals, *, confidence: float = 0.95):
    from scipy.stats import norm

    if not 0 < confidence < 1:
        raise ValueError("confidence 必须位于 (0, 1)")
    prediction = np.asarray(prediction, dtype=float)
    error = np.asarray(residuals, dtype=float)
    margin = norm.ppf((1 + confidence) / 2) * np.std(error, ddof=1)
    return prediction - margin, prediction + margin


def evaluate_forecasts(actual, forecasts: dict[str, object], aic_values: dict[str, float] | None = None):
    rows = [{"model": name, "rmse": rmse(actual, prediction),
             "aic": np.nan if aic_values is None else aic_values.get(name, np.nan)}
            for name, prediction in forecasts.items()]
    return pd.DataFrame(rows).sort_values(["rmse", "aic"], na_position="last").reset_index(drop=True)
