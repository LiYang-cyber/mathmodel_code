"""Trend/seasonal decomposition and empirical mode decomposition."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def stl_decompose(values, period: int, *, robust: bool = True) -> pd.DataFrame:
    from statsmodels.tsa.seasonal import STL

    series = pd.Series(values, dtype=float)
    if period < 2 or len(series.dropna()) < 2 * period:
        raise ValueError("STL 至少需要两个完整周期，且 period >= 2")
    fitted = STL(series, period=period, robust=robust).fit()
    return pd.DataFrame({"observed": series, "trend": fitted.trend,
                         "seasonal": fitted.seasonal, "residual": fitted.resid})


def loess_smooth(values, *, fraction: float = 0.25, iterations: int = 3,
                 x=None) -> pd.Series:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    series = pd.Series(values, dtype=float)
    positions = np.arange(len(series)) if x is None else np.asarray(x)
    if not 0 < fraction <= 1:
        raise ValueError("fraction 必须位于 (0, 1]")
    valid = series.notna().to_numpy()
    fitted = lowess(series.to_numpy()[valid], positions[valid], frac=fraction, it=iterations,
                    return_sorted=False)
    result = pd.Series(np.nan, index=series.index, name="loess")
    result.iloc[np.flatnonzero(valid)] = fitted
    return result


def emd_decompose(values, *, max_imfs: int = -1) -> pd.DataFrame:
    """EMD via the optional ``EMD-signal`` package."""
    try:
        from PyEMD import EMD
    except ImportError as exc:
        raise ImportError("EMD 需要安装可选依赖：pip install EMD-signal") from exc
    series = pd.Series(values, dtype=float)
    if series.isna().any():
        raise ValueError("EMD 前请先处理缺失值")
    imfs = EMD()(series.to_numpy(), max_imf=max_imfs)
    result = pd.DataFrame(imfs.T, index=series.index,
                          columns=[f"imf_{i + 1}" for i in range(len(imfs))])
    result["residual"] = series - result.sum(axis=1)
    return result


def vmd_decompose(values, *, modes: int = 5, alpha: float = 2000,
                  tau: float = 0.0, dc: bool = False, init: Literal[0, 1, 2] = 1,
                  tolerance: float = 1e-7) -> pd.DataFrame:
    """Variational mode decomposition via the optional ``vmdpy`` package."""
    try:
        from vmdpy import VMD
    except ImportError as exc:
        raise ImportError("VMD 需要安装可选依赖：pip install vmdpy") from exc
    series = pd.Series(values, dtype=float)
    if series.isna().any() or modes < 1:
        raise ValueError("VMD 需要无缺失序列且 modes 为正整数")
    modes_array, _, _ = VMD(series.to_numpy(), alpha, tau, modes, int(dc), init, tolerance)
    return pd.DataFrame(modes_array.T, index=series.index,
                        columns=[f"mode_{i + 1}" for i in range(modes_array.shape[0])])


def cubic_spline_interpolate(values, *, x=None, extrapolate: bool = False) -> pd.Series:
    """Fill internal missing values with a natural cubic spline."""
    series = pd.Series(values, dtype=float)
    positions = np.arange(len(series), dtype=float) if x is None else np.asarray(x, dtype=float)
    valid = series.notna().to_numpy()
    if valid.sum() < 2 or len(positions) != len(series):
        raise ValueError("三次样条至少需要两个有效点，且 x 与序列等长")
    spline = CubicSpline(positions[valid], series.to_numpy()[valid], bc_type="natural",
                         extrapolate=extrapolate)
    result = series.copy()
    fill = ~valid
    if not extrapolate:
        fill &= (positions >= positions[valid].min()) & (positions <= positions[valid].max())
    result.iloc[np.flatnonzero(fill)] = spline(positions[fill])
    return result.rename("cubic_spline")
