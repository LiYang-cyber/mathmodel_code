"""Lightweight descriptive time-series analysis without optional dependencies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def autocorrelation(values: pd.Series, max_lag: int = 20) -> pd.Series:
    """Sample autocorrelation with pairwise complete observations."""
    series = pd.Series(values, dtype=float).dropna()
    if len(series) < 3 or not 0 <= max_lag < len(series):
        raise ValueError("max_lag 需介于 0 和序列长度 - 1 之间")
    centered = series - series.mean()
    denominator = float(centered @ centered)
    if denominator == 0:
        raise ValueError("常数序列没有可用的自相关")
    return pd.Series(
        [1.0] + [float(centered.iloc[lag:].to_numpy() @ centered.iloc[:-lag].to_numpy()) /
                 denominator for lag in range(1, max_lag + 1)],
        index=pd.Index(range(max_lag + 1), name="lag"), name="acf",
    )


def ljung_box(values: pd.Series, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box portmanteau test based on the sample autocorrelations."""
    series = pd.Series(values, dtype=float).dropna()
    if not 1 <= lags < len(series):
        raise ValueError("lags 需介于 1 和序列长度 - 1 之间")
    acf = autocorrelation(series, lags).iloc[1:]
    n = len(series)
    q_values = np.cumsum(n * (n + 2) * acf.to_numpy() ** 2 /
                         (n - np.arange(1, lags + 1)))
    return pd.DataFrame({"Q": q_values, "p_value": stats.chi2.sf(q_values, np.arange(1, lags + 1))},
                        index=pd.Index(range(1, lags + 1), name="lag"))


def moving_average(values: pd.Series, window: int, *, center: bool = False) -> pd.Series:
    if window < 1:
        raise ValueError("window 必须为正整数")
    return pd.Series(values, dtype=float).rolling(window, center=center).mean().rename("moving_average")


def linear_trend(values: pd.Series) -> pd.DataFrame:
    """Fit a linear time trend and return fitted values and residuals."""
    series = pd.Series(values, dtype=float).dropna()
    if len(series) < 3:
        raise ValueError("至少需要三个非缺失观测")
    time = np.arange(len(series), dtype=float)
    fit = stats.linregress(time, series.to_numpy())
    fitted = fit.intercept + fit.slope * time
    result = pd.DataFrame({"actual": series.to_numpy(), "trend": fitted,
                           "residual": series.to_numpy() - fitted}, index=series.index)
    result.attrs.update(slope=fit.slope, intercept=fit.intercept,
                        r_squared=fit.rvalue**2, p_value=fit.pvalue)
    return result
