"""Correlation and lag-dependence analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def autocorrelation(values, max_lag: int = 20) -> pd.Series:
    """Sample autocorrelation with pairwise complete observations."""
    series = pd.Series(values, dtype=float).dropna()
    if len(series) < 3 or not 0 <= max_lag < len(series):
        raise ValueError("max_lag 需介于 0 和序列长度 - 1 之间")
    centered = series - series.mean()
    denominator = float(centered @ centered)
    if denominator == 0:
        raise ValueError("常数序列没有可用的自相关")
    coefficients = [1.0]
    coefficients.extend(
        float(centered.iloc[lag:].to_numpy() @ centered.iloc[:-lag].to_numpy()) / denominator
        for lag in range(1, max_lag + 1)
    )
    return pd.Series(coefficients, index=pd.Index(range(max_lag + 1), name="lag"), name="acf")


def correlation_matrix(frame: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    if method not in {"pearson", "spearman"}:
        raise ValueError("method 必须为 pearson 或 spearman")
    return frame.astype(float).corr(method=method)


def rolling_correlation(left, right, window: int, *, method: str = "pearson") -> pd.Series:
    if window < 2:
        raise ValueError("window 至少为 2")
    left_series, right_series = pd.Series(left, dtype=float), pd.Series(right, dtype=float)
    if method == "pearson":
        return left_series.rolling(window).corr(right_series).rename("rolling_correlation")
    if method == "spearman":
        output = pd.Series(np.nan, index=left_series.index, name="rolling_correlation")
        for end in range(window, len(left_series) + 1):
            output.iloc[end - 1] = left_series.iloc[end - window:end].corr(
                right_series.iloc[end - window:end], method="spearman"
            )
        return output
    raise ValueError("method 必须为 pearson 或 spearman")


def acf_pacf(values, nlags: int = 20, *, alpha: float = 0.05) -> pd.DataFrame:
    from statsmodels.tsa.stattools import acf, pacf

    series = pd.Series(values, dtype=float).dropna()
    if not 1 <= nlags < len(series) // 2:
        raise ValueError("nlags 必须小于有效样本数的一半")
    acf_values, acf_ci = acf(series, nlags=nlags, alpha=alpha, fft=True)
    pacf_values, pacf_ci = pacf(series, nlags=nlags, alpha=alpha, method="ywm")
    return pd.DataFrame({"acf": acf_values, "acf_lower": acf_ci[:, 0],
                         "acf_upper": acf_ci[:, 1], "pacf": pacf_values,
                         "pacf_lower": pacf_ci[:, 0], "pacf_upper": pacf_ci[:, 1]},
                        index=pd.Index(range(nlags + 1), name="lag"))


def lag_correlation(left, right=None, max_lag: int = 20) -> pd.Series:
    """Correlation corr(left[t], right[t-lag]) for positive and negative lags."""
    left_series = pd.Series(left, dtype=float)
    right_series = left_series if right is None else pd.Series(right, dtype=float)
    if max_lag < 0:
        raise ValueError("max_lag 必须非负")
    lags = range(-max_lag, max_lag + 1)
    return pd.Series({lag: left_series.corr(right_series.shift(lag)) for lag in lags},
                     name="lag_correlation").rename_axis("lag")


def cross_correlation(left, right, max_lag: int = 20, *, normalize: bool = True) -> pd.Series:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("CCF 需要等长一维序列")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid] - np.mean(x[valid]), y[valid] - np.mean(y[valid])
    raw = np.correlate(x, y, mode="full")
    lags = np.arange(-len(x) + 1, len(x))
    keep = np.abs(lags) <= max_lag
    result = raw[keep]
    if normalize:
        result = result / np.sqrt(np.dot(x, x) * np.dot(y, y))
    return pd.Series(result, index=pd.Index(lags[keep], name="lag"), name="ccf")


def granger_causality(cause, effect, max_lag: int = 5) -> pd.DataFrame:
    """Test whether ``cause`` Granger-causes ``effect`` for each lag."""
    from statsmodels.tsa.stattools import grangercausalitytests

    data = pd.DataFrame({"effect": effect, "cause": cause}).dropna()
    results = grangercausalitytests(data[["effect", "cause"]], maxlag=max_lag, verbose=False)
    rows = []
    for lag, result in results.items():
        f_stat, p_value, df_denom, df_num = result[0]["ssr_ftest"]
        rows.append({"lag": lag, "f_statistic": f_stat, "p_value": p_value,
                     "df_num": df_num, "df_denom": df_denom})
    return pd.DataFrame(rows).set_index("lag")
