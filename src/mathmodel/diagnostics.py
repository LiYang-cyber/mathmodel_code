"""Statistical tests and long-memory diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .correlation import autocorrelation


def ljung_box(values, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box portmanteau test based on sample autocorrelations."""
    series = pd.Series(values, dtype=float).dropna()
    if not 1 <= lags < len(series):
        raise ValueError("lags 需介于 1 和序列长度 - 1 之间")
    coefficients = autocorrelation(series, lags).iloc[1:]
    n = len(series)
    q_values = np.cumsum(
        n * (n + 2) * coefficients.to_numpy() ** 2 / (n - np.arange(1, lags + 1))
    )
    return pd.DataFrame(
        {"Q": q_values, "p_value": stats.chi2.sf(q_values, np.arange(1, lags + 1))},
        index=pd.Index(range(1, lags + 1), name="lag"),
    )


def normality_tests(values) -> pd.DataFrame:
    series = pd.Series(values, dtype=float).dropna().to_numpy()
    if len(series) < 3:
        raise ValueError("正态性检验至少需要 3 个有效观测")
    rows = []
    if len(series) <= 5000:
        statistic, p_value = stats.shapiro(series)
        rows.append(("shapiro", statistic, p_value))
    if len(series) >= 8:
        statistic, p_value = stats.normaltest(series)
        rows.append(("dagostino_k2", statistic, p_value))
    statistic, p_value = stats.jarque_bera(series)
    rows.append(("jarque_bera", statistic, p_value))
    return pd.DataFrame(rows, columns=["test", "statistic", "p_value"]).set_index("test")


def stationarity_tests(values, *, regression: str = "c", nlags: str | int = "auto") -> pd.DataFrame:
    from statsmodels.tsa.stattools import adfuller, kpss

    series = pd.Series(values, dtype=float).dropna()
    adf = adfuller(series, regression=regression, autolag="AIC" if nlags == "auto" else None,
                   maxlag=None if nlags == "auto" else int(nlags))
    kpss_result = kpss(series, regression=regression, nlags=nlags)
    return pd.DataFrame([
        {"test": "ADF", "statistic": adf[0], "p_value": adf[1], "lags": adf[2],
         "null_hypothesis": "unit root (non-stationary)"},
        {"test": "KPSS", "statistic": kpss_result[0], "p_value": kpss_result[1],
         "lags": kpss_result[2], "null_hypothesis": "stationary"},
    ]).set_index("test")


def gph_estimate(values, *, bandwidth: int | None = None) -> pd.Series:
    """Geweke-Porter-Hudak log-periodogram estimate of fractional order d."""
    series = pd.Series(values, dtype=float).dropna().to_numpy()
    n = len(series)
    bandwidth = int(np.sqrt(n)) if bandwidth is None else int(bandwidth)
    if n < 16 or not 2 <= bandwidth < n // 2:
        raise ValueError("GPH 至少需要 16 个样本，且 bandwidth 位于 [2, n/2)")
    centered = series - series.mean()
    frequencies = 2 * np.pi * np.arange(1, bandwidth + 1) / n
    fft_values = np.fft.fft(centered)
    periodogram = np.abs(fft_values[1:bandwidth + 1]) ** 2 / (2 * np.pi * n)
    regressor = np.log(4 * np.sin(frequencies / 2) ** 2)
    slope, intercept, r_value, p_value, std_error = stats.linregress(
        regressor, np.log(np.maximum(periodogram, np.finfo(float).tiny))
    )
    return pd.Series({"d": -slope, "intercept": intercept, "r_squared": r_value**2,
                      "p_value": p_value, "std_error": std_error,
                      "bandwidth": bandwidth})


def acf_long_memory(values, *, max_lag: int | None = None) -> pd.Series:
    """Heuristic power-law ACF diagnostic; use GPH for formal estimation."""
    from statsmodels.tsa.stattools import acf

    series = pd.Series(values, dtype=float).dropna()
    max_lag = min(len(series) // 4, 100) if max_lag is None else max_lag
    coefficients = acf(series, nlags=max_lag, fft=True)[1:]
    lags = np.arange(1, max_lag + 1)
    positive = coefficients > 0
    if positive.sum() < 3:
        return pd.Series({"decay_exponent": np.nan, "r_squared": np.nan,
                          "long_memory_candidate": False})
    fit = stats.linregress(np.log(lags[positive]), np.log(coefficients[positive]))
    return pd.Series({"decay_exponent": -fit.slope, "r_squared": fit.rvalue**2,
                      "long_memory_candidate": bool(0 < -fit.slope < 1 and fit.rvalue**2 > 0.5)})
