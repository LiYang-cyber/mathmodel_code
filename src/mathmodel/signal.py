"""Time-series preprocessing, smoothing and transformations."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def _series(values) -> pd.Series:
    result = pd.Series(values, dtype=float).copy()
    if result.empty:
        raise ValueError("序列不能为空")
    return result


def smooth(
    values,
    method: Literal["ma", "gaussian", "savgol", "hampel"] = "ma",
    *,
    window: int = 5,
    sigma: float = 1.0,
    polyorder: int = 2,
    threshold: float = 3.0,
) -> pd.Series:
    """Apply a named smoother while preserving the original index."""
    series = _series(values)
    if window < 1:
        raise ValueError("window 必须为正整数")
    if method == "ma":
        return series.rolling(window, center=True, min_periods=1).mean().rename("ma")
    if method == "gaussian":
        if sigma <= 0:
            raise ValueError("sigma 必须大于 0")
        filled = series.interpolate(limit_direction="both")
        return pd.Series(gaussian_filter1d(filled, sigma), index=series.index, name="gaussian")
    if method == "savgol":
        if window % 2 == 0 or window <= polyorder or window > len(series):
            raise ValueError("SG window 必须为不大于序列长度的奇数，且大于 polyorder")
        filled = series.interpolate(limit_direction="both")
        return pd.Series(signal.savgol_filter(filled, window, polyorder), index=series.index,
                         name="savgol")
    if method == "hampel":
        median = series.rolling(window, center=True, min_periods=1).median()
        mad = (series - median).abs().rolling(window, center=True, min_periods=1).median()
        outlier = (series - median).abs() > threshold * 1.4826 * mad
        return series.mask(outlier, median).rename("hampel")
    raise ValueError(f"未知平滑方法: {method}")


def scale(values, method: Literal["zscore", "minmax", "robust"] = "zscore"):
    """Fit and apply a scaler; return transformed data and the fitted scaler."""
    array = np.asarray(values, dtype=float)
    one_dimensional = array.ndim == 1
    array = array.reshape(-1, 1) if one_dimensional else array
    scalers = {"zscore": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    if method not in scalers:
        raise ValueError(f"未知标准化方法: {method}")
    fitted = scalers[method]().fit(array)
    transformed = fitted.transform(array)
    return (transformed.ravel() if one_dimensional else transformed), fitted


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Column-wise Box-Cox transform with learned positive shifts and inverse transform."""

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        array = array.reshape(-1, 1) if array.ndim == 1 else array
        minimum = np.nanmin(array, axis=0)
        self.shift_ = np.where(minimum <= 0, 1 - minimum, 0.0)
        self.lambdas_ = np.array([stats.boxcox_normmax(array[:, i] + self.shift_[i])
                                  for i in range(array.shape[1])])
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float)
        one_dimensional = array.ndim == 1
        array = array.reshape(-1, 1) if one_dimensional else array
        shifted = array + self.shift_
        if np.any(shifted <= 0):
            raise ValueError("Box-Cox 新数据经训练集平移后必须为正数")
        output = np.column_stack([stats.boxcox(shifted[:, i], self.lambdas_[i])
                                  for i in range(array.shape[1])])
        return output.ravel() if one_dimensional else output

    def inverse_transform(self, X):
        from scipy.special import inv_boxcox

        array = np.asarray(X, dtype=float)
        one_dimensional = array.ndim == 1
        array = array.reshape(-1, 1) if one_dimensional else array
        output = np.column_stack([inv_boxcox(array[:, i], self.lambdas_[i]) - self.shift_[i]
                                  for i in range(array.shape[1])])
        return output.ravel() if one_dimensional else output


def detrend(values, method: Literal["linear", "constant"] = "linear") -> pd.Series:
    series = _series(values)
    if series.isna().any():
        raise ValueError("去趋势前请先处理缺失值")
    return pd.Series(signal.detrend(series.to_numpy(), type=method), index=series.index,
                     name="detrended")


def linear_trend(values) -> pd.DataFrame:
    """Fit a linear trend and return actual, trend and residual series."""
    series = _series(values).dropna()
    if len(series) < 3:
        raise ValueError("至少需要三个非缺失观测")
    time = np.arange(len(series), dtype=float)
    fitted_model = stats.linregress(time, series.to_numpy())
    fitted = fitted_model.intercept + fitted_model.slope * time
    result = pd.DataFrame(
        {"actual": series.to_numpy(), "trend": fitted, "residual": series.to_numpy() - fitted},
        index=series.index,
    )
    result.attrs.update(
        slope=fitted_model.slope,
        intercept=fitted_model.intercept,
        r_squared=fitted_model.rvalue**2,
        p_value=fitted_model.pvalue,
    )
    return result


def difference(values, order: int = 1, lag: int = 1) -> pd.Series:
    if order < 0 or lag < 1:
        raise ValueError("order 必须非负且 lag 必须为正整数")
    result = _series(values)
    for _ in range(order):
        result = result.diff(lag)
    return result.rename("difference")


def fractional_weights(d: float, *, threshold: float = 1e-5, max_lags: int = 1000) -> np.ndarray:
    """Return truncated binomial weights for (1-L)^d, newest observation first."""
    if threshold <= 0 or max_lags < 1:
        raise ValueError("threshold 必须大于 0，max_lags 必须为正整数")
    weights = [1.0]
    for k in range(1, max_lags):
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
    return np.asarray(weights)


def fractional_difference(values, d: float, *, threshold: float = 1e-5,
                          max_lags: int = 1000) -> pd.Series:
    series = _series(values)
    weights = fractional_weights(d, threshold=threshold, max_lags=min(max_lags, len(series)))
    result = series.rolling(len(weights), min_periods=len(weights)).apply(
        lambda window: float(np.dot(weights, window[::-1])), raw=True
    )
    return result.rename(f"fracdiff_{d:g}")
