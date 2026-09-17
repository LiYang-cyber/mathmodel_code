"""Classical forecasting models and Prophet-compatible building blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signal import fractional_difference, fractional_weights


@dataclass
class ForecastResult:
    mean: pd.Series
    lower: pd.Series | None = None
    upper: pd.Series | None = None


class ARIMAForecaster:
    """Small statsmodels ARMA/ARIMA adapter with prediction intervals."""

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0), trend: str | None = None):
        self.order = order
        self.trend = trend

    def fit(self, values):
        from statsmodels.tsa.arima.model import ARIMA

        self.model_ = ARIMA(pd.Series(values, dtype=float), order=self.order,
                            trend=self.trend).fit()
        return self

    def predict(self, horizon: int, *, alpha: float = 0.05) -> ForecastResult:
        forecast = self.model_.get_forecast(steps=horizon)
        interval = forecast.conf_int(alpha=alpha)
        return ForecastResult(pd.Series(forecast.predicted_mean, name="forecast"),
                              pd.Series(interval.iloc[:, 0], name="lower"),
                              pd.Series(interval.iloc[:, 1], name="upper"))


class ARFIMAForecaster:
    """Practical ARFIMA adapter: fractional differencing followed by ARMA."""

    def __init__(self, order: tuple[int, float, int] = (1, 0.2, 0), threshold: float = 1e-4):
        self.order = order
        self.threshold = threshold

    def fit(self, values):
        p, d, q = self.order
        self.original_ = pd.Series(values, dtype=float)
        self.weights_ = fractional_weights(d, threshold=self.threshold,
                                           max_lags=len(self.original_))
        transformed = fractional_difference(self.original_, d, threshold=self.threshold).dropna()
        self.start_ = transformed.index[0]
        self.model_ = ARIMAForecaster((p, 0, q)).fit(transformed)
        return self

    def predict(self, horizon: int, *, alpha: float = 0.05) -> ForecastResult:
        result = self.model_.predict(horizon, alpha=alpha)
        history = self.original_.to_list()

        def invert(transformed_values) -> pd.Series:
            reconstructed = history.copy()
            for value in np.asarray(transformed_values, dtype=float):
                available = min(len(self.weights_) - 1, len(reconstructed))
                lagged = np.asarray(reconstructed[-available:][::-1])
                level = value - np.dot(self.weights_[1:available + 1], lagged)
                reconstructed.append(float(level))
            return pd.Series(reconstructed[-horizon:])

        return ForecastResult(invert(result.mean).rename("forecast"),
                              None if result.lower is None else invert(result.lower).rename("lower"),
                              None if result.upper is None else invert(result.upper).rename("upper"))


class KalmanForecaster:
    def __init__(self, level: str = "local linear trend", seasonal: int | None = None):
        self.level, self.seasonal = level, seasonal

    def fit(self, values):
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        self.model_ = UnobservedComponents(pd.Series(values, dtype=float), level=self.level,
                                           seasonal=self.seasonal).fit(disp=False)
        return self

    def predict(self, horizon: int, *, alpha: float = 0.05) -> ForecastResult:
        forecast = self.model_.get_forecast(horizon)
        interval = forecast.conf_int(alpha=alpha)
        return ForecastResult(pd.Series(forecast.predicted_mean, name="forecast"),
                              pd.Series(interval.iloc[:, 0], name="lower"),
                              pd.Series(interval.iloc[:, 1], name="upper"))


class ProphetForecaster:
    """Lazy wrapper around Prophet; Prophet remains an optional dependency."""

    def __init__(self, **params):
        self.params = params

    def fit(self, dates, values, events: pd.DataFrame | None = None):
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError("Prophet 预测需要安装可选依赖：pip install prophet") from exc
        self.model_ = Prophet(**self.params)
        if events is not None:
            for name in events.columns:
                self.model_.add_regressor(name)
        frame = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})
        if events is not None:
            frame = pd.concat([frame.reset_index(drop=True), events.reset_index(drop=True)], axis=1)
        self.model_.fit(frame)
        self.last_date_ = frame["ds"].max()
        return self

    def predict(self, horizon: int, *, frequency: str = "D", events: pd.DataFrame | None = None):
        future = pd.DataFrame({"ds": pd.date_range(self.last_date_, periods=horizon + 1,
                                                    freq=frequency)[1:]})
        if events is not None:
            future = pd.concat([future, events.reset_index(drop=True)], axis=1)
        result = self.model_.predict(future)
        return result[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def piecewise_linear_trend(t, growth: float, offset: float, changepoints,
                           deltas) -> np.ndarray:
    """Prophet-style continuous piecewise-linear trend."""
    t = np.asarray(t, dtype=float)
    changepoints, deltas = np.asarray(changepoints, dtype=float), np.asarray(deltas, dtype=float)
    if len(changepoints) != len(deltas):
        raise ValueError("changepoints 与 deltas 必须等长")
    trend = growth * t + offset
    for point, delta in zip(changepoints, deltas, strict=True):
        trend += delta * np.maximum(t - point, 0)
    return trend


def logistic_growth(t, capacity, growth: float, midpoint: float, floor=0.0) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    return floor + (capacity - floor) / (1 + np.exp(-growth * (t - midpoint)))


def fourier_features(t, period: float, order: int, *, prefix: str = "season") -> pd.DataFrame:
    if period <= 0 or order < 1:
        raise ValueError("period 和 order 必须为正数")
    t = np.asarray(t, dtype=float)
    data = {}
    for harmonic in range(1, order + 1):
        angle = 2 * np.pi * harmonic * t / period
        data[f"{prefix}_sin_{harmonic}"] = np.sin(angle)
        data[f"{prefix}_cos_{harmonic}"] = np.cos(angle)
    return pd.DataFrame(data)


def event_indicators(dates, events: pd.DataFrame, *, lower_window: int = 0,
                     upper_window: int = 0) -> pd.DataFrame:
    """Create event dummy columns from an events table with ``event`` and ``date``."""
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    required = {"event", "date"}
    if not required.issubset(events.columns):
        raise ValueError("events 必须包含 event 和 date 列")
    output = pd.DataFrame(0, index=index, columns=sorted(events["event"].unique()), dtype=int)
    for row in events.itertuples(index=False):
        for shift in range(lower_window, upper_window + 1):
            date = pd.Timestamp(row.date) + pd.Timedelta(days=shift)
            if date in output.index:
                output.loc[date, row.event] = 1
    return output


def laplace_log_prior(parameters, scale: float = 0.05) -> float:
    """Log-density (up to an additive constant) of Prophet's sparse Laplace prior."""
    if scale <= 0:
        raise ValueError("scale 必须大于 0")
    return float(-np.abs(np.asarray(parameters, dtype=float)).sum() / scale)
