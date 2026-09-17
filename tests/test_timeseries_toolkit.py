import numpy as np
import pandas as pd
import pytest

from mathmodel.anomaly import detect_multivariate, statistical_anomalies
from mathmodel.correlation import cross_correlation, lag_correlation, rolling_correlation
from mathmodel.decomposition import cubic_spline_interpolate
from mathmodel.diagnostics import gph_estimate, normality_tests
from mathmodel.forecast_models import (
    event_indicators,
    fourier_features,
    logistic_growth,
    piecewise_linear_trend,
)
from mathmodel.signal import (
    BoxCoxTransformer,
    detrend,
    difference,
    fractional_difference,
    scale,
    smooth,
)
from mathmodel.validation import chronological_split, time_series_splits


def test_smoothing_scaling_and_transforms():
    values = pd.Series([1, 2, 100, 4, 5, 6, 7], index=list("abcdefg"))
    assert smooth(values, "hampel", window=3).loc["c"] < 100
    assert smooth(values, "ma", window=3).index.equals(values.index)
    assert np.isfinite(smooth(values, "gaussian")).all()
    assert np.isfinite(smooth(values, "savgol", window=5, polyorder=2)).all()
    for method in ["zscore", "minmax", "robust"]:
        transformed, fitted = scale(values, method)
        assert transformed.shape == values.shape
        assert fitted.inverse_transform(transformed.reshape(-1, 1)).shape == (7, 1)
    transformer = BoxCoxTransformer().fit(values)
    assert np.allclose(transformer.inverse_transform(transformer.transform(values)), values)
    assert abs(detrend(pd.Series(np.arange(10.0))).sum()) < 1e-10
    assert difference(values).isna().sum() == 1
    assert fractional_difference(np.arange(100.0), 0.2).notna().any()


@pytest.mark.parametrize("method", ["knn", "lof", "dbscan", "one_class_svm", "isolation_forest"])
def test_anomaly_detectors_share_output(method):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
    frame.loc[39] = [20, 20]
    result = detect_multivariate(frame, method, contamination=0.1)
    assert list(result) == ["score", "is_anomaly"]
    assert result["is_anomaly"].dtype == bool
    assert statistical_anomalies(frame["a"], "iqr").loc[39, "is_anomaly"]


def test_correlations_interpolation_and_diagnostics():
    rng = np.random.default_rng(9)
    values = pd.Series(rng.normal(size=128)).cumsum()
    assert rolling_correlation(values, values, 10, method="spearman").iloc[-1] == pytest.approx(1)
    assert lag_correlation(values, max_lag=3).loc[0] == pytest.approx(1)
    assert cross_correlation(values, values, 3).loc[0] == pytest.approx(1)
    missing = pd.Series([0.0, np.nan, 2.0, np.nan, 4.0])
    assert cubic_spline_interpolate(missing).isna().sum() == 0
    assert "jarque_bera" in normality_tests(rng.normal(size=100)).index
    estimate = gph_estimate(values)
    assert np.isfinite(estimate["d"])


def test_prophet_building_blocks_and_time_splits():
    t = np.arange(10)
    trend = piecewise_linear_trend(t, 1, 0, [5], [1])
    assert trend[4] == 4 and trend[6] == 7
    assert np.all(np.diff(logistic_growth(t, 10, 1, 5)) > 0)
    assert fourier_features(t, 7, 3).shape == (10, 6)
    dates = pd.date_range("2025-01-01", periods=5)
    events = pd.DataFrame({"event": ["launch"], "date": [dates[2]]})
    assert event_indicators(dates, events).loc[dates[2], "launch"] == 1
    train, test = chronological_split(20, 5, gap=2)
    assert train[-1] == 12 and test[0] == 15
    splits = list(time_series_splits(20, initial=8, horizon=3, step=3, window="rolling"))
    assert len(splits) == 4 and len(splits[0][0]) == 8
