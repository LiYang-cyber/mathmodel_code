import numpy as np
import pandas as pd
import pytest

from mathmodel.decision import (
    beta_binomial_update,
    economic_order_quantity,
    expected_utility,
    markov_n_step,
    markov_stationary,
    mm1_queue,
)
from mathmodel.statistics import (
    canonical_correlation,
    discriminant_classifier,
    factor_analysis,
    multiple_regression,
    one_way_anova,
    principal_components,
    stepwise_regression,
)
from mathmodel.time_series_analysis import autocorrelation, linear_trend, ljung_box


def test_multiple_and_stepwise_regression_recover_signal():
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"signal": np.arange(80), "noise": rng.normal(size=80)})
    y = pd.Series(2 + 3 * X["signal"] + rng.normal(scale=0.1, size=80))
    result = multiple_regression(X, y)
    selected, stepwise = stepwise_regression(X, y)
    assert result.r_squared > 0.99
    assert result.coefficients.loc["signal", "estimate"] == pytest.approx(3, rel=0.01)
    assert selected == ["signal"]
    assert stepwise.adjusted_r_squared > 0.99


def test_anova_and_multivariate_methods():
    values = pd.Series([1, 2, 3, 10, 11, 12])
    groups = pd.Series(["a"] * 3 + ["b"] * 3)
    assert one_way_anova(values, groups).loc["between", "p_value"] < 0.01
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(40, 5)), columns=list("abcde"))
    pca, scores = principal_components(X, 0.8)
    _, factors, loadings = factor_analysis(X, 2)
    _, xs, ys, correlations = canonical_correlation(X.iloc[:, :3], X.iloc[:, 3:], 2)
    assert scores.shape[1] == pca.n_components_
    assert factors.shape == (40, 2) and loadings.shape == (5, 2)
    assert xs.shape == ys.shape == (40, 2)
    assert (correlations.abs() <= 1).all()


@pytest.mark.parametrize("method", ["distance", "fisher", "bayes"])
def test_discriminant_estimators(method):
    rng = np.random.default_rng(12)
    X = np.r_[rng.normal(0, 0.2, (8, 2)), rng.normal(4, 0.2, (8, 2))]
    y = np.array([0] * 8 + [1] * 8)
    model = discriminant_classifier(method).fit(X, y)
    assert (model.predict(X) == y).mean() == 1


def test_bayes_markov_queue_inventory_and_decision():
    posterior = beta_binomial_update(8, 10)
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    stationary = markov_stationary(transition)
    assert posterior["posterior_mean"] == pytest.approx(0.75)
    assert stationary == pytest.approx([0.6, 0.4])
    assert markov_n_step(np.array([1.0, 0.0]), transition, 0) == pytest.approx([1, 0])
    assert mm1_queue(2, 3)["utilization"] == pytest.approx(2 / 3)
    assert economic_order_quantity(1000, 20, 5)["eoq"] == pytest.approx(np.sqrt(8000))
    payoffs = pd.DataFrame([[5, 1], [2, 4]], index=["A", "B"], columns=["good", "bad"])
    ranked = expected_utility(payoffs, pd.Series({"good": 0.75, "bad": 0.25}))
    assert ranked.index[0] == "A"


def test_time_series_descriptives():
    series = pd.Series(np.arange(30, dtype=float))
    assert autocorrelation(series, 3).iloc[0] == 1
    assert ljung_box(series, 3).shape == (3, 2)
    trend = linear_trend(series)
    assert trend.attrs["slope"] == pytest.approx(1)
    assert np.abs(trend["residual"]).max() < 1e-12
