"""Bayesian, Markov, queueing, inventory and decision-analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def beta_binomial_update(successes: int, trials: int, *, alpha: float = 1, beta: float = 1,
                         credibility: float = 0.95) -> pd.Series:
    """Conjugate Bayesian update for an unknown Bernoulli probability."""
    if not 0 <= successes <= trials or alpha <= 0 or beta <= 0:
        raise ValueError("需满足 0 <= successes <= trials，且先验参数为正")
    posterior = stats.beta(alpha + successes, beta + trials - successes)
    tail = (1 - credibility) / 2
    low, high = posterior.ppf([tail, 1 - tail])
    return pd.Series({"posterior_alpha": alpha + successes,
                      "posterior_beta": beta + trials - successes,
                      "posterior_mean": posterior.mean(), "interval_low": low,
                      "interval_high": high})


def markov_stationary(transition: np.ndarray, *, tolerance: float = 1e-10) -> np.ndarray:
    """Compute a finite Markov chain stationary distribution."""
    matrix = np.asarray(transition, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("转移矩阵必须为方阵")
    if (matrix < 0).any() or not np.allclose(matrix.sum(axis=1), 1, atol=tolerance):
        raise ValueError("转移概率必须非负，且每行和为 1")
    n = len(matrix)
    lhs = np.vstack([matrix.T - np.eye(n), np.ones(n)])
    rhs = np.r_[np.zeros(n), 1]
    solution = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    solution[np.abs(solution) < tolerance] = 0
    return solution / solution.sum()


def markov_n_step(initial: np.ndarray, transition: np.ndarray, steps: int) -> np.ndarray:
    initial = np.asarray(initial, dtype=float)
    matrix = np.asarray(transition, dtype=float)
    if steps < 0 or initial.ndim != 1 or matrix.shape != (len(initial), len(initial)):
        raise ValueError("初始分布、转移矩阵或步数不合法")
    markov_stationary(matrix)
    if (initial < 0).any() or not np.isclose(initial.sum(), 1):
        raise ValueError("初始概率必须非负且和为 1")
    return initial @ np.linalg.matrix_power(matrix, steps)


def mm1_queue(arrival_rate: float, service_rate: float) -> pd.Series:
    """Steady-state M/M/1 queue measures (rho, L, Lq, W, Wq)."""
    if arrival_rate < 0 or service_rate <= arrival_rate:
        raise ValueError("稳定 M/M/1 队列需要 0 <= 到达率 < 服务率")
    rho = arrival_rate / service_rate
    return pd.Series({"utilization": rho, "mean_system_size": rho / (1 - rho),
                      "mean_queue_size": rho**2 / (1 - rho),
                      "mean_system_time": 1 / (service_rate - arrival_rate),
                      "mean_wait_time": rho / (service_rate - arrival_rate)})


def economic_order_quantity(demand: float, order_cost: float, holding_cost: float,
                            *, lead_time: float = 0, safety_stock: float = 0) -> pd.Series:
    """Classical deterministic EOQ and reorder point."""
    if demand < 0 or order_cost <= 0 or holding_cost <= 0 or lead_time < 0:
        raise ValueError("需求非负，订购/持有成本为正，提前期非负")
    quantity = np.sqrt(2 * demand * order_cost / holding_cost)
    return pd.Series({"eoq": quantity, "orders_per_period": demand / quantity if quantity else 0,
                      "cycle_inventory": quantity / 2,
                      "reorder_point": demand * lead_time + safety_stock})


def expected_utility(payoffs: pd.DataFrame, probabilities: pd.Series) -> pd.DataFrame:
    """Rank alternatives by expected payoff; columns are states of nature."""
    probabilities = pd.Series(probabilities, dtype=float).reindex(payoffs.columns)
    if probabilities.isna().any() or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1):
        raise ValueError("状态概率须覆盖全部收益列、非负且和为 1")
    expected = payoffs.astype(float).mul(probabilities, axis=1).sum(axis=1)
    return pd.DataFrame({"expected_payoff": expected, "rank": expected.rank(ascending=False, method="min")}) \
        .sort_values("rank")
