"""综合评价与关联分析算法。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def orient_indicators(
    frame: pd.DataFrame,
    negative: list[str] | None = None,
    targets: dict[str, float] | None = None,
) -> pd.DataFrame:
    """将成本型和目标型指标转换为越大越优，保留列名与索引。"""
    result = frame.astype(float).copy()
    for column in negative or []:
        result[column] = result[column].max() - result[column]
    for column, target in (targets or {}).items():
        distance = (result[column] - target).abs()
        result[column] = distance.max() - distance
    return result


def entropy_weights(frame: pd.DataFrame, epsilon: float = 1e-12) -> pd.Series:
    """按信息熵计算客观权重；常数列权重为零。"""
    values = frame.astype(float).to_numpy()
    ranges = np.ptp(values, axis=0)
    normalized = np.divide(
        values - values.min(axis=0),
        ranges,
        out=np.zeros_like(values, dtype=float),
        where=ranges > epsilon,
    )
    proportions = normalized / np.maximum(normalized.sum(axis=0), epsilon)
    n = len(frame)
    entropy = -(proportions * np.log(np.maximum(proportions, epsilon))).sum(axis=0) / np.log(n)
    diversity = np.where(ranges > epsilon, 1 - entropy, 0)
    if diversity.sum() <= epsilon:
        weights = np.full(len(frame.columns), 1 / len(frame.columns))
    else:
        weights = diversity / diversity.sum()
    return pd.Series(weights, index=frame.columns, name="weight")


def topsis(frame: pd.DataFrame, weights: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    """计算 TOPSIS 贴近度及名次，输入指标须已完成方向统一。"""
    values = frame.astype(float).to_numpy()
    denominator = np.linalg.norm(values, axis=0)
    normalized = np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)
    weight_values = entropy_weights(frame).to_numpy() if weights is None else np.asarray(weights)
    if len(weight_values) != values.shape[1] or np.any(weight_values < 0):
        raise ValueError("权重数量须与指标列数一致，且不能为负")
    weight_values = weight_values / weight_values.sum()
    weighted = normalized * weight_values
    positive, negative = weighted.max(axis=0), weighted.min(axis=0)
    distance_positive = np.linalg.norm(weighted - positive, axis=1)
    distance_negative = np.linalg.norm(weighted - negative, axis=1)
    score = distance_negative / np.maximum(distance_positive + distance_negative, 1e-12)
    return pd.DataFrame(
        {"score": score, "rank": pd.Series(score, index=frame.index).rank(ascending=False, method="min").astype(int)},
        index=frame.index,
    )


def grey_relational_grade(
    frame: pd.DataFrame,
    reference: str | pd.Series,
    rho: float = 0.5,
) -> pd.Series:
    """计算各序列相对参考序列的灰色关联度。"""
    if not 0 < rho < 1:
        raise ValueError("分辨系数 rho 必须位于 (0, 1)")
    reference_values = frame[reference] if isinstance(reference, str) else reference
    candidates = frame.drop(columns=[reference]) if isinstance(reference, str) else frame
    scaled = (candidates - candidates.min()) / (candidates.max() - candidates.min()).replace(0, 1)
    ref = (reference_values - reference_values.min()) / max(reference_values.max() - reference_values.min(), 1e-12)
    differences = scaled.sub(ref, axis=0).abs()
    minimum, maximum = differences.to_numpy().min(), differences.to_numpy().max()
    coefficients = (minimum + rho * maximum) / (differences + rho * maximum + 1e-12)
    return coefficients.mean().sort_values(ascending=False).rename("grey_relational_grade")

