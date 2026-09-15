"""数据预处理组件；所有有状态变换均遵循 sklearn fit/transform 接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier


class RandomSampleImputer(BaseEstimator, TransformerMixin):
    """从训练列的已观测值中随机抽样填补；同一随机种子可复现。"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=object)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.observed_ = [column[~pd.isna(column)] for column in values.T]
        if any(len(column) == 0 for column in self.observed_):
            raise ValueError("随机填补不能处理训练集中全为空的列")
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=object).copy()
        one_dimensional = values.ndim == 1
        if one_dimensional:
            values = values.reshape(-1, 1)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("输入列数与 fit() 时不一致")
        rng = np.random.default_rng(self.random_state)
        for index, observed in enumerate(self.observed_):
            missing = pd.isna(values[:, index])
            values[missing, index] = rng.choice(observed, size=int(missing.sum()), replace=True)
        try:
            values = values.astype(float)
        except (TypeError, ValueError):
            pass
        return values.ravel() if one_dimensional else values


class DecimalScaler(BaseEstimator, TransformerMixin):
    """小数定标：每列除以 10^j，使绝对值最大值小于 1。"""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        finite_magnitude = np.where(np.isfinite(values), np.abs(values), -np.inf)
        maximum = np.max(finite_magnitude, axis=0)
        maximum = np.where(np.isfinite(maximum), maximum, 0)
        self.powers_ = np.where(maximum > 0, np.floor(np.log10(maximum)).astype(int) + 1, 0)
        self.scale_ = np.power(10.0, self.powers_)
        self.n_features_in_ = values.shape[1] if values.ndim > 1 else 1
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) / self.scale_

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float) * self.scale_


class LogisticScaler(BaseEstimator, TransformerMixin):
    """先按训练集中心和尺度归一，再映射到 Logistic 函数的 (0, 1)。"""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.center_ = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        self.n_features_in_ = values.shape[1] if values.ndim > 1 else 1
        return self

    def transform(self, X):
        return expit((np.asarray(X, dtype=float) - self.center_) / self.scale_)

    def inverse_transform(self, X):
        values = np.clip(np.asarray(X, dtype=float), 1e-12, 1 - 1e-12)
        return (np.log(values / (1 - values)) * self.scale_) + self.center_


class LogShiftTransformer(BaseEstimator, TransformerMixin):
    """按训练列学习非负平移量后执行 log1p，适合右偏变量。"""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        minimum = np.nanmin(values, axis=0)
        self.shift_ = np.where(minimum <= -1, -minimum, 0.0)
        return self

    def transform(self, X):
        shifted = np.asarray(X, dtype=float) + self.shift_
        if np.any(shifted <= -1):
            raise ValueError("新数据超出训练集范围，log1p 输入小于等于 -1")
        return np.log1p(shifted)

    def inverse_transform(self, X):
        return np.expm1(np.asarray(X, dtype=float)) - self.shift_


def make_encoder(method: Literal["ordinal", "onehot"]):
    """返回数字编码或 One-Hot 编码器，未知类别不会导致预测失败。"""
    if method == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if method == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    raise ValueError(f"未知编码方法: {method}")


def make_imputer(method: Literal["mean", "median", "random", "model"], random_state: int = 42):
    """返回均值、随机样本或 ExtraTrees 迭代填补器。"""
    if method == "mean":
        return SimpleImputer(strategy="mean")
    if method == "median":
        return SimpleImputer(strategy="median")
    if method == "random":
        return RandomSampleImputer(random_state=random_state)
    if method == "model":
        estimator = ExtraTreesRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)
        return IterativeImputer(estimator=estimator, max_iter=10, random_state=random_state,
                                initial_strategy="median", skip_complete=True)
    raise ValueError(f"未知填补方法: {method}")


def drop_missing(frame: pd.DataFrame, axis: Literal["rows", "columns"] = "rows",
                 threshold: float = 0.0) -> pd.DataFrame:
    """删除缺失比例超过阈值的行或列；threshold=0 表示含缺失即删除。"""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须位于 [0, 1]")
    target_axis = 1 if axis == "rows" else 0
    ratios = frame.isna().mean(axis=target_axis)
    if axis == "rows":
        return frame.loc[ratios <= threshold].copy()
    if axis == "columns":
        return frame.loc[:, ratios <= threshold].copy()
    raise ValueError("axis 必须是 rows 或 columns")


def make_scaler(method: Literal["zscore", "minmax", "decimal", "logistic", "center"]):
    if method == "zscore":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    if method == "decimal":
        return DecimalScaler()
    if method == "logistic":
        return LogisticScaler()
    if method == "center":
        return StandardScaler(with_std=False)
    raise ValueError(f"未知缩放方法: {method}")


def make_unsupervised_discretizer(method: Literal["uniform", "quantile", "kmeans"], bins: int = 5):
    return KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=method)


class SupervisedDiscretizer(BaseEstimator, TransformerMixin):
    """信息增益、ChiMerge 或 CAIM 监督离散化；当前处理单个连续特征。"""

    def __init__(self, method: Literal["information_gain", "chimerge", "caim"] = "information_gain",
                 max_bins: int = 5, min_samples_leaf: int = 5):
        self.method = method
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        values = np.asarray(X, dtype=float).reshape(-1)
        labels = np.asarray(y)
        valid = ~np.isnan(values) & ~pd.isna(labels)
        values, labels = values[valid], labels[valid]
        if len(np.unique(labels)) < 2:
            raise ValueError("监督离散化至少需要两个类别")
        if self.method == "information_gain":
            tree = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=self.max_bins,
                                          min_samples_leaf=self.min_samples_leaf, random_state=42)
            tree.fit(values.reshape(-1, 1), labels)
            self.bin_edges_ = np.sort(tree.tree_.threshold[tree.tree_.threshold != -2])
        elif self.method == "chimerge":
            self.bin_edges_ = self._chimerge(values, labels)
        elif self.method == "caim":
            self.bin_edges_ = self._caim(values, labels)
        else:
            raise ValueError(f"未知监督离散化方法: {self.method}")
        return self

    @staticmethod
    def _candidate_edges(values: np.ndarray, labels: np.ndarray) -> list[float]:
        order = np.argsort(values)
        sorted_values, sorted_labels = values[order], labels[order]
        candidates = sorted({float((left + right) / 2) for left, right, left_y, right_y in zip(
            sorted_values[:-1], sorted_values[1:], sorted_labels[:-1], sorted_labels[1:], strict=True
        ) if left < right and left_y != right_y})
        if len(candidates) > 256:
            candidates = np.unique(np.quantile(candidates, np.linspace(0, 1, 256))).tolist()
        return candidates

    def _chimerge(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        edges = self._candidate_edges(values, labels)
        while len(edges) + 1 > self.max_bins:
            bins = np.digitize(values, edges)
            contingency = pd.crosstab(bins, labels).reindex(
                index=range(len(edges) + 1), fill_value=0
            ).to_numpy()
            scores = []
            for boundary in range(len(edges)):
                table = contingency[boundary:boundary + 2]
                nonempty_columns = table.sum(axis=0) > 0
                table = table[:, nonempty_columns]
                if table.shape[0] < 2 or table.shape[1] < 2:
                    scores.append(0.0)
                else:
                    scores.append(float(chi2_contingency(table, correction=False)[0]))
            edges.pop(int(np.argmin(scores)))
        return np.asarray(edges)

    @staticmethod
    def _caim_score(values: np.ndarray, labels: np.ndarray, edges: list[float]) -> float:
        table = pd.crosstab(np.digitize(values, edges), labels).to_numpy()
        interval_totals = table.sum(axis=1)
        return float(np.sum(np.max(table, axis=1) ** 2 / np.maximum(interval_totals, 1)) / len(table))

    def _caim(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        remaining = self._candidate_edges(values, labels)
        selected: list[float] = []
        best_score = self._caim_score(values, labels, selected)
        while remaining and len(selected) + 1 < self.max_bins:
            scores = [self._caim_score(values, labels, sorted([*selected, edge])) for edge in remaining]
            best_index = int(np.argmax(scores))
            if scores[best_index] <= best_score and len(selected) + 1 >= len(np.unique(labels)):
                break
            edge = remaining.pop(best_index)
            selected.append(edge)
            selected.sort()
            best_score = scores[best_index]
        return np.asarray(selected)

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.digitize(values, self.bin_edges_)


def make_normalizer(method: Literal["log", "yeo-johnson", "quantile-normal"], n_samples: int = 1000):
    if method == "log":
        return LogShiftTransformer()
    if method == "yeo-johnson":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    if method == "quantile-normal":
        return QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, n_samples),
                                   random_state=42)
    raise ValueError(f"未知正态化方法: {method}")


@dataclass
class OutlierResult:
    mask: pd.DataFrame
    row_mask: pd.Series
    lower: pd.Series | None = None
    upper: pd.Series | None = None


def detect_outliers(frame: pd.DataFrame, method: Literal["zscore", "iqr", "mad", "lof"] = "iqr",
                    threshold: float = 3.0, contamination: float | str = "auto") -> OutlierResult:
    """返回单元格及行级异常标记；LOF 只提供行级标记。"""
    numeric = frame.astype(float)
    if method == "zscore":
        center, scale = numeric.mean(), numeric.std(ddof=0).replace(0, np.nan)
        mask = numeric.sub(center).div(scale).abs().gt(threshold).fillna(False)
        return OutlierResult(mask, mask.any(axis=1), center - threshold * scale, center + threshold * scale)
    if method == "iqr":
        q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        mask = numeric.lt(lower) | numeric.gt(upper)
        return OutlierResult(mask, mask.any(axis=1), lower, upper)
    if method == "mad":
        median = numeric.median()
        raw_mad = numeric.sub(median).abs().median()
        mad = raw_mad.replace(0, np.nan)
        deviations = numeric.sub(median)
        score = 0.6745 * deviations.div(mad)
        mask = score.abs().gt(threshold).fillna(False)
        zero_mad = raw_mad.eq(0)
        mask.loc[:, zero_mad] = deviations.loc[:, zero_mad].ne(0)
        bound = (threshold * mad / 0.6745).fillna(0)
        return OutlierResult(mask, mask.any(axis=1), median - bound, median + bound)
    if method == "lof":
        if len(frame) < 3:
            raise ValueError("LOF 至少需要 3 行数据")
        complete = SimpleImputer(strategy="median").fit_transform(numeric)
        prediction = LocalOutlierFactor(n_neighbors=min(20, len(frame) - 1), contamination=contamination).fit_predict(complete)
        row_mask = pd.Series(prediction == -1, index=frame.index)
        return OutlierResult(pd.DataFrame(False, index=frame.index, columns=frame.columns), row_mask)
    raise ValueError(f"未知异常检测方法: {method}")


def handle_outliers(frame: pd.DataFrame, result: OutlierResult,
                    action: Literal["remove", "clip", "nan"] = "clip") -> pd.DataFrame:
    if action == "remove":
        return frame.loc[~result.row_mask].copy()
    if action == "nan":
        return frame.mask(result.mask)
    if action == "clip":
        if result.lower is None or result.upper is None:
            raise ValueError("LOF 没有逐列边界，不能执行 clip；请使用 remove")
        return frame.clip(lower=result.lower, upper=result.upper, axis="columns")
    raise ValueError(f"未知异常值处理动作: {action}")
