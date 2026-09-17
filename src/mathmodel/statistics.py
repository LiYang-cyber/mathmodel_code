"""Classical statistical-analysis building blocks.

The functions in this module keep tabular inputs and labelled outputs so they can
be used directly in competition reports.  Estimators themselves are delegated
to SciPy and scikit-learn rather than reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RegressionResult:
    model: LinearRegression
    coefficients: pd.DataFrame
    fitted: pd.Series
    residuals: pd.Series
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    f_pvalue: float


def multiple_regression(X: pd.DataFrame, y: pd.Series) -> RegressionResult:
    """Fit OLS and return coefficients, standard errors and classical tests."""
    X = _numeric_frame(X, "X")
    y = pd.Series(y, index=X.index, dtype=float, name=getattr(y, "name", None) or "y")
    if len(X) <= X.shape[1] + 1:
        raise ValueError("样本数必须大于特征数 + 1")
    model = LinearRegression().fit(X, y)
    fitted = pd.Series(model.predict(X), index=X.index, name="fitted")
    residuals = (y - fitted).rename("residual")
    n = len(X)
    design = np.column_stack([np.ones(n), X.to_numpy()])
    rank = np.linalg.matrix_rank(design)
    dof = n - rank
    if dof <= 0:
        raise ValueError("设计矩阵没有足够的残差自由度")
    sse = float(residuals @ residuals)
    covariance = (sse / dof) * np.linalg.pinv(design.T @ design)
    estimates = np.r_[model.intercept_, model.coef_]
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_value = np.divide(estimates, standard_error, out=np.full_like(estimates, np.nan), where=standard_error > 0)
    p_value = 2 * stats.t.sf(np.abs(t_value), dof)
    names = ["intercept", *X.columns]
    coefficients = pd.DataFrame({"estimate": estimates, "std_error": standard_error,
                                 "t_value": t_value, "p_value": p_value}, index=names)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else 1.0
    adjusted = 1 - (1 - r2) * (n - 1) / dof
    model_df = rank - 1
    f_value = ((sst - sse) / model_df) / (sse / dof) if model_df and sse > 0 else np.inf
    return RegressionResult(model, coefficients, fitted, residuals, r2, adjusted,
                            float(f_value), float(stats.f.sf(f_value, model_df, dof)))


def stepwise_regression(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    enter: float = 0.05,
    remove: float = 0.10,
    max_steps: int = 100,
) -> tuple[list[str], RegressionResult]:
    """Bidirectional p-value stepwise OLS.

    This is intended as an interpretable exploratory baseline.  Selection and
    final inference on the same data are optimistic; confirm with held-out data.
    """
    if not 0 < enter < remove < 1:
        raise ValueError("需要满足 0 < enter < remove < 1")
    X = _numeric_frame(X, "X")
    selected: list[str] = []
    for _ in range(max_steps):
        changed = False
        excluded = [column for column in X.columns if column not in selected]
        if excluded:
            candidates = {
                column: multiple_regression(X[selected + [column]], y).coefficients.loc[column, "p_value"]
                for column in excluded
            }
            best = min(candidates, key=candidates.get)
            if candidates[best] < enter:
                selected.append(best)
                changed = True
        if len(selected) > 1:
            result = multiple_regression(X[selected], y)
            worst = result.coefficients.drop(index="intercept")["p_value"].idxmax()
            if result.coefficients.loc[worst, "p_value"] > remove:
                selected.remove(worst)
                changed = True
        if not changed:
            break
    if not selected:
        raise ValueError("没有变量满足进入阈值；请调整 enter 或改用正则化回归")
    return selected, multiple_regression(X[selected], y)


def one_way_anova(values: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Return a one-way ANOVA table including effect size eta squared."""
    frame = pd.DataFrame({"value": values, "group": groups}).dropna()
    samples = [part["value"].to_numpy(float) for _, part in frame.groupby("group", sort=False)]
    if len(samples) < 2 or any(len(sample) < 2 for sample in samples):
        raise ValueError("方差分析至少需要两个组，且每组至少两个观测")
    grand = frame["value"].mean()
    ss_between = sum(len(sample) * (sample.mean() - grand) ** 2 for sample in samples)
    ss_within = sum(((sample - sample.mean()) ** 2).sum() for sample in samples)
    df_between, df_within = len(samples) - 1, len(frame) - len(samples)
    ms_between, ms_within = ss_between / df_between, ss_within / df_within
    f_value, p_value = stats.f_oneway(*samples)
    total = ss_between + ss_within
    return pd.DataFrame(
        [
            {"source": "between", "sum_sq": ss_between, "df": df_between,
             "mean_sq": ms_between, "F": f_value, "p_value": p_value,
             "eta_squared": ss_between / total if total else np.nan},
            {"source": "within", "sum_sq": ss_within, "df": df_within,
             "mean_sq": ms_within, "F": np.nan, "p_value": np.nan, "eta_squared": np.nan},
        ]
    ).set_index("source")


def discriminant_classifier(
    method: Literal["distance", "fisher", "bayes"], **params: object
) -> ClassifierMixin:
    """Create distance, Fisher or Gaussian Bayes discriminant estimator."""
    estimators = {
        "distance": NearestCentroid,
        "fisher": LinearDiscriminantAnalysis,
        "bayes": QuadraticDiscriminantAnalysis,
    }
    if method not in estimators:
        raise ValueError(f"未知判别方法: {method}")
    return estimators[method](**params)


def kmeans_cluster(
    X: pd.DataFrame, n_clusters: int, *, random_state: int = 42, n_init: int = 20
) -> tuple[KMeans, pd.Series, float | None]:
    """Standardize data and run efficient Lloyd K-means with a silhouette score."""
    X = _numeric_frame(X, "X")
    values = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state).fit(values)
    labels = pd.Series(model.labels_, index=X.index, name="cluster")
    score = silhouette_score(values, labels) if 1 < labels.nunique() < len(labels) else None
    return model, labels, score


def principal_components(X: pd.DataFrame, n_components=0.95) -> tuple[PCA, pd.DataFrame]:
    X = _numeric_frame(X, "X")
    values = StandardScaler().fit_transform(X)
    model = PCA(n_components=n_components).fit(values)
    scores = pd.DataFrame(model.transform(values), index=X.index,
                          columns=[f"PC{i + 1}" for i in range(model.n_components_)])
    return model, scores


def factor_analysis(
    X: pd.DataFrame, n_components: int, *, rotation: Literal["varimax", "quartimax"] | None = "varimax",
    random_state: int = 42,
) -> tuple[FactorAnalysis, pd.DataFrame, pd.DataFrame]:
    X = _numeric_frame(X, "X")
    values = StandardScaler().fit_transform(X)
    model = FactorAnalysis(n_components=n_components, rotation=rotation,
                           random_state=random_state).fit(values)
    scores = pd.DataFrame(model.transform(values), index=X.index,
                          columns=[f"Factor{i + 1}" for i in range(n_components)])
    loadings = pd.DataFrame(model.components_.T, index=X.columns, columns=scores.columns)
    return model, scores, loadings


def canonical_correlation(
    X: pd.DataFrame, Y: pd.DataFrame, n_components: int = 2
) -> tuple[CCA, pd.DataFrame, pd.DataFrame, pd.Series]:
    X, Y = _numeric_frame(X, "X"), _numeric_frame(Y, "Y")
    if not X.index.equals(Y.index):
        raise ValueError("X 与 Y 的索引必须一致")
    model = CCA(n_components=n_components, scale=True).fit(X, Y)
    x_scores, y_scores = model.transform(X, Y)
    columns = [f"CC{i + 1}" for i in range(n_components)]
    x_frame = pd.DataFrame(x_scores, index=X.index, columns=columns)
    y_frame = pd.DataFrame(y_scores, index=Y.index, columns=columns)
    correlations = pd.Series(
        [np.corrcoef(x_scores[:, i], y_scores[:, i])[0, 1] for i in range(n_components)],
        index=columns, name="canonical_correlation",
    )
    return model, x_frame, y_frame, correlations


def gaussian_bayes_classifier(**params: object) -> GaussianNB:
    """Return Gaussian naive Bayes for high-dimensional conditional-independence baselines."""
    return GaussianNB(**params)


def _numeric_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = pd.DataFrame(frame).astype(float)
    if result.empty or not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name} 必须是非空、无缺失的有限数值表")
    return result
