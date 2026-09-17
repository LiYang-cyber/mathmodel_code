from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .common import prepare_output, save_figure, save_manifest
from .io import load_table


def statistical_anomalies(values, method: str = "zscore", *, threshold: float = 3.0,
                          window: int = 7) -> pd.DataFrame:
    """Return point-wise scores and flags for 3σ/Z-score, IQR or Hampel detection."""
    series = pd.Series(values, dtype=float)
    if method in {"3sigma", "zscore"}:
        scale = series.std(ddof=0)
        score = (series - series.mean()).abs() / (scale if scale > 0 else 1)
        flag = score > threshold
    elif method in {"boxplot", "iqr"}:
        q1, q3 = series.quantile([0.25, 0.75])
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        score = np.maximum((lower - series) / max(spread, 1e-12),
                           (series - upper) / max(spread, 1e-12)).clip(lower=0)
        flag = (series < lower) | (series > upper)
    elif method == "hampel":
        median = series.rolling(window, center=True, min_periods=1).median()
        mad = (series - median).abs().rolling(window, center=True, min_periods=1).median()
        score = (series - median).abs() / (1.4826 * mad).replace(0, np.nan)
        score = score.fillna(0)
        flag = score > threshold
    else:
        raise ValueError(f"未知统计异常检测方法: {method}")
    return pd.DataFrame({"score": score, "is_anomaly": flag.astype(bool)}, index=series.index)


def detect_multivariate(
    frame: pd.DataFrame,
    method: str = "isolation_forest",
    *,
    contamination: float | str = "auto",
    n_neighbors: int = 20,
    eps: float = 0.5,
    min_samples: int = 5,
    nu: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Unified KNN/LOF/DBSCAN/OCSVM/Isolation Forest detector output."""
    values = SimpleImputer(strategy="median").fit_transform(frame.astype(float))
    scaled = StandardScaler().fit_transform(values)
    if method == "knn":
        k = min(n_neighbors, len(frame) - 1)
        distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(scaled).kneighbors(scaled)
        score = distances[:, -1]
        rate = 0.1 if contamination == "auto" else float(contamination)
        flag = score >= np.quantile(score, 1 - rate)
    elif method == "lof":
        model = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(frame) - 1),
                                   contamination=contamination)
        flag = model.fit_predict(scaled) == -1
        score = -model.negative_outlier_factor_
    elif method == "dbscan":
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
        flag = labels == -1
        score = flag.astype(float)
    elif method == "one_class_svm":
        model = OneClassSVM(nu=nu, gamma="scale").fit(scaled)
        flag = model.predict(scaled) == -1
        score = -model.decision_function(scaled).ravel()
    elif method == "isolation_forest":
        model = IsolationForest(contamination=contamination, random_state=random_state).fit(scaled)
        flag = model.predict(scaled) == -1
        score = -model.decision_function(scaled)
    else:
        raise ValueError(f"未知多变量异常检测方法: {method}")
    return pd.DataFrame({"score": score, "is_anomaly": flag}, index=frame.index)


def residual_anomalies(values, method: str = "arima", *, threshold: float = 3.0,
                       order: tuple[int, int, int] = (1, 0, 0)) -> pd.DataFrame:
    """Detect anomalies from ARIMA or local-level Kalman standardized residuals."""
    series = pd.Series(values, dtype=float)
    if method == "arima":
        from statsmodels.tsa.arima.model import ARIMA

        residual = pd.Series(ARIMA(series, order=order).fit().resid, index=series.index)
    elif method == "kalman":
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        fitted = UnobservedComponents(series, level="local level").fit(disp=False)
        residual = pd.Series(fitted.resid, index=series.index)
    else:
        raise ValueError("method 必须为 arima 或 kalman")
    score = (residual - residual.mean()).abs() / max(residual.std(ddof=0), 1e-12)
    return pd.DataFrame({"score": score, "is_anomaly": score > threshold}, index=series.index)


def decomposition_anomalies(values, method: str = "emd", *, threshold: float = 3.0,
                            **kwargs) -> pd.DataFrame:
    """Detect anomalies in EMD/VMD residual (or highest-frequency reconstruction error)."""
    from .decomposition import emd_decompose, vmd_decompose

    if method == "emd":
        parts = emd_decompose(values, **kwargs)
    elif method == "vmd":
        parts = vmd_decompose(values, **kwargs)
    else:
        raise ValueError("method 必须为 emd 或 vmd")
    reconstructed = parts.drop(columns="residual", errors="ignore").sum(axis=1)
    residual = pd.Series(values, dtype=float) - reconstructed
    return statistical_anomalies(residual, "hampel", threshold=threshold)


def run(config: dict[str, Any]) -> Path:
    frame = load_table(config["data"], config.get("sheet_name", 0))
    label_col = config.get("label")
    excluded = [label_col] if label_col else []
    features = config.get("features") or frame.drop(columns=excluded).select_dtypes(include="number").columns.tolist()
    X = frame[features]
    seed = int(config.get("random_state", 42))
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ("model", IsolationForest(contamination=config.get("contamination", "auto"), random_state=seed)),
    ])
    raw = model.fit_predict(X)
    score = -model.decision_function(X)
    predicted = (raw == -1).astype(int)
    result = frame.copy()
    result["anomaly_score"] = score
    result["is_anomaly"] = predicted
    metrics: dict[str, Any] = {"anomalies": int(predicted.sum()), "anomaly_rate": float(predicted.mean())}
    if label_col:
        truth = frame[label_col].astype(int)
        metrics.update({"f1": f1_score(truth, predicted), "roc_auc": roc_auc_score(truth, score)})
    output = prepare_output(config)
    result.to_csv(output / "anomalies.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(score, bins=30, color="#4472C4", alpha=.8)
    ax.set(xlabel="Anomaly score", ylabel="Count", title="Anomaly score distribution")
    save_figure(fig, output, "anomaly_scores")
    joblib.dump(model, output / "anomaly_model.joblib")
    save_manifest(config, output, metrics)
    return output
