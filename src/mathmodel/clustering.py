from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import prepare_output, save_figure, save_manifest
from .io import load_table


def run(config: dict[str, Any]) -> Path:
    frame = load_table(config["data"], config.get("sheet_name", 0))
    features = config.get("features") or frame.select_dtypes(include="number").columns.tolist()
    X = frame[features]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.columns.difference(numeric).tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    Z = prep.fit_transform(X)
    seed, method = int(config.get("random_state", 42)), config.get("method", "kmeans")
    if method == "kmeans":
        model = KMeans(n_clusters=int(config.get("n_clusters", 3)), random_state=seed, n_init=20)
        labels = model.fit_predict(Z)
    elif method == "gmm":
        model = GaussianMixture(n_components=int(config.get("n_clusters", 3)), random_state=seed)
        labels = model.fit_predict(Z)
    elif method == "dbscan":
        model = DBSCAN(eps=float(config.get("eps", .5)), min_samples=int(config.get("min_samples", 5)))
        labels = model.fit_predict(Z)
    else:
        raise ValueError(f"未知聚类方法: {method}")
    valid = labels != -1
    score = silhouette_score(Z[valid], labels[valid]) if len(set(labels[valid])) > 1 else None
    projection = PCA(n_components=2, random_state=seed).fit_transform(Z)
    result = frame.copy()
    result["cluster"] = labels
    output = prepare_output(config)
    result.to_csv(output / "clusters.csv", index=False)
    pd.DataFrame([{"method": method, "silhouette": score, "clusters": len(set(labels)) - (-1 in labels)}]).to_csv(output / "metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="tab10", alpha=.75)
    ax.set(xlabel="PC1", ylabel="PC2", title=f"Clusters: {method}")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    save_figure(fig, output, "cluster_pca")
    joblib.dump({"preprocessor": prep, "model": model}, output / "cluster_model.joblib")
    save_manifest(config, output, {"silhouette": score})
    return output

