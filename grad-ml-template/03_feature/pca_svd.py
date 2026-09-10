"""PCA/SVD 降维模板：输出主成分、载荷、贡献率和矢量图。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.common import save_figure
from mathmodel.io import load_table


def run_pca(data: str, output: str, features: list[str] | None = None, variance: float = 0.9) -> Path:
    if not 0 < variance <= 1:
        raise ValueError("variance 必须位于 (0, 1]")
    frame = load_table(repo_path(data))
    features = features or frame.select_dtypes(include="number").columns.tolist()
    if len(features) < 2:
        raise ValueError("PCA 至少需要两个数值特征")
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=variance, svd_solver="full")),
    ])
    transformed = pipeline.fit_transform(frame[features])
    model = pipeline.named_steps["pca"]
    component_names = [f"PC{i}" for i in range(1, transformed.shape[1] + 1)]
    out = repo_path(output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(transformed, columns=component_names, index=frame.index).to_csv(out / "principal_components.csv", index_label="row_index")
    pd.DataFrame(model.components_.T, index=features, columns=component_names).to_csv(out / "loadings.csv", index_label="feature")
    pd.DataFrame({"component": component_names, "explained_ratio": model.explained_variance_ratio_,
                  "cumulative_ratio": np.cumsum(model.explained_variance_ratio_)}).to_csv(out / "explained_variance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(component_names, model.explained_variance_ratio_, label="Individual")
    ax.plot(component_names, np.cumsum(model.explained_variance_ratio_), marker="o", color="#C00000", label="Cumulative")
    ax.axhline(variance, color="black", linestyle="--", linewidth=1)
    ax.set(ylabel="Explained variance ratio", title="PCA explained variance", ylim=(0, 1.05))
    ax.legend()
    save_figure(fig, out, "pca_explained_variance")
    joblib.dump(pipeline, out / "pca_pipeline.joblib")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="执行 PCA/SVD 降维")
    parser.add_argument("data")
    parser.add_argument("--features", nargs="*")
    parser.add_argument("--variance", type=float, default=.9)
    parser.add_argument("--output", default="outputs/pca")
    args = parser.parse_args()
    print(run_pca(args.data, args.output, args.features, args.variance).resolve())


if __name__ == "__main__":
    main()

