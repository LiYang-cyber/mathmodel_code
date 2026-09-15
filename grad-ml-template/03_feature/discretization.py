"""等距、等频、K-Means、信息增益、ChiMerge 与 CAIM 离散化模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.io import load_table
from mathmodel.preprocessing import SupervisedDiscretizer, make_unsupervised_discretizer


def discretize(data: str, column: str, method: str, bins: int, output: str,
               target: str | None = None) -> Path:
    frame = load_table(repo_path(data))
    if method in {"uniform", "quantile", "kmeans"}:
        transformer = make_unsupervised_discretizer(method, bins)
        transformed = transformer.fit_transform(frame[[column]]).ravel()
    else:
        if target is None:
            raise ValueError(f"{method} 是监督离散化，必须指定 --target")
        transformer = SupervisedDiscretizer(method=method, max_bins=bins)
        transformed = transformer.fit_transform(frame[[column]], frame[target]).ravel()
    result = frame.copy()
    result[f"{column}_bin"] = transformed
    out = repo_path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    joblib.dump(transformer, out.with_suffix(".joblib"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="连续特征离散化")
    parser.add_argument("data")
    parser.add_argument("--column", required=True)
    parser.add_argument("--method", choices=["uniform", "quantile", "kmeans", "information_gain", "chimerge", "caim"], required=True)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--target")
    parser.add_argument("--output", default="data/processed/discretized.csv")
    args = parser.parse_args()
    print(discretize(args.data, args.column, args.method, args.bins, args.output, args.target).resolve())


if __name__ == "__main__":
    main()
