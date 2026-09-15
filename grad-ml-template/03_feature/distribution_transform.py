"""对数、Yeo-Johnson 与分位数正态转换模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.io import load_table
from mathmodel.preprocessing import make_normalizer


def transform_distribution(data: str, columns: list[str], method: str, output: str) -> Path:
    frame = load_table(repo_path(data))
    transformer = make_normalizer(method, n_samples=len(frame))
    result = frame.copy()
    result[columns] = transformer.fit_transform(frame[columns])
    out = repo_path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    joblib.dump(transformer, out.with_suffix(".joblib"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="转换偏态或非正态特征")
    parser.add_argument("data")
    parser.add_argument("--columns", nargs="+", required=True)
    parser.add_argument("--method", choices=["log", "yeo-johnson", "quantile-normal"], required=True)
    parser.add_argument("--output", default="data/processed/distribution_transformed.csv")
    args = parser.parse_args()
    print(transform_distribution(args.data, args.columns, args.method, args.output).resolve())


if __name__ == "__main__":
    main()
