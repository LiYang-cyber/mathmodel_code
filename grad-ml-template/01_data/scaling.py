"""Z-score、Min-Max、小数定标、Logistic 与中心化转换模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.io import load_table
from mathmodel.preprocessing import make_scaler


def scale_columns(data: str, columns: list[str], method: str, output: str) -> Path:
    frame = load_table(repo_path(data))
    scaler = make_scaler(method)
    result = frame.copy()
    result[columns] = scaler.fit_transform(frame[columns])
    out = repo_path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    joblib.dump(scaler, out.with_suffix(".joblib"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="缩放或中心化数值特征")
    parser.add_argument("data")
    parser.add_argument("--columns", nargs="+", required=True)
    parser.add_argument("--method", choices=["zscore", "minmax", "decimal", "logistic", "center"], required=True)
    parser.add_argument("--output", default="data/processed/scaled.csv")
    args = parser.parse_args()
    print(scale_columns(args.data, args.columns, args.method, args.output).resolve())


if __name__ == "__main__":
    main()
