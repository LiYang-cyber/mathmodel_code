"""删除、均值、随机样本和模型填补模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.io import load_table
from mathmodel.preprocessing import drop_missing, make_imputer


def process_missing(data: str, output: str, method: str, columns: list[str] | None,
                    axis: str = "rows", threshold: float = 0.0) -> Path:
    frame = load_table(repo_path(data))
    selected = columns or frame.select_dtypes(include="number").columns.tolist()
    out = repo_path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if method == "drop":
        result = drop_missing(frame, axis=axis, threshold=threshold)
    else:
        imputer = make_imputer(method)
        result = frame.copy()
        result[selected] = imputer.fit_transform(frame[selected])
        joblib.dump(imputer, out.with_suffix(".joblib"))
    result.to_csv(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="处理缺失值")
    parser.add_argument("data")
    parser.add_argument("--method", choices=["drop", "mean", "random", "model"], required=True)
    parser.add_argument("--columns", nargs="*")
    parser.add_argument("--axis", choices=["rows", "columns"], default="rows")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output", default="data/processed/imputed.csv")
    args = parser.parse_args()
    print(process_missing(args.data, args.output, args.method, args.columns, args.axis, args.threshold).resolve())


if __name__ == "__main__":
    main()
