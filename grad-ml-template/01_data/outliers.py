"""统计法与局部离群因子异常检测、删除、截尾和置空模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.io import load_table
from mathmodel.preprocessing import detect_outliers, handle_outliers


def process_outliers(data: str, columns: list[str], method: str, action: str, output: str,
                     threshold: float = 3.0, contamination: float | str = "auto") -> Path:
    frame = load_table(repo_path(data))
    detection = detect_outliers(frame[columns], method=method, threshold=threshold,
                                contamination=contamination)
    result = frame.copy()
    if action == "remove":
        result = frame.loc[~detection.row_mask].copy()
    else:
        result[columns] = handle_outliers(frame[columns], detection, action=action)
    out = repo_path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    pd.DataFrame({"is_outlier": detection.row_mask}).to_csv(out.with_name(f"{out.stem}_flags.csv"), index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="检测并处理异常值")
    parser.add_argument("data")
    parser.add_argument("--columns", nargs="+", required=True)
    parser.add_argument("--method", choices=["zscore", "iqr", "mad", "lof"], default="iqr")
    parser.add_argument("--action", choices=["remove", "clip", "nan"], default="clip")
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--contamination", default="auto")
    parser.add_argument("--output", default="data/processed/outliers_processed.csv")
    args = parser.parse_args()
    contamination = args.contamination if args.contamination == "auto" else float(args.contamination)
    print(process_outliers(args.data, args.columns, args.method, args.action, args.output,
                           args.threshold, contamination).resolve())


if __name__ == "__main__":
    main()

