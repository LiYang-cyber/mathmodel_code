"""常用表格特征工程：日期拆分、偏态变换、交互项。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.io import load_table


def build_features(
    frame: pd.DataFrame,
    date_columns: list[str] | None = None,
    log_columns: list[str] | None = None,
    interactions: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    for column in date_columns or []:
        values = pd.to_datetime(result[column], errors="coerce")
        result[f"{column}_year"] = values.dt.year
        result[f"{column}_month"] = values.dt.month
        result[f"{column}_day"] = values.dt.day
        result[f"{column}_weekday"] = values.dt.dayofweek
        result.drop(columns=column, inplace=True)
    for column in log_columns or []:
        minimum = result[column].min(skipna=True)
        shift = 1 - minimum if minimum <= 0 else 0
        result[f"{column}_log1p"] = np.log1p(result[column] + shift)
    for left, right in interactions or []:
        result[f"{left}_x_{right}"] = result[left] * result[right]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="生成常用衍生特征")
    parser.add_argument("data")
    parser.add_argument("--output", default="data/processed/features.csv")
    parser.add_argument("--date", nargs="*", default=[])
    parser.add_argument("--log", nargs="*", default=[])
    parser.add_argument("--interaction", nargs=2, action="append", default=[])
    args = parser.parse_args()
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_features(load_table(repo_path(args.data)), args.date, args.log, args.interaction).to_csv(output, index=False)
    print(output.resolve())


if __name__ == "__main__":
    main()

