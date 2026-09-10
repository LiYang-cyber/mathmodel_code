"""CSV/Excel/Parquet 统一加载模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.io import load_table


def load_data(path: str, sheet_name: str | int = 0):
    return load_table(repo_path(path), sheet_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="加载并预览表格数据")
    parser.add_argument("data")
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    frame = load_data(args.data, args.sheet_name)
    print(f"shape={frame.shape}")
    print(frame.head(args.rows).to_string())


if __name__ == "__main__":
    main()

