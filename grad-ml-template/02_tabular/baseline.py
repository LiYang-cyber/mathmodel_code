"""表格分类/回归基线模型比较。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.io import load_config
from mathmodel.tabular import run


def main() -> None:
    parser = argparse.ArgumentParser(description="运行表格任务 baseline")
    parser.add_argument("--config", "-c", default="configs/classification.yaml")
    parser.add_argument("--data")
    parser.add_argument("--target")
    args = parser.parse_args()
    config = load_config(repo_path(args.config))
    if args.data:
        config["data"] = str(repo_path(args.data))
    if args.target:
        config["target"] = args.target
    print(run(config).resolve())


if __name__ == "__main__":
    main()

