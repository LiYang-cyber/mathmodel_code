"""XGBoost 分类/回归模板；需要 Pixi full 环境。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if str(Path(path or ".").resolve()) != SCRIPT_DIR]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.io import load_config
from mathmodel.tabular import run


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 XGBoost")
    parser.add_argument("--config", "-c", default="configs/classification.yaml")
    args = parser.parse_args()
    config = load_config(repo_path(args.config))
    config["models"] = ["xgboost"]
    config["run_name"] = config.get("run_name", config["task"]) + "_xgboost"
    print(run(config).resolve())


if __name__ == "__main__":
    main()

