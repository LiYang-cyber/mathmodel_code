"""训练仓库内适配的 Mamba 系列时序模型。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if str(Path(path or ".").resolve()) != SCRIPT_DIR]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _bootstrap import repo_path

from mathmodel.deep_timeseries.runner import run
from mathmodel.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 S-Mamba、TimePro、STM3 或 MambaSL")
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    print(run(load_config(repo_path(args.config))).resolve())


if __name__ == "__main__":
    main()
