"""运行独立环境中的研究型时序模型。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if str(Path(path or ".").resolve()) != SCRIPT_DIR]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _bootstrap import repo_path

from mathmodel.external_timeseries import describe_models, run
from mathmodel.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 S-Mamba、TimePro、STM3 或 MambaSL")
    parser.add_argument("-c", "--config", help="外部模型 YAML 配置")
    parser.add_argument("--list", action="store_true", help="列出已注册的模型")
    args = parser.parse_args()
    if args.list:
        print(describe_models())
        return
    if not args.config:
        parser.error("请提供 --config，或使用 --list")
    print(run(load_config(repo_path(args.config))).resolve())


if __name__ == "__main__":
    main()
