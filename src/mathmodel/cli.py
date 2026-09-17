from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import anomaly, clustering, external_timeseries, tabular, timeseries
from .common import inspect_frame
from .io import load_config, load_table


def _run_deep_timeseries(config):
    try:
        from .deep_timeseries.runner import run
    except ImportError as exc:
        raise ImportError("深度时序依赖未安装，请使用 pixi run -e deep-timeseries 执行") from exc
    return run(config)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="mathmodel", description="数学建模机器学习模板")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="运行配置文件")
    run.add_argument("-c", "--config", required=True)
    run.add_argument("--data", help="覆盖配置中的数据路径")
    run.add_argument("--target", help="覆盖配置中的目标列")
    inspect = commands.add_parser("inspect", help="检查表格数据")
    inspect.add_argument("--data", required=True)
    inspect.add_argument("--target")
    inspect.add_argument("--sheet-name", default=0)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "inspect":
        result = inspect_frame(load_table(args.data, args.sheet_name), args.target)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    config = load_config(args.config)
    if args.data:
        config["data"] = args.data
    if args.target:
        config["target"] = args.target
    runners = {
        "classification": tabular.run, "regression": tabular.run,
        "clustering": clustering.run, "anomaly": anomaly.run, "timeseries": timeseries.run,
        "external_timeseries": external_timeseries.run,
        "deep_timeseries": _run_deep_timeseries,
    }
    if config["task"] not in runners:
        raise ValueError(f"不支持的任务: {config['task']}")
    output = runners[config["task"]](config)
    print(f"运行完成: {Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
