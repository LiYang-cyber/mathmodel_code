"""自动 EDA：数据质量表、描述统计和目标分布。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.common import inspect_frame, save_figure
from mathmodel.io import load_table, write_json


def run_eda(data: str, output: str, target: str | None = None) -> Path:
    frame = load_table(repo_path(data))
    out = repo_path(output)
    out.mkdir(parents=True, exist_ok=True)
    summary = inspect_frame(frame, target)
    write_json(summary, out / "data_quality.json")
    frame.describe(include="all").transpose().to_csv(out / "descriptive_statistics.csv")
    missing = frame.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(7, len(missing) * 0.35), 4))
    missing.plot.bar(ax=ax, color="#4472C4")
    ax.set(ylabel="Missing ratio", title="Missing values by column", ylim=(0, 1))
    save_figure(fig, out, "missing_values")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="生成探索性数据分析报告")
    parser.add_argument("data")
    parser.add_argument("--target")
    parser.add_argument("--output", default="outputs/eda")
    args = parser.parse_args()
    output = run_eda(args.data, args.output, args.target)
    print(json.dumps({"output": str(output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

