"""Pearson、Spearman、Kendall 与灰色关联分析模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.common import save_figure
from mathmodel.evaluation import grey_relational_grade
from mathmodel.io import load_table


def run_correlation(data: str, output: str, features: list[str] | None = None, reference: str | None = None) -> Path:
    frame = load_table(repo_path(data))
    features = features or frame.select_dtypes(include="number").columns.tolist()
    values = frame[features]
    out = repo_path(output)
    out.mkdir(parents=True, exist_ok=True)
    for method in ("pearson", "spearman", "kendall"):
        values.corr(method=method).to_csv(out / f"{method}_correlation.csv")
    if reference:
        grey_relational_grade(values, reference).to_csv(out / "grey_relational_grade.csv")
    matrix = values.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(max(6, len(features) * .55), max(5, len(features) * .5)))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(features)), features, rotation=60, ha="right")
    ax.set_yticks(range(len(features)), features)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    save_figure(fig, out, "correlation_heatmap")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="计算相关性与灰色关联度")
    parser.add_argument("data")
    parser.add_argument("--features", nargs="*")
    parser.add_argument("--reference")
    parser.add_argument("--output", default="outputs/correlation")
    args = parser.parse_args()
    print(run_correlation(args.data, args.output, args.features, args.reference).resolve())


if __name__ == "__main__":
    main()

