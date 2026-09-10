"""熵权 TOPSIS 综合评价模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _bootstrap import repo_path

from mathmodel.common import save_figure
from mathmodel.evaluation import entropy_weights, orient_indicators, topsis
from mathmodel.io import load_table


def run_entropy_topsis(data: str, output: str, indicators: list[str], negative: list[str] | None = None) -> Path:
    frame = load_table(repo_path(data))
    oriented = orient_indicators(frame[indicators], negative=negative)
    weights = entropy_weights(oriented)
    ranking = topsis(oriented, weights)
    out = repo_path(output)
    out.mkdir(parents=True, exist_ok=True)
    weights.to_csv(out / "weights.csv", header=True)
    frame.join(ranking).sort_values("rank").to_csv(out / "ranking.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ranking.sort_values("score")["score"].plot.barh(ax=ax, color="#4472C4")
    ax.set(xlabel="TOPSIS score", ylabel="Alternative", title="Entropy-weight TOPSIS ranking")
    save_figure(fig, out, "topsis_ranking")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="运行熵权 TOPSIS")
    parser.add_argument("data")
    parser.add_argument("--indicators", nargs="+", required=True)
    parser.add_argument("--negative", nargs="*", default=[])
    parser.add_argument("--output", default="outputs/entropy_topsis")
    args = parser.parse_args()
    print(run_entropy_topsis(args.data, args.output, args.indicators, args.negative).resolve())


if __name__ == "__main__":
    main()

