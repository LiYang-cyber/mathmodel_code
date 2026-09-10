"""论文常用绘图函数：真实-预测、残差、特征重要性。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.common import save_figure


def plot_predictions(frame: pd.DataFrame, output: Path, actual="actual", predicted="predicted") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(frame[actual], frame[predicted], alpha=.65, color="#4472C4")
    low = min(frame[actual].min(), frame[predicted].min())
    high = max(frame[actual].max(), frame[predicted].max())
    axes[0].plot([low, high], [low, high], "--", color="black")
    axes[0].set(xlabel="Actual", ylabel="Predicted", title="Actual vs predicted")
    residual = frame[actual] - frame[predicted]
    axes[1].scatter(frame[predicted], residual, alpha=.65, color="#ED7D31")
    axes[1].axhline(0, linestyle="--", color="black")
    axes[1].set(xlabel="Predicted", ylabel="Residual", title="Residual diagnostic")
    save_figure(fig, output, "prediction_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser(description="从预测 CSV 生成论文图")
    parser.add_argument("predictions")
    parser.add_argument("--actual", default="actual")
    parser.add_argument("--predicted", default="predicted")
    parser.add_argument("--output", default="outputs/figures")
    args = parser.parse_args()
    output = repo_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plot_predictions(pd.read_csv(repo_path(args.predictions)), output, args.actual, args.predicted)
    print(output.resolve())


if __name__ == "__main__":
    main()

