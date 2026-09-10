"""GM(1,1) 灰色预测模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _bootstrap import repo_path

from mathmodel.common import save_figure
from mathmodel.forecasting import GM11
from mathmodel.io import load_table


def run_gm11(data: str, target: str, horizon: int, output: str) -> Path:
    frame = load_table(repo_path(data)).dropna(subset=[target])
    model = GM11().fit(frame[target])
    prediction = model.predict(horizon)
    out = repo_path(output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"step": range(1, horizon + 1), "prediction": prediction}).to_csv(out / "forecast.csv", index=False)
    pd.DataFrame([{"a": model.a_, "b": model.b_, "posterior_error_ratio": model.posterior_error_ratio()}]).to_csv(out / "diagnostics.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(frame)), frame[target], marker="o", label="Observed")
    ax.plot(range(len(frame), len(frame) + horizon), prediction, marker="o", label="GM(1,1)")
    ax.set(xlabel="Step", ylabel=target, title="GM(1,1) forecast")
    ax.legend()
    save_figure(fig, out, "grey_forecast")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 GM(1,1) 灰色预测")
    parser.add_argument("data")
    parser.add_argument("--target", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--output", default="outputs/gm11")
    args = parser.parse_args()
    print(run_gm11(args.data, args.target, args.horizon, args.output).resolve())


if __name__ == "__main__":
    main()

