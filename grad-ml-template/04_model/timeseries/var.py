"""多变量时间序列 VAR 预测模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _bootstrap import repo_path

from mathmodel.common import save_figure
from mathmodel.io import load_table


def run_var(data: str, time_column: str, targets: list[str], horizon: int, output: str, maxlags: int = 8) -> Path:
    try:
        from statsmodels.tsa.api import VAR
    except ImportError as exc:
        raise ImportError("statsmodels 未安装，请使用 pixi run -e full 执行") from exc
    frame = load_table(repo_path(data))
    frame[time_column] = pd.to_datetime(frame[time_column])
    values = frame.sort_values(time_column).set_index(time_column)[targets].dropna()
    frequency = pd.infer_freq(values.index)
    if frequency:
        values = values.asfreq(frequency)
    fitted = VAR(values).fit(maxlags=maxlags, ic="aic")
    prediction = fitted.forecast(values.to_numpy()[-fitted.k_ar:], steps=horizon)
    forecast = pd.DataFrame(prediction, columns=targets)
    forecast.insert(0, "step", range(1, horizon + 1))
    out = repo_path(output)
    out.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(out / "forecast.csv", index=False)
    pd.DataFrame([{"selected_lag": fitted.k_ar, "aic": fitted.aic, "bic": fitted.bic}]).to_csv(out / "diagnostics.csv", index=False)
    fig, axes = plt.subplots(len(targets), 1, figsize=(9, max(4, 3 * len(targets))), squeeze=False)
    history = min(30, len(values))
    for axis, column in zip(axes.flat, targets, strict=True):
        axis.plot(range(-history, 0), values[column].iloc[-history:], label="Observed")
        axis.plot(range(1, horizon + 1), forecast[column], marker="o", label="VAR forecast")
        axis.set(xlabel="Relative step", ylabel=column, title=f"VAR forecast: {column}")
        axis.legend()
    fig.tight_layout()
    save_figure(fig, out, "var_forecast")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 VAR 多变量预测")
    parser.add_argument("data")
    parser.add_argument("--time", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--maxlags", type=int, default=8)
    parser.add_argument("--output", default="outputs/var")
    args = parser.parse_args()
    print(run_var(args.data, args.time, args.targets, args.horizon, args.output, args.maxlags).resolve())


if __name__ == "__main__":
    main()
