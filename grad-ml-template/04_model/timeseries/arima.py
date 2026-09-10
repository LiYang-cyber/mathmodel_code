"""ARIMA 时序预测与留出集评估模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _bootstrap import repo_path

from mathmodel.common import save_figure
from mathmodel.io import load_table


def run_arima(data: str, time_column: str, target: str, output: str, order=(1, 1, 1), horizon=14) -> Path:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError("statsmodels 未安装，请使用 pixi run -e full 执行") from exc
    frame = load_table(repo_path(data))
    frame[time_column] = pd.to_datetime(frame[time_column])
    frame = frame.sort_values(time_column).dropna(subset=[target])
    train, test = frame.iloc[:-horizon], frame.iloc[-horizon:]
    fitted = ARIMA(train[target], order=order).fit()
    forecast = fitted.forecast(horizon).to_numpy()
    out = repo_path(output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({time_column: test[time_column], "actual": test[target], "predicted": forecast}).to_csv(out / "forecast.csv", index=False)
    pd.DataFrame([{"rmse": mean_squared_error(test[target], forecast) ** .5,
                   "mae": mean_absolute_error(test[target], forecast), "order": str(order)}]).to_csv(out / "metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(test[time_column], test[target], label="Actual")
    ax.plot(test[time_column], forecast, label="ARIMA forecast")
    ax.legend()
    save_figure(fig, out, "arima_forecast")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 ARIMA")
    parser.add_argument("data")
    parser.add_argument("--time", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--order", nargs=3, type=int, default=[1, 1, 1])
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--output", default="outputs/arima")
    args = parser.parse_args()
    print(run_arima(args.data, args.time, args.target, args.output, tuple(args.order), args.horizon).resolve())


if __name__ == "__main__":
    main()
