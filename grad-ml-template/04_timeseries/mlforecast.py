"""MLForecast 滞后与滚动特征预测模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if str(Path(path or ".").resolve()) != SCRIPT_DIR]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.io import load_table


def run_mlforecast(data: str, time_column: str, target: str, output: str, horizon=14, frequency="D") -> Path:
    try:
        from mlforecast import MLForecast
    except ImportError as exc:
        raise ImportError("MLForecast 未安装，请使用 pixi run -e full 执行") from exc
    frame = load_table(repo_path(data))[[time_column, target]].rename(columns={time_column: "ds", target: "y"})
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame["unique_id"] = "series_1"
    model = MLForecast(models={"random_forest": RandomForestRegressor(n_estimators=300, random_state=42)},
                       freq=frequency, lags=[1, 2, 3, 7, 14])
    model.fit(frame)
    forecast = model.predict(horizon)
    out = repo_path(output)
    out.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(out / "future_forecast.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 MLForecast")
    parser.add_argument("data")
    parser.add_argument("--time", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--frequency", default="D")
    parser.add_argument("--output", default="outputs/mlforecast")
    args = parser.parse_args()
    print(run_mlforecast(args.data, args.time, args.target, args.output, args.horizon, args.frequency).resolve())


if __name__ == "__main__":
    main()

