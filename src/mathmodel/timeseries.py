from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .common import prepare_output, save_figure, save_manifest
from .io import load_table


def _supervised(values: pd.Series, lags: list[int]) -> pd.DataFrame:
    data = {f"lag_{lag}": values.shift(lag) for lag in lags}
    data["target"] = values
    return pd.DataFrame(data).dropna()


def run(config: dict[str, Any]) -> Path:
    frame = load_table(config["data"], config.get("sheet_name", 0))
    time_col, target = config["time_column"], config["target"]
    frame[time_col] = pd.to_datetime(frame[time_col])
    frame = frame.sort_values(time_col).dropna(subset=[target])
    lags = sorted({int(x) for x in config.get("lags", [1, 2, 3, 7])})
    supervised = _supervised(frame[target], lags)
    horizon = int(config.get("test_horizon", max(1, len(supervised) // 5)))
    train, test = supervised.iloc[:-horizon], supervised.iloc[-horizon:]
    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]
    seed = int(config.get("random_state", 42))
    candidates = {
        "ridge_lag": Ridge(alpha=float(config.get("ridge_alpha", 1.0))),
        "random_forest_lag": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
    }
    rows, predictions, fitted = [], {}, {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name], predictions[name] = model, pred
        rows.append({"model": name, "rmse": mean_squared_error(y_test, pred) ** .5,
                     "mae": mean_absolute_error(y_test, pred)})
    seasonal = int(config.get("seasonal_period", 1))
    naive = frame[target].shift(seasonal).loc[y_test.index].to_numpy()
    if not np.isnan(naive).any():
        predictions["seasonal_naive"] = naive
        rows.append({"model": "seasonal_naive", "rmse": mean_squared_error(y_test, naive) ** .5,
                     "mae": mean_absolute_error(y_test, naive)})
    metrics = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    best_name = metrics.loc[0, "model"]
    pred = predictions[best_name]
    output = prepare_output(config)
    dates = frame.loc[y_test.index, time_col]
    pd.DataFrame({time_col: dates.to_numpy(), "actual": y_test.to_numpy(), "predicted": pred}).to_csv(output / "forecast.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dates, y_test, label="Actual", marker="o", ms=3)
    ax.plot(dates, pred, label=f"Predicted ({best_name})", marker="o", ms=3)
    ax.set(xlabel="Time", ylabel=target, title="Holdout forecast")
    ax.legend()
    save_figure(fig, output, "forecast")
    if best_name in fitted:
        joblib.dump({"model": fitted[best_name], "lags": lags}, output / "forecast_model.joblib")
    save_manifest(config, output, {"best_model": best_name, "test_horizon": horizon})
    return output
