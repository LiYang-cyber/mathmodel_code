import numpy as np
import pandas as pd

from mathmodel import anomaly, clustering, tabular, timeseries


def test_all_pipelines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80),
                          "category": np.where(rng.normal(size=80) > 0, "x", "y")})
    frame["target"] = (frame["a"] + frame["b"] > 0).astype(int)
    frame.loc[0, "a"] = np.nan
    path = tmp_path / "table.csv"
    frame.to_csv(path, index=False)
    base = {"data": str(path), "output_dir": "outputs", "random_state": 2}
    out = tabular.run({**base, "task": "classification", "target": "target",
                       "models": ["logistic"], "cv_folds": 3, "run_name": "classification"})
    assert (out / "best_model.joblib").exists()
    out = tabular.run({
        **base,
        "task": "classification",
        "target": "target",
        "models": ["extra_trees"],
        "custom_models": {
            "extra_trees": {
                "class_path": "sklearn.ensemble.ExtraTreesClassifier",
                "params": {"n_estimators": 20, "random_state": 2},
            }
        },
        "numeric_imputer": "random",
        "numeric_scaler": "logistic",
        "categorical_encoder": "ordinal",
        "cv_folds": 3,
        "run_name": "custom_model",
    })
    assert (out / "best_model.joblib").exists()
    out = clustering.run({**base, "task": "clustering", "features": ["a", "b"],
                          "n_clusters": 2, "run_name": "clustering"})
    assert (out / "clusters.csv").exists()
    out = anomaly.run({**base, "task": "anomaly", "features": ["a", "b"],
                       "contamination": .1, "run_name": "anomaly"})
    assert (out / "anomalies.csv").exists()
    dates = pd.date_range("2025-01-01", periods=70)
    ts_path = tmp_path / "ts.csv"
    pd.DataFrame({"date": dates, "value": np.sin(np.arange(70) / 5) + np.arange(70) / 50}).to_csv(ts_path, index=False)
    out = timeseries.run({"task": "timeseries", "data": str(ts_path), "time_column": "date",
                          "target": "value", "lags": [1, 2, 7], "test_horizon": 10,
                          "output_dir": "outputs", "run_name": "timeseries"})
    assert (out / "forecast.csv").exists()
