from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import prepare_output, save_figure, save_manifest
from .io import load_table


def run(config: dict[str, Any]) -> Path:
    frame = load_table(config["data"], config.get("sheet_name", 0))
    label_col = config.get("label")
    excluded = [label_col] if label_col else []
    features = config.get("features") or frame.drop(columns=excluded).select_dtypes(include="number").columns.tolist()
    X = frame[features]
    seed = int(config.get("random_state", 42))
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ("model", IsolationForest(contamination=config.get("contamination", "auto"), random_state=seed)),
    ])
    raw = model.fit_predict(X)
    score = -model.decision_function(X)
    predicted = (raw == -1).astype(int)
    result = frame.copy()
    result["anomaly_score"] = score
    result["is_anomaly"] = predicted
    metrics: dict[str, Any] = {"anomalies": int(predicted.sum()), "anomaly_rate": float(predicted.mean())}
    if label_col:
        truth = frame[label_col].astype(int)
        metrics.update({"f1": f1_score(truth, predicted), "roc_auc": roc_auc_score(truth, score)})
    output = prepare_output(config)
    result.to_csv(output / "anomalies.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(score, bins=30, color="#4472C4", alpha=.8)
    ax.set(xlabel="Anomaly score", ylabel="Count", title="Anomaly score distribution")
    save_figure(fig, output, "anomaly_scores")
    joblib.dump(model, output / "anomaly_model.joblib")
    save_manifest(config, output, metrics)
    return output

