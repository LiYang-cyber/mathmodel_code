"""可复用 Stacking 分类/回归模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.io import load_table
from mathmodel.tabular import _preprocessor


def run_stacking(data: str, target: str, task: str, output: str, seed: int = 42) -> Path:
    frame = load_table(repo_path(data)).dropna(subset=[target])
    X, y = frame.drop(columns=target), frame[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=seed, stratify=y if task == "classification" else None
    )
    if task == "classification":
        estimator = StackingClassifier(
            [("rf", RandomForestClassifier(n_estimators=200, random_state=seed)),
             ("gb", GradientBoostingClassifier(random_state=seed))],
            final_estimator=LogisticRegression(max_iter=2000), cv=5,
        )
    else:
        estimator = StackingRegressor(
            [("rf", RandomForestRegressor(n_estimators=200, random_state=seed)),
             ("gb", GradientBoostingRegressor(random_state=seed))],
            final_estimator=Ridge(), cv=5,
        )
    model = Pipeline([("preprocess", _preprocessor(X_train)), ("model", estimator)])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = ({"accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred, average="weighted")}
               if task == "classification" else
               {"rmse": mean_squared_error(y_test, pred) ** .5, "r2": r2_score(y_test, pred)})
    out = repo_path(output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({"actual": y_test, "predicted": pred}).to_csv(out / "predictions.csv", index=False)
    joblib.dump(model, out / "stacking_model.joblib")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Stacking 集成模型")
    parser.add_argument("data")
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--output", default="outputs/stacking")
    args = parser.parse_args()
    print(run_stacking(args.data, args.target, args.task, args.output).resolve())


if __name__ == "__main__":
    main()

