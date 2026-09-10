from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .common import prepare_output, save_figure, save_manifest
from .io import load_table


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.columns.difference(numeric).tolist()
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def _optional_model(name: str, task: str, seed: int):
    definitions = {
        "lightgbm": ("lightgbm", "LGBMClassifier", "LGBMRegressor"),
        "xgboost": ("xgboost", "XGBClassifier", "XGBRegressor"),
        "catboost": ("catboost", "CatBoostClassifier", "CatBoostRegressor"),
    }
    module_name, classifier, regressor = definitions[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"模型 {name} 需要可选依赖，请执行 pip install -e '.[boosting]'") from exc
    cls = getattr(module, classifier if task == "classification" else regressor)
    kwargs = {"random_state": seed}
    if name == "catboost":
        kwargs["verbose"] = False
    return cls(**kwargs)


def _custom_model(name: str, specification: dict[str, Any]):
    class_path = specification.get("class_path", "")
    if "." not in class_path:
        raise ValueError(f"自定义模型 {name} 的 class_path 必须是完整 Python 类路径")
    module_name, class_name = class_path.rsplit(".", 1)
    try:
        model_class = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"无法导入自定义模型 {name}: {class_path}") from exc
    estimator = model_class(**specification.get("params", {}))
    if not callable(getattr(estimator, "fit", None)) or not callable(
        getattr(estimator, "predict", None)
    ):
        raise TypeError(f"自定义模型 {name} 必须实现 fit() 和 predict()")
    return estimator


def _models(
    task: str,
    names: list[str],
    seed: int,
    custom_models: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    defaults = {
        "classification": {
            "logistic": LogisticRegression(max_iter=2000),
            "svm": CalibratedClassifierCV(SVC(random_state=seed)),
            "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        },
        "regression": {
            "linear": LinearRegression(),
            "ridge": Ridge(),
            "random_forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(random_state=seed),
        },
    }[task]
    custom_models = custom_models or {}
    selected = {}
    for name in names:
        if name in defaults:
            selected[name] = defaults[name]
        elif name in {"lightgbm", "xgboost", "catboost"}:
            selected[name] = _optional_model(name, task, seed)
        elif name in custom_models:
            selected[name] = _custom_model(name, custom_models[name])
        else:
            available = sorted([*defaults, "lightgbm", "xgboost", "catboost", *custom_models])
            raise ValueError(f"未知模型: {name}；当前已注册: {', '.join(available)}")
    return selected


def run(config: dict[str, Any]) -> Path:
    task = config["task"]
    if task not in {"classification", "regression"}:
        raise ValueError("tabular.run 仅支持 classification/regression")
    frame = load_table(config["data"], config.get("sheet_name", 0))
    target = config["target"]
    drop = config.get("drop_columns", [])
    if target not in frame:
        raise ValueError(f"目标列不存在: {target}")
    frame = frame.dropna(subset=[target])
    y = frame[target]
    X = frame.drop(columns=[target, *drop], errors="ignore")
    seed = int(config.get("random_state", 42))
    test_size = float(config.get("test_size", 0.2))
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    names = config.get("models") or (["logistic", "svm", "random_forest", "gradient_boosting"]
                                      if task == "classification" else
                                      ["linear", "ridge", "random_forest", "gradient_boosting"])
    models = _models(task, names, seed, config.get("custom_models"))
    cv_folds = int(config.get("cv_folds", 5))
    if task == "classification":
        cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
        scoring = {"accuracy": "accuracy", "f1": "f1_weighted"}
        rank_metric = "cv_f1"
    else:
        cv = KFold(cv_folds, shuffle=True, random_state=seed)
        scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
        rank_metric = "cv_rmse"
    rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", _preprocessor(X_train)), ("model", estimator)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        if task == "classification":
            rows.append({"model": name, "cv_accuracy": scores["test_accuracy"].mean(),
                         "cv_f1": scores["test_f1"].mean()})
        else:
            rows.append({"model": name, "cv_rmse": -scores["test_rmse"].mean(),
                         "cv_mae": -scores["test_mae"].mean(), "cv_r2": scores["test_r2"].mean()})
    metrics = pd.DataFrame(rows)
    ascending = task == "regression"
    metrics = metrics.sort_values(rank_metric, ascending=ascending).reset_index(drop=True)
    best_name = metrics.loc[0, "model"]
    best = fitted[best_name]
    pred = best.predict(X_test)
    output = prepare_output(config)
    predictions = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": pred}, index=y_test.index)
    if task == "classification":
        test_metrics = {"test_accuracy": accuracy_score(y_test, pred),
                        "test_f1": f1_score(y_test, pred, average="weighted")}
        if y.nunique() == 2 and hasattr(best, "predict_proba"):
            probability = best.predict_proba(X_test)[:, 1]
            predictions["probability"] = probability
            test_metrics["test_roc_auc"] = roc_auc_score(y_test, probability)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap="Blues")
        ax.set_title(f"Confusion matrix: {best_name}")
        save_figure(fig, output, "confusion_matrix")
    else:
        test_metrics = {"test_rmse": mean_squared_error(y_test, pred) ** 0.5,
                        "test_mae": mean_absolute_error(y_test, pred), "test_r2": r2_score(y_test, pred)}
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, pred, alpha=.65)
        low, high = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
        ax.plot([low, high], [low, high], "--", color="black")
        ax.set(xlabel="Actual", ylabel="Predicted", title=f"Prediction: {best_name}")
        save_figure(fig, output, "prediction_scatter")
    for key, value in test_metrics.items():
        metrics.loc[metrics["model"] == best_name, key] = value
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index_label="row_index")
    joblib.dump(best, output / "best_model.joblib")
    save_manifest(config, output, {"best_model": best_name, "train_rows": len(X_train), "test_rows": len(X_test)})
    return output
