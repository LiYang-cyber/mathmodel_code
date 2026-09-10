"""分类、回归及时序交叉验证切分模板。"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit


def make_splitter(task: str, folds: int = 5, seed: int = 42):
    if task == "classification":
        return StratifiedKFold(folds, shuffle=True, random_state=seed)
    if task == "regression":
        return KFold(folds, shuffle=True, random_state=seed)
    if task == "timeseries":
        return TimeSeriesSplit(folds)
    raise ValueError(f"未知任务: {task}")


def describe_splits(task: str, samples: int, folds: int, labels: np.ndarray | None = None):
    splitter = make_splitter(task, folds)
    X = np.arange(samples).reshape(-1, 1)
    y = labels if labels is not None else np.zeros(samples)
    return [{"fold": index, "train": len(train), "validation": len(valid),
             "train_last": int(train[-1]), "validation_first": int(valid[0])}
            for index, (train, valid) in enumerate(splitter.split(X, y), 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="预览交叉验证切分")
    parser.add_argument("--task", choices=["classification", "regression", "timeseries"], required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    labels = np.arange(args.samples) % 2 if args.task == "classification" else None
    print(json.dumps(describe_splits(args.task, args.samples, args.folds, labels), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

