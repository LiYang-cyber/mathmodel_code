"""PyOD 多异常检测器统一比较模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _bootstrap import repo_path

from mathmodel.io import load_table


def compare(data: str, output: str, features: list[str] | None = None, label: str | None = None, contamination=.1) -> Path:
    try:
        from pyod.models.copod import COPOD
        from pyod.models.ecod import ECOD
        from pyod.models.iforest import IForest
        from pyod.models.knn import KNN
    except ImportError as exc:
        raise ImportError("PyOD 未安装，请使用 pixi run -e full 执行") from exc
    frame = load_table(repo_path(data))
    features = features or frame.drop(columns=[label] if label else []).select_dtypes(include="number").columns.tolist()
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(frame[features]))
    detectors = {"iforest": IForest(contamination=contamination, random_state=42),
                 "knn": KNN(contamination=contamination), "copod": COPOD(contamination=contamination),
                 "ecod": ECOD(contamination=contamination)}
    rows, result = [], frame.copy()
    for name, model in detectors.items():
        model.fit(X)
        result[f"{name}_label"] = model.labels_
        result[f"{name}_score"] = model.decision_scores_
        row = {"model": name, "anomalies": int(model.labels_.sum())}
        if label:
            row.update(f1=f1_score(frame[label], model.labels_), roc_auc=roc_auc_score(frame[label], model.decision_scores_))
        rows.append(row)
    out = repo_path(output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    result.to_csv(out / "anomaly_comparison.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 PyOD 异常检测器")
    parser.add_argument("data")
    parser.add_argument("--features", nargs="*")
    parser.add_argument("--label")
    parser.add_argument("--contamination", type=float, default=.1)
    parser.add_argument("--output", default="outputs/pyod_compare")
    args = parser.parse_args()
    print(compare(args.data, args.output, args.features, args.label, args.contamination).resolve())


if __name__ == "__main__":
    main()
