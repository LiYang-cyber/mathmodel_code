"""对已保存的表格 Pipeline 生成 SHAP 全局特征重要性。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.common import save_figure
from mathmodel.io import load_table


def explain(model_path: str, data_path: str, output: str, target: str | None, max_samples: int = 500) -> Path:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP 未安装，请使用 pixi run -e full 执行") from exc
    pipeline = joblib.load(repo_path(model_path))
    frame = load_table(repo_path(data_path))
    X = frame.drop(columns=[target], errors="ignore")
    sample = X.sample(min(max_samples, len(X)), random_state=42)
    transformed = pipeline.named_steps["preprocess"].transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    estimator = pipeline.named_steps["model"]
    explainer = shap.Explainer(estimator, transformed, feature_names=feature_names)
    values = explainer(transformed)
    out = repo_path(output)
    out.mkdir(parents=True, exist_ok=True)
    shap.plots.beeswarm(values, max_display=20, show=False)
    save_figure(plt.gcf(), out, "shap_summary")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 SHAP 解释图")
    parser.add_argument("model")
    parser.add_argument("data")
    parser.add_argument("--target")
    parser.add_argument("--output", default="outputs/shap")
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()
    print(explain(args.model, args.data, args.output, args.target, args.max_samples).resolve())


if __name__ == "__main__":
    main()
