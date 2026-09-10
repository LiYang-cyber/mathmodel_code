"""BP 神经网络基线，使用 sklearn MLP 并复用统一评估流水线。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _bootstrap import repo_path

from mathmodel.io import load_config
from mathmodel.tabular import run


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 MLP/BP 神经网络")
    parser.add_argument("--config", "-c", default="configs/classification.yaml")
    parser.add_argument("--layers", nargs="+", type=int, default=[64, 32])
    args = parser.parse_args()
    config = load_config(repo_path(args.config))
    class_name = "MLPClassifier" if config["task"] == "classification" else "MLPRegressor"
    config["models"] = ["mlp"]
    config["custom_models"] = {
        "mlp": {
            "class_path": f"sklearn.neural_network.{class_name}",
            "params": {"hidden_layer_sizes": args.layers, "max_iter": 2000, "early_stopping": True, "random_state": 42},
        }
    }
    config["run_name"] = config.get("run_name", config["task"]) + "_mlp"
    print(run(config).resolve())


if __name__ == "__main__":
    main()

