"""KMeans/GMM/DBSCAN 聚类统一入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _bootstrap import repo_path

from mathmodel.clustering import run


def main() -> None:
    parser = argparse.ArgumentParser(description="运行聚类分析")
    parser.add_argument("data")
    parser.add_argument("--features", nargs="*")
    parser.add_argument("--method", choices=["kmeans", "gmm", "dbscan"], default="kmeans")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    config = {"task": "clustering", "run_name": f"cluster_{args.method}",
              "data": str(repo_path(args.data)), "features": args.features, "method": args.method,
              "n_clusters": args.clusters, "output_dir": str(repo_path(args.output)), "random_state": 42}
    print(run(config).resolve())


if __name__ == "__main__":
    main()
