"""粒子群优化模板；示例目标函数可替换为赛题模型。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _bootstrap import repo_path

from mathmodel.common import save_figure
from mathmodel.io import write_json
from mathmodel.optimization import particle_swarm


def sphere(values: np.ndarray) -> float:
    """演示目标；实际使用时替换为题目的目标函数与约束罚项。"""
    return float(np.sum(values**2))


def main() -> None:
    parser = argparse.ArgumentParser(description="运行粒子群优化演示")
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--lower", type=float, default=-5)
    parser.add_argument("--upper", type=float, default=5)
    parser.add_argument("--particles", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", default="outputs/particle_swarm")
    args = parser.parse_args()
    result = particle_swarm(sphere, [(args.lower, args.upper)] * args.dimensions,
                            particles=args.particles, iterations=args.iterations)
    out = repo_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json({"x": result.x.tolist(), "objective": result.fun, "iterations": result.nit,
                "evaluations": result.nfev}, out / "result.json")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result.history)
    ax.set(xlabel="Iteration", ylabel="Best objective", title="Particle swarm convergence")
    ax.set_yscale("log")
    save_figure(fig, out, "particle_swarm_convergence")
    print(json.dumps({"x": result.x.tolist(), "objective": result.fun}))


if __name__ == "__main__":
    main()

