"""不依赖问题结构的元启发式优化基线。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import OptimizeResult


def particle_swarm(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    particles: int = 40,
    iterations: int = 200,
    seed: int = 42,
    inertia: float = 0.72,
    cognitive: float = 1.49,
    social: float = 1.49,
) -> OptimizeResult:
    """连续变量粒子群最小化器；约束问题可在目标函数中加入罚项。"""
    rng = np.random.default_rng(seed)
    limits = np.asarray(bounds, dtype=float)
    if limits.ndim != 2 or limits.shape[1] != 2 or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("bounds 必须是 (下界, 上界) 序列")
    lower, upper = limits[:, 0], limits[:, 1]
    positions = rng.uniform(lower, upper, size=(particles, len(bounds)))
    velocities = rng.uniform(-(upper - lower), upper - lower, size=positions.shape) * 0.1
    personal = positions.copy()
    personal_values = np.asarray([objective(item) for item in positions])
    best_index = int(np.argmin(personal_values))
    global_best, global_value = personal[best_index].copy(), float(personal_values[best_index])
    history = [global_value]
    evaluations = particles
    for _ in range(iterations):
        r1, r2 = rng.random(positions.shape), rng.random(positions.shape)
        velocities = inertia * velocities + cognitive * r1 * (personal - positions) + social * r2 * (global_best - positions)
        positions = np.clip(positions + velocities, lower, upper)
        values = np.asarray([objective(item) for item in positions])
        evaluations += particles
        improved = values < personal_values
        personal[improved], personal_values[improved] = positions[improved], values[improved]
        best_index = int(np.argmin(personal_values))
        if personal_values[best_index] < global_value:
            global_best, global_value = personal[best_index].copy(), float(personal_values[best_index])
        history.append(global_value)
    return OptimizeResult(x=global_best, fun=global_value, nit=iterations, nfev=evaluations,
                          success=True, message="iteration limit reached", history=np.asarray(history))

