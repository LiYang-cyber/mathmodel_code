"""小样本预测模型。"""

from __future__ import annotations

import numpy as np


class GM11:
    """GM(1,1) 灰色预测，适合短且近似指数变化的正数序列。"""

    def fit(self, values) -> GM11:
        sequence = np.asarray(values, dtype=float)
        if sequence.ndim != 1 or len(sequence) < 4:
            raise ValueError("GM(1,1) 至少需要 4 个一维观测值")
        if np.any(sequence <= 0):
            raise ValueError("GM(1,1) 输入必须为正数")
        accumulated = np.cumsum(sequence)
        background = -0.5 * (accumulated[1:] + accumulated[:-1])
        design = np.column_stack([background, np.ones(len(background))])
        self.a_, self.b_ = np.linalg.lstsq(design, sequence[1:], rcond=None)[0]
        self.first_ = sequence[0]
        self.n_obs_ = len(sequence)
        fitted = self._response(np.arange(self.n_obs_))
        restored = np.r_[fitted[0], np.diff(fitted)]
        self.residuals_ = sequence - restored
        return self

    def _response(self, steps: np.ndarray) -> np.ndarray:
        if abs(self.a_) < 1e-12:
            return self.first_ + self.b_ * steps
        return (self.first_ - self.b_ / self.a_) * np.exp(-self.a_ * steps) + self.b_ / self.a_

    def predict(self, horizon: int) -> np.ndarray:
        if not hasattr(self, "n_obs_"):
            raise RuntimeError("请先调用 fit()")
        if horizon < 1:
            raise ValueError("horizon 必须为正整数")
        steps = np.arange(self.n_obs_ - 1, self.n_obs_ + horizon)
        accumulated = self._response(steps)
        return np.diff(accumulated)

    def posterior_error_ratio(self) -> float:
        """返回后验差比 C；越小表示拟合残差相对原序列越小。"""
        original_std = np.std(self.residuals_ + self._fitted_values(), ddof=1)
        return float(np.std(self.residuals_, ddof=1) / max(original_std, 1e-12))

    def _fitted_values(self) -> np.ndarray:
        response = self._response(np.arange(self.n_obs_))
        return np.r_[response[0], np.diff(response)]
