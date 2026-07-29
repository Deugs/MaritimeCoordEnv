# ============================================================================
# FILE: marlin_twin/training/reward_normalizer.py
# ============================================================================

import numpy as np

class RewardNormalizer:
    """Running mean and standard deviation scaler for rewards."""

    def __init__(self, shape=(), gamma: float = 0.99, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_ms = RunningMeanStd(shape=shape)
        self.returns = np.zeros(shape)

    def normalize(self, reward: float) -> float:
        self.returns = self.returns * self.gamma + reward
        self.running_ms.update(self.returns)
        return float(reward / np.sqrt(self.running_ms.var + self.epsilon))


class RunningMeanStd:
    """Tracks running mean and variance."""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = 1 if x.ndim == 0 else x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
