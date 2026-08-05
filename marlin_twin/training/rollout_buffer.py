"""Multi-agent PPO rollout buffer with GAE advantage computation."""

import numpy as np


class RolloutBuffer:
    """Multi-Agent PPO Rollout Buffer."""

    def __init__(
        self, buffer_size: int = 500, n_vessels: int = 5, obs_dim: int = 32, act_dim: int = 2
    ):
        self.buffer_size = buffer_size
        self.n_vessels = n_vessels
        self.obs_buf = np.zeros((buffer_size, n_vessels, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((buffer_size, n_vessels, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((buffer_size, n_vessels), dtype=np.float32)
        self.val_buf = np.zeros((buffer_size, n_vessels), dtype=np.float32)
        self.logp_buf = np.zeros((buffer_size, n_vessels), dtype=np.float32)
        self.adv_buf = np.zeros((buffer_size, n_vessels), dtype=np.float32)
        self.ret_buf = np.zeros((buffer_size, n_vessels), dtype=np.float32)
        self.ptr = 0

    def add(
        self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, val: np.ndarray, logp: np.ndarray
    ) -> None:
        if self.ptr < self.buffer_size:
            self.obs_buf[self.ptr] = obs
            self.act_buf[self.ptr] = act
            self.rew_buf[self.ptr] = rew
            self.val_buf[self.ptr] = val
            self.logp_buf[self.ptr] = logp
            self.ptr += 1

    def compute_returns_and_advantages(
        self, last_values: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE) and discounted returns."""
        last_gae = np.zeros(self.n_vessels, dtype=np.float32)
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_values = last_values
            else:
                next_values = self.val_buf[t + 1]

            delta = self.rew_buf[t] + gamma * next_values - self.val_buf[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            self.adv_buf[t] = last_gae
            self.ret_buf[t] = self.adv_buf[t] + self.val_buf[t]

        # Normalize advantages per agent
        mean_adv = np.mean(self.adv_buf[: self.ptr])
        std_adv = np.std(self.adv_buf[: self.ptr]) + 1e-8
        self.adv_buf[: self.ptr] = (self.adv_buf[: self.ptr] - mean_adv) / std_adv

    def clear(self) -> None:
        self.ptr = 0
