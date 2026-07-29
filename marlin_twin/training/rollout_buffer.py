# ============================================================================
# FILE: marlin_twin/training/rollout_buffer.py
# ============================================================================

import numpy as np

class RolloutBuffer:
    """Multi-Agent PPO Rollout Buffer."""

    def __init__(self, buffer_size: int = 500, n_agents: int = 5, obs_dim: int = 32, act_dim: int = 2):
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.obs_buf = np.zeros((buffer_size, n_agents, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((buffer_size, n_agents, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.val_buf = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.logp_buf = np.zeros((buffer_size, n_agents), dtype=np.float32)
        self.ptr = 0

    def add(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, val: np.ndarray, logp: np.ndarray) -> None:
        if self.ptr < self.buffer_size:
            self.obs_buf[self.ptr] = obs
            self.act_buf[self.ptr] = act
            self.rew_buf[self.ptr] = rew
            self.val_buf[self.ptr] = val
            self.logp_buf[self.ptr] = logp
            self.ptr += 1

    def clear(self) -> None:
        self.ptr = 0
