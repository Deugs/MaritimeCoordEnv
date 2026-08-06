"""Multi-agent PPO rollout buffer with GAE advantage computation."""

import numpy as np
import torch

try:
    from torch_geometric.data import Batch
except ImportError:
    Batch = None


class RolloutBuffer:
    """Multi-Agent PPO Rollout Buffer.

    Stores per-vessel own-state features densely (always), plus one shared
    per-timestep scene graph for graph-based policies (`GATPolicy`,
    `MeanPoolingPolicy`). The graph is kept around (not just a cached
    embedding) so the PPO update can recompute the encoder's forward pass
    WITH gradients via `batched_graph()` — training on a frozen, no-grad
    embedding captured at rollout time would never actually update the
    encoder's parameters.
    """

    def __init__(
        self, buffer_size: int = 500, n_vessels: int = 5, feat_dim: int = 6, act_dim: int = 2
    ):
        self.buffer_size = buffer_size
        self.n_vessels = n_vessels
        self.own_feats_buf = np.zeros((buffer_size, n_vessels, feat_dim), dtype=np.float32)
        self.act_buf = np.zeros((buffer_size, n_vessels, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((buffer_size, n_vessels), dtype=np.float32)
        self.val_buf = np.zeros((buffer_size, n_vessels), dtype=np.float32)
        self.logp_buf = np.zeros((buffer_size, n_vessels), dtype=np.float32)
        self.adv_buf = np.zeros((buffer_size, n_vessels), dtype=np.float32)
        self.ret_buf = np.zeros((buffer_size, n_vessels), dtype=np.float32)
        self.node_idx_buf = np.zeros((buffer_size, n_vessels), dtype=np.int64)
        self.graphs: list = [None] * buffer_size
        self.ptr = 0

    def add(
        self,
        own_feats: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        val: np.ndarray,
        logp: np.ndarray,
        graph=None,
        node_idx: np.ndarray | None = None,
    ) -> None:
        if self.ptr < self.buffer_size:
            self.own_feats_buf[self.ptr] = own_feats
            self.act_buf[self.ptr] = act
            self.rew_buf[self.ptr] = rew
            self.val_buf[self.ptr] = val
            self.logp_buf[self.ptr] = logp
            self.graphs[self.ptr] = graph
            if node_idx is not None:
                self.node_idx_buf[self.ptr] = node_idx
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

    def batched_graph(self, vid: int) -> tuple:
        """Batch this rollout's stored graphs (up to `ptr`) into one
        `torch_geometric.data.Batch`, plus vessel `vid`'s global node index
        at each timestep within that batch — for `GATPolicy.evaluate_tensors`
        during the PPO update."""
        if Batch is None:
            raise ImportError("torch_geometric is required for graph-based policies")
        batch = Batch.from_data_list(self.graphs[: self.ptr])
        local_idx = torch.tensor(self.node_idx_buf[: self.ptr, vid], dtype=torch.long)
        global_idx = local_idx + batch.ptr[:-1]
        return batch, global_idx

    def clear(self) -> None:
        self.ptr = 0
        self.graphs = [None] * self.buffer_size
