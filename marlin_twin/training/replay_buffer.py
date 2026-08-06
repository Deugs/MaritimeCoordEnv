"""Off-policy replay buffer for MADDPG joint (all-vessel) transitions."""

import numpy as np
import torch

try:
    from torch_geometric.data import Batch
except ImportError:
    Batch = None


class ReplayBuffer:
    """Circular replay buffer over joint per-timestep transitions.

    Each entry is one environment step: every vessel's own-state features,
    action, reward, and next own-state features, plus the shared scene
    graph (and each vessel's node index within it) before and after the
    step. MADDPG's centralized critic needs every vessel's data jointly,
    unlike `RolloutBuffer`'s per-agent GAE bookkeeping.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        n_vessels: int = 5,
        feat_dim: int = 6,
        act_dim: int = 2,
    ):
        self.capacity = capacity
        self.n_vessels = n_vessels
        self.own_feats = np.zeros((capacity, n_vessels, feat_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_vessels, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_vessels), dtype=np.float32)
        self.next_own_feats = np.zeros((capacity, n_vessels, feat_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.node_idx = np.zeros((capacity, n_vessels), dtype=np.int64)
        self.next_node_idx = np.zeros((capacity, n_vessels), dtype=np.int64)
        self.graphs: list = [None] * capacity
        self.next_graphs: list = [None] * capacity
        self.ptr = 0
        self.size = 0

    def add(
        self,
        own_feats: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_own_feats: np.ndarray,
        done: bool,
        graph,
        node_idx: np.ndarray,
        next_graph,
        next_node_idx: np.ndarray,
    ) -> None:
        i = self.ptr
        self.own_feats[i] = own_feats
        self.actions[i] = actions
        self.rewards[i] = rewards
        self.next_own_feats[i] = next_own_feats
        self.dones[i] = float(done)
        self.graphs[i] = graph
        self.node_idx[i] = node_idx
        self.next_graphs[i] = next_graph
        self.next_node_idx[i] = next_node_idx
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int) -> dict:
        """Random minibatch of joint transitions, with per-timestep graphs
        batched via `Batch.from_data_list` — `node_idx`/`next_node_idx` are
        translated into global row indices into the returned batches using
        `batch.ptr`, the same offset arithmetic `RolloutBuffer.batched_graph`
        uses."""
        if Batch is None:
            raise ImportError("torch_geometric is required for MADDPG's graph-based replay buffer")
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = Batch.from_data_list([self.graphs[i] for i in idx])
        next_batch = Batch.from_data_list([self.next_graphs[i] for i in idx])
        node_idx = torch.tensor(self.node_idx[idx], dtype=torch.long) + batch.ptr[:-1].unsqueeze(1)
        next_node_idx = torch.tensor(self.next_node_idx[idx], dtype=torch.long) + next_batch.ptr[
            :-1
        ].unsqueeze(1)
        return {
            "own_feats": torch.tensor(self.own_feats[idx], dtype=torch.float32),
            "actions": torch.tensor(self.actions[idx], dtype=torch.float32),
            "rewards": torch.tensor(self.rewards[idx], dtype=torch.float32),
            "next_own_feats": torch.tensor(self.next_own_feats[idx], dtype=torch.float32),
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32),
            "graph_batch": batch,
            "node_idx": node_idx,
            "next_graph_batch": next_batch,
            "next_node_idx": next_node_idx,
        }
