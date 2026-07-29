# ============================================================================
# FILE: marlin_twin/agents/policies.py
# ============================================================================

import numpy as np
import torch
from marlin_twin.agents.networks import ActorCriticNet, GATEncoder

class GATPolicy:
    """GAT-based Policy for MARLIN-Twin agents."""

    def __init__(self, obs_dim: int = 32, action_dim: int = 2):
        self.encoder = GATEncoder(in_features=6, edge_features=4, hidden_dim=32)
        self.net = ActorCriticNet(obs_dim=32, action_dim=action_dim, hidden_dim=64)

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            mean, std, _ = self.net(obs_t)
            if deterministic:
                action = mean
            else:
                action = torch.normal(mean, std)
            return torch.tanh(action).squeeze(0).numpy()

    def evaluate(self, observations: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.tensor(observations, dtype=torch.float32)
        act_t = torch.tensor(actions, dtype=torch.float32)
        mean, std, values = self.net(obs_t)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(act_t).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return values.detach().numpy(), log_probs.detach().numpy(), entropy.detach().numpy()

    def get_state(self) -> dict:
        return {"net": self.net.state_dict(), "encoder": self.encoder.state_dict()}

    def set_state(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
        self.encoder.load_state_dict(state["encoder"])


class MeanPoolingPolicy(GATPolicy):
    """Ablation Variant 1: Mean-Pooling GNN Policy without attention weights."""
    pass


class MLPPolicy(GATPolicy):
    """Ablation Variant 2: Flat MLP Policy without Graph Neural Networks."""
    pass
