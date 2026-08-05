"""GAT-based and ablation RL policies for MARLIN-Twin agents."""

import numpy as np
import torch
import torch.optim as optim
from marlin_twin.agents.networks import ActorCriticNet, GATEncoder


class GATPolicy:
    """GAT-based Policy for MARLIN-Twin agents."""

    def __init__(self, obs_dim: int = 32, action_dim: int = 2, lr: float = 3e-4):
        self.encoder = GATEncoder(in_features=6, edge_features=4, hidden_dim=32)
        self.net = ActorCriticNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64)
        self.optimizer = optim.Adam(
            list(self.net.parameters()) + list(self.encoder.parameters()), lr=lr
        )

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

    def get_action_and_val(self, observation: np.ndarray) -> tuple[np.ndarray, float, float]:
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            mean, std, value = self.net(obs_t)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1).item()
            act_vec = torch.tanh(action).squeeze(0).numpy()
            return act_vec, float(value.item()), float(log_prob)

    def evaluate_tensors(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.net.train()
        mean, std, values = self.net(observations)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return values, log_probs, entropy

    def evaluate(
        self, observations: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(observations, dtype=torch.float32)
            act_t = torch.tensor(actions, dtype=torch.float32)
            mean, std, values = self.net(obs_t)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(act_t).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            return values.numpy(), log_probs.numpy(), entropy.numpy()

    def get_state(self) -> dict:
        return {"net": self.net.state_dict(), "encoder": self.encoder.state_dict()}

    def set_state(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
        self.encoder.load_state_dict(state["encoder"])


class MeanPoolingPolicy(GATPolicy):
    """Ablation Variant 1: Mean-Pooling GNN Policy without attention weights."""

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            # Uniform mean-pooling on neighbor features (indices 6..22)
            obs_mod = observation.copy()
            if len(obs_mod) >= 22:
                neighbor_feats = obs_mod[6:22].reshape(-1, 4)
                mean_feat = np.mean(neighbor_feats, axis=0)
                obs_mod[6:10] = mean_feat
                obs_mod[10:22] = 0.0

            obs_t = torch.tensor(obs_mod, dtype=torch.float32).unsqueeze(0)
            mean, std, _ = self.net(obs_t)
            action = mean if deterministic else torch.normal(mean, std)
            return torch.tanh(action).squeeze(0).numpy()

    def evaluate_tensors(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.net.train()
        obs_mod = observations.clone()
        if obs_mod.shape[-1] >= 22:
            n_feats = obs_mod[:, 6:22].view(-1, 4, 4)
            m_feat = n_feats.mean(dim=1)
            obs_mod[:, 6:10] = m_feat
            obs_mod[:, 10:22] = 0.0

        mean, std, values = self.net(obs_mod)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return values, log_probs, entropy


class MLPPolicy(GATPolicy):
    """Ablation Variant 2: Flat MLP Policy without Graph Neural Networks."""

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            obs_mod = observation.copy()
            obs_t = torch.tensor(obs_mod, dtype=torch.float32).unsqueeze(0)
            mean, std, _ = self.net(obs_t)
            action = mean if deterministic else torch.normal(mean, std)
            return torch.tanh(action).squeeze(0).numpy()
