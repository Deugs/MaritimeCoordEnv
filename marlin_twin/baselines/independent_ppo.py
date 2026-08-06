"""Independent PPO baseline policy (no inter-agent communication)."""

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Normal
from marlin_twin.agents.networks import ActorCriticNet
from marlin_twin.agents.policies import own_feats, tanh_corrected_log_prob
from marlin_twin.data_classes import VesselObservation


class IndependentPPOPolicy:
    """Independent PPO Baseline Policy (No inter-agent communication).

    Uses only the vessel's own 6-d state — never its neighbors' states nor
    the encounter graph — matching the "independent learner" baseline
    convention: each agent trains and acts as if it were alone.
    """

    USES_GRAPH = False
    FEAT_DIM = 6

    def __init__(self, action_dim: int = 2, lr: float = 3e-4):
        self.net = ActorCriticNet(obs_dim=6, action_dim=action_dim, hidden_dim=64)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def featurize(self, observation: VesselObservation) -> np.ndarray:
        return own_feats(observation)

    def act(
        self, observation: VesselObservation, graph=None, node_idx=None, deterministic: bool = False
    ) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(own_feats(observation), dtype=torch.float32).unsqueeze(0)
            mean, std, _ = self.net(obs_t)
            action = mean if deterministic else torch.normal(mean, std)
            return torch.tanh(action).squeeze(0).numpy()

    def get_action_and_val(
        self, observation: VesselObservation, graph=None, node_idx=None
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(own_feats(observation), dtype=torch.float32).unsqueeze(0)
            mean, std, value = self.net(obs_t)
            dist = Normal(mean, std)
            raw_action = dist.sample()
            log_prob = tanh_corrected_log_prob(dist, raw_action)
            tanh_action = torch.tanh(raw_action).squeeze(0).numpy()
            raw_vec = raw_action.squeeze(0).numpy()
            return tanh_action, raw_vec, float(value.item()), float(log_prob.item())

    def evaluate_tensors(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.net.train()
        mean, std, values = self.net(observations)
        dist = Normal(mean, std)
        log_probs = tanh_corrected_log_prob(dist, actions)
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
            dist = Normal(mean, std)
            log_probs = tanh_corrected_log_prob(dist, act_t)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            return values.numpy(), log_probs.numpy(), entropy.numpy()

    def get_state(self) -> dict:
        return {"net": self.net.state_dict()}

    def set_state(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
