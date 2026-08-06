"""MADDPG baseline policy: real Multi-Agent Deep Deterministic Policy Gradient."""

import copy

import numpy as np
import torch
import torch.optim as optim
from marlin_twin.agents.networks import CentralizedCritic, DeterministicActor, GATEncoder
from marlin_twin.agents.policies import own_feats
from marlin_twin.data_classes import VesselObservation


class GaussianNoise:
    """Decaying-Gaussian exploration noise added to MADDPG's deterministic
    actions during training (no noise at evaluation time)."""

    def __init__(
        self,
        action_dim: int = 2,
        sigma: float = 0.3,
        decay: float = 0.9995,
        sigma_min: float = 0.05,
    ):
        self.action_dim = action_dim
        self.sigma = sigma
        self.decay = decay
        self.sigma_min = sigma_min

    def sample(self) -> torch.Tensor:
        noise = torch.randn(self.action_dim) * self.sigma
        self.sigma = max(self.sigma_min, self.sigma * self.decay)
        return noise


class MADDPGPolicy:
    """Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Baseline Policy.

    Each agent has its own decentralized deterministic actor (`own_feats ++
    GATEncoder(graph)[own_node]` -> tanh action) plus its own centralized
    critic conditioned on every vessel's own_feats++embedding and every
    vessel's action, with target networks and soft updates — real CTDE via
    off-policy replay (`MADDPGTrainer`), not PPO. Deliberately does **not**
    subclass `GATPolicy`: `MAPPOTrainer.train()` gates its PPO update on
    `hasattr(pol, "optimizer") and hasattr(pol, "evaluate_tensors")`, and
    this policy has neither.
    """

    USES_GRAPH = True
    FEAT_DIM = 6

    def __init__(
        self,
        n_vessels: int = 5,
        action_dim: int = 2,
        lr: float = 1e-3,
        hidden_dim: int = 64,
    ):
        self.n_vessels = n_vessels
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        obs_dim = 6 + hidden_dim

        self.encoder = GATEncoder(in_features=6, edge_features=4, hidden_dim=hidden_dim, heads=4)
        self.actor = DeterministicActor(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim
        )
        self.critic = CentralizedCritic(
            n_vessels=n_vessels, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim
        )

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        for net in (self.target_encoder, self.target_actor, self.target_critic):
            for p in net.parameters():
                p.requires_grad = False

        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.encoder.parameters()), lr=lr
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.noise = GaussianNoise(action_dim=action_dim)

    def featurize(self, observation: VesselObservation) -> np.ndarray:
        return own_feats(observation)

    def _embed(self, encoder, graph, node_idx: int) -> torch.Tensor:
        node_emb = encoder(graph.x, graph.edge_index, graph.edge_attr)
        return node_emb[node_idx]

    def act(
        self,
        observation: VesselObservation,
        graph=None,
        node_idx: int = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        with torch.no_grad():
            own = torch.tensor(own_feats(observation), dtype=torch.float32)
            emb = self._embed(self.encoder, graph, node_idx)
            obs_t = torch.cat([own, emb], dim=-1).unsqueeze(0)
            action = self.actor(obs_t).squeeze(0)
            if not deterministic:
                action = torch.clamp(action + self.noise.sample(), -1.0, 1.0)
            return action.numpy()

    def evaluate(self, observations, actions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Documented stub satisfying the `Policy` protocol: a deterministic
        policy has no distribution, so there is no log-prob/entropy to
        return. `MADDPGTrainer` uses its own off-policy `_update`, not this
        method — never exercised by any real call site."""
        n = len(observations)
        zeros = np.zeros((n, 1), dtype=np.float32)
        return zeros, zeros, zeros

    def get_state(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }

    def set_state(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_encoder.load_state_dict(state["target_encoder"])
        self.target_actor.load_state_dict(state["target_actor"])
        self.target_critic.load_state_dict(state["target_critic"])
