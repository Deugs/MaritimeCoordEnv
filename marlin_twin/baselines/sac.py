"""Multi-agent SAC (MASAC) baseline policy: maximum-entropy off-policy CTDE."""

import copy

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Normal

from marlin_twin.agents.networks import CentralizedCritic, GATEncoder, SquashedGaussianActor
from marlin_twin.agents.policies import own_feats, tanh_corrected_log_prob
from marlin_twin.data_classes import VesselObservation


class SACPolicy:
    """Multi-Agent Soft Actor-Critic (MASAC) baseline policy.

    Each agent owns a decentralized squashed-Gaussian actor (`own_feats ++
    GATEncoder(graph)[own_node]` -> tanh-bounded action with a tractable
    log-prob) plus TWIN centralized critics conditioned on every vessel's
    own_feats++embedding and every vessel's action, with soft-updated target
    copies of both critics and an automatically tuned entropy coefficient
    `alpha` (target entropy = -action_dim).

    Simpler than `MADDPGPolicy` in two ways: no target actor (next actions
    are resampled from the CURRENT stochastic actor at update time) and no
    target encoder (only the critics are target-tracked).

    Deliberately does **not** subclass `GATPolicy`: `MAPPOTrainer.train()`
    gates its PPO update on `hasattr(pol, "optimizer") and
    hasattr(pol, "evaluate_tensors")` (mappo.py). This class exposes
    NEITHER name -- its optimizers are `actor_optimizer`/`critic_optimizer`/
    `alpha_optimizer`, and it has no `evaluate_tensors`. Do not add an
    attribute named `optimizer` or a method named `evaluate_tensors`; either
    one alone is harmless, but both together would make MAPPOTrainer
    silently apply a PPO update to a SAC policy (the exact trap
    `MADDPGPolicy`'s own docstring documents).
    """

    USES_GRAPH = True
    FEAT_DIM = 6

    def __init__(
        self,
        n_vessels: int = 5,
        action_dim: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.2,
        target_entropy: float | None = None,
        hidden_dim: int = 64,
    ):
        self.n_vessels = n_vessels
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        obs_dim = 6 + hidden_dim

        self.encoder = GATEncoder(in_features=6, edge_features=4, hidden_dim=hidden_dim, heads=4)
        self.actor = SquashedGaussianActor(
            obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim
        )
        # Independently initialized, not deepcopy'd from one another -- the
        # twin-Q trick relies on the two critics disagreeing.
        self.critic1 = CentralizedCritic(
            n_vessels=n_vessels, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim
        )
        self.critic2 = CentralizedCritic(
            n_vessels=n_vessels, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim
        )
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        for net in (self.target_critic1, self.target_critic2):
            for p in net.parameters():
                p.requires_grad = False

        self.log_alpha = torch.tensor(float(np.log(init_alpha)), requires_grad=True)
        self.target_entropy = (
            float(target_entropy) if target_entropy is not None else -float(action_dim)
        )

        # Encoder trained only via the actor loss, matching MADDPGPolicy's
        # convention -- the critic loss below consumes a no_grad embedding.
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.encoder.parameters()), lr=actor_lr
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        # No exploration-noise object -- the stochastic actor sampled below
        # (deterministic=False) is SAC's own exploration mechanism.

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def featurize(self, observation: VesselObservation) -> np.ndarray:
        return own_feats(observation)

    def _embed(self, encoder, graph, node_idx: int) -> torch.Tensor:
        node_emb = encoder(graph.x, graph.edge_index, graph.edge_attr)
        return node_emb[node_idx]

    def sample_action(
        self, obs_t: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`obs_t`: [B, obs_dim]. Returns `(action [B, action_dim] in
        (-1,1), log_prob [B, 1])`. Differentiable w.r.t. `obs_t` and actor
        params via `rsample` -- this is the reparameterization SAC's actor
        gradient needs."""
        mean, log_std = self.actor(obs_t)
        if deterministic:
            zero_logp = torch.zeros(obs_t.shape[0], 1, dtype=obs_t.dtype)
            return torch.tanh(mean), zero_logp
        dist = Normal(mean, log_std.exp())
        raw = dist.rsample()
        return torch.tanh(raw), tanh_corrected_log_prob(dist, raw)

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
            action, _ = self.sample_action(obs_t, deterministic=deterministic)
            return action.squeeze(0).numpy()

    def evaluate(self, observations, actions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Documented stub satisfying the `Policy` protocol -- `MASACTrainer`
        uses its own off-policy `_update`, not this method. Never exercised
        by any real call site."""
        n = len(observations)
        zeros = np.zeros((n, 1), dtype=np.float32)
        return zeros, zeros, zeros

    def get_state(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
        }

    def set_state(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.target_critic1.load_state_dict(state["target_critic1"])
        self.target_critic2.load_state_dict(state["target_critic2"])
        if "log_alpha" in state:
            # copy_, NOT rebinding -- `self.alpha_optimizer` holds a
            # reference to this exact tensor object; assigning
            # `self.log_alpha = ...` would leave the optimizer updating an
            # orphaned parameter and alpha permanently frozen after reload.
            with torch.no_grad():
                self.log_alpha.copy_(state["log_alpha"])
