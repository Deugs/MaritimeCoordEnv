"""GAT-based and ablation RL policies for MARLIN-Twin agents."""

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Normal
from marlin_twin.agents.networks import ActorCriticNet, GATEncoder, MeanPoolingEncoder
from marlin_twin.data_classes import VesselObservation


def own_feats(observation: VesselObservation) -> np.ndarray:
    """Normalize a vessel's own state into the 6-d feature vector shared by
    every policy family (own x, y, heading, speed, surge velocity, yaw rate) —
    the same normalization `ObservationBuilder.to_vector` uses for indices 0:6."""
    s = observation.own_state
    return np.array(
        [
            s.x / 5000.0,
            s.y / 5000.0,
            s.heading / np.pi,
            s.speed / 15.0,
            s.surge_velocity / 15.0,
            s.yaw_rate,
        ],
        dtype=np.float32,
    )


def tanh_corrected_log_prob(dist: Normal, raw_action: torch.Tensor) -> torch.Tensor:
    """Log-prob of a squashed-Gaussian sample under the actual `tanh(raw_action)`
    action distribution: subtracts the tanh change-of-variables correction
    `log(1 - tanh(x)^2)` so PPO's importance ratio and entropy aren't biased
    near the [-1, 1] action boundaries."""
    log_prob = dist.log_prob(raw_action)
    log_prob = log_prob - torch.log(1 - torch.tanh(raw_action) ** 2 + 1e-6)
    return log_prob.sum(dim=-1, keepdim=True)


class GATPolicy:
    """GAT-based Policy for MARLIN-Twin agents.

    Input is `own_feats (6d) ++ GATEncoder(scene_graph)[own_node] (hidden_dim d)`.
    The encoder actually runs as part of the network's forward pass (both at
    inference and, critically, during the PPO update — see
    `evaluate_tensors`), so its parameters receive real gradients.
    """

    ENCODER_CLS = GATEncoder
    USES_GRAPH = True
    FEAT_DIM = 6  # width of the per-vessel dense feature the trainer buffers

    def __init__(self, action_dim: int = 2, lr: float = 3e-4, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        self.encoder = self.ENCODER_CLS(
            in_features=6, edge_features=4, hidden_dim=hidden_dim, heads=4
        )
        self.net = ActorCriticNet(obs_dim=6 + hidden_dim, action_dim=action_dim, hidden_dim=64)
        self.optimizer = optim.Adam(
            list(self.net.parameters()) + list(self.encoder.parameters()), lr=lr
        )

    def featurize(self, observation: VesselObservation) -> np.ndarray:
        """The dense per-vessel feature the trainer stores in the rollout
        buffer every step (own state only — the graph is stored separately,
        since its size varies with the encounter neighborhood)."""
        return own_feats(observation)

    def _embed(self, graph, node_idx: int) -> torch.Tensor:
        node_emb = self.encoder(graph.x, graph.edge_index, graph.edge_attr)
        return node_emb[node_idx]

    def _build_input(self, observation: VesselObservation, graph, node_idx: int) -> torch.Tensor:
        own = torch.tensor(own_feats(observation), dtype=torch.float32)
        return torch.cat([own, self._embed(graph, node_idx)], dim=-1)

    def act(
        self,
        observation: VesselObservation,
        graph=None,
        node_idx: int = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            obs_t = self._build_input(observation, graph, node_idx).unsqueeze(0)
            mean, std, _ = self.net(obs_t)
            action = mean if deterministic else torch.normal(mean, std)
            return torch.tanh(action).squeeze(0).numpy()

    def get_action_and_val(
        self, observation: VesselObservation, graph=None, node_idx: int = None
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Sample one action and return everything derived from that single
        sample: the tanh-squashed [-1,1] action (for building the executed
        VesselAction), the raw pre-tanh action (for buffer storage, matching
        the Normal(mean, std) support `evaluate_tensors` computes log-probs
        against), the value estimate, and the log-prob."""
        self.net.eval()
        with torch.no_grad():
            obs_t = self._build_input(observation, graph, node_idx).unsqueeze(0)
            mean, std, value = self.net(obs_t)
            dist = Normal(mean, std)
            raw_action = dist.sample()
            log_prob = tanh_corrected_log_prob(dist, raw_action)
            tanh_action = torch.tanh(raw_action).squeeze(0).numpy()
            raw_vec = raw_action.squeeze(0).numpy()
            return tanh_action, raw_vec, float(value.item()), float(log_prob.item())

    def evaluate_tensors(
        self, own_feats_t: torch.Tensor, batch, local_node_idx: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO-update-time recompute. `own_feats_t` is `[T, 6]`; `batch` is a
        `torch_geometric.data.Batch` of the T per-timestep scene graphs
        collected during rollout; `local_node_idx` is `[T]`, this vessel's
        node index within `batch` at each timestep; `actions` is `[T, 2]`,
        the raw pre-tanh actions stored at rollout time. Runs the encoder
        WITH gradients (no `torch.no_grad()`), so its parameters actually
        get updated by the PPO loss — this is the fix for the encoder never
        training."""
        self.net.train()
        node_emb = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        obs_t = torch.cat([own_feats_t, node_emb[local_node_idx]], dim=-1)
        mean, std, values = self.net(obs_t)
        dist = Normal(mean, std)
        log_probs = tanh_corrected_log_prob(dist, actions)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return values, log_probs, entropy

    def evaluate(
        self, own_feats_batch: np.ndarray, graphs: list, node_indices: list, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched eval satisfying the `Policy` protocol. Not exercised by any
        current call site (kept for protocol conformance / future use)."""
        self.net.eval()
        with torch.no_grad():
            own_t = torch.tensor(own_feats_batch, dtype=torch.float32)
            embeds = torch.stack(
                [
                    self.encoder(g.x, g.edge_index, g.edge_attr)[i]
                    for g, i in zip(graphs, node_indices)
                ]
            )
            obs_t = torch.cat([own_t, embeds], dim=-1)
            act_t = torch.tensor(actions, dtype=torch.float32)
            mean, std, values = self.net(obs_t)
            dist = Normal(mean, std)
            log_probs = tanh_corrected_log_prob(dist, act_t)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            return values.numpy(), log_probs.numpy(), entropy.numpy()

    def get_state(self) -> dict:
        return {"net": self.net.state_dict(), "encoder": self.encoder.state_dict()}

    def set_state(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
        self.encoder.load_state_dict(state["encoder"])


class MeanPoolingPolicy(GATPolicy):
    """Ablation Variant 1: Mean-Pooling GNN Policy without attention weights.
    Identical to `GATPolicy` except neighbor embeddings are uniformly
    averaged (`MeanPoolingEncoder`) rather than attention-weighted."""

    ENCODER_CLS = MeanPoolingEncoder


class MLPPolicy:
    """Ablation Variant 2: Flat MLP Policy without Graph Neural Networks.
    Uses the original fixed-4-neighbor-cap flattened vector
    (`ObservationBuilder.to_vector`) instead of the encounter graph — the
    "no GNN, no attention, fixed neighbor featurization" ablation."""

    USES_GRAPH = False

    def __init__(self, action_dim: int = 2, lr: float = 3e-4, obs_dim: int = 32):
        self.obs_dim = obs_dim
        self.FEAT_DIM = obs_dim
        self.net = ActorCriticNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    @staticmethod
    def _vec(observation: VesselObservation) -> np.ndarray:
        from marlin_twin.agents.observation_builder import ObservationBuilder

        return ObservationBuilder.to_vector(observation)

    def featurize(self, observation: VesselObservation) -> np.ndarray:
        return self._vec(observation)

    def act(
        self, observation: VesselObservation, graph=None, node_idx=None, deterministic: bool = False
    ) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(self._vec(observation), dtype=torch.float32).unsqueeze(0)
            mean, std, _ = self.net(obs_t)
            action = mean if deterministic else torch.normal(mean, std)
            return torch.tanh(action).squeeze(0).numpy()

    def get_action_and_val(
        self, observation: VesselObservation, graph=None, node_idx=None
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(self._vec(observation), dtype=torch.float32).unsqueeze(0)
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
