"""GAT encoder and actor-critic neural network architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATEncoder(nn.Module):
    """
    Graph Attention Network (GAT) Encoder for Encounter Graphs.
    Computes dynamic multi-head attention weights over neighbor vessels.
    """

    def __init__(
        self, in_features: int = 6, edge_features: int = 4, hidden_dim: int = 64, heads: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads

        self.node_proj = nn.Linear(in_features, hidden_dim)
        self.edge_proj = nn.Linear(edge_features, hidden_dim)

        self.attn_src = nn.Parameter(torch.zeros(1, heads, hidden_dim // heads))
        self.attn_dst = nn.Parameter(torch.zeros(1, heads, hidden_dim // heads))
        self.attn_edge = nn.Parameter(torch.zeros(1, heads, hidden_dim // heads))

        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.xavier_uniform_(self.attn_edge)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: [N, in_features], edge_index: [2, E], edge_attr: [E, edge_features]
        N = x.size(0)
        h_node = self.node_proj(x).view(N, self.heads, self.hidden_dim // self.heads)

        if edge_index.size(1) == 0:
            out = F.relu(self.out_proj(h_node.view(N, self.hidden_dim)))
            return (out, torch.zeros(0)) if return_attention else out

        src, dst = edge_index[0], edge_index[1]
        h_edge = self.edge_proj(edge_attr).view(-1, self.heads, self.hidden_dim // self.heads)

        score_src = (h_node[src] * self.attn_src).sum(dim=-1)
        score_dst = (h_node[dst] * self.attn_dst).sum(dim=-1)
        score_e = (h_edge * self.attn_edge).sum(dim=-1)

        scores = F.leaky_relu(score_src + score_dst + score_e, negative_slope=0.2)

        # Softmax over each destination node's incoming edges (not over all
        # edges globally), so every node's attention weights sum to 1 per head
        # regardless of how many neighbors it has.
        exp_scores = torch.exp(scores - scores.max())
        denom = torch.zeros(N, self.heads, dtype=exp_scores.dtype)
        denom.index_add_(0, dst, exp_scores)
        alpha = exp_scores / denom[dst].clamp_min(1e-12)

        out = torch.zeros_like(h_node)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * h_node[src])

        h_out = F.relu(self.out_proj(out.view(N, self.hidden_dim)))
        return (h_out, alpha) if return_attention else h_out


class MeanPoolingEncoder(nn.Module):
    """
    Mean-Pooling GNN Encoder for Encounter Graphs (Ablation Variant 1).
    Same interface as `GATEncoder`, but neighbor features are aggregated with
    uniform 1/in-degree weighting instead of learned attention.
    """

    def __init__(
        self, in_features: int = 6, edge_features: int = 4, hidden_dim: int = 64, heads: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(in_features, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: [N, in_features], edge_index: [2, E]. edge_attr is accepted for
        # interface parity with GATEncoder but unused — pooling is uniform.
        N = x.size(0)
        h_node = self.node_proj(x)

        if edge_index.size(1) == 0:
            out = F.relu(self.out_proj(h_node))
            return (out, torch.zeros(0)) if return_attention else out

        src, dst = edge_index[0], edge_index[1]
        degree = torch.zeros(N, dtype=h_node.dtype)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=h_node.dtype))
        alpha = 1.0 / degree[dst].clamp_min(1.0)

        out = torch.zeros_like(h_node)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * h_node[src])

        h_out = F.relu(self.out_proj(out))
        return (h_out, alpha) if return_attention else h_out


class ActorCriticNet(nn.Module):
    """Multi-Agent PPO Actor-Critic Network."""

    def __init__(self, obs_dim: int = 32, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),  # mean + std log
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.actor(obs)
        mean, log_std = params.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-2.0, max=0.5)
        value = self.critic(obs)
        return mean, log_std.exp(), value


class DeterministicActor(nn.Module):
    """MADDPG decentralized deterministic actor: `own_feats ++ encoder
    embedding` -> a tanh-bounded action (no distribution — MADDPG's policy
    gradient flows through a fixed action, not a sampled log-prob)."""

    def __init__(self, obs_dim: int = 70, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class CentralizedCritic(nn.Module):
    """MADDPG centralized critic: a joint Q-value over every vessel's
    `own_feats ++ embedding` and every vessel's action, concatenated —
    the "centralized training" half of MADDPG's CTDE."""

    def __init__(
        self, n_vessels: int, obs_dim: int = 70, action_dim: int = 2, hidden_dim: int = 64
    ):
        super().__init__()
        in_dim = n_vessels * (obs_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        # joint_obs: [B, n_vessels, obs_dim], joint_actions: [B, n_vessels, action_dim]
        b = joint_obs.size(0)
        x = torch.cat([joint_obs, joint_actions], dim=-1).view(b, -1)
        return self.net(x)
