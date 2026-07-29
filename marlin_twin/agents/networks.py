# ============================================================================
# FILE: marlin_twin/agents/networks.py
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATEncoder(nn.Module):
    """
    Graph Attention Network (GAT) Encoder for Encounter Graphs.
    Computes dynamic multi-head attention weights over neighbor vessels.
    """

    def __init__(self, in_features: int = 6, edge_features: int = 4, hidden_dim: int = 64, heads: int = 4):
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x: [N, in_features], edge_index: [2, E], edge_attr: [E, edge_features]
        N = x.size(0)
        h_node = self.node_proj(x).view(N, self.heads, self.hidden_dim // self.heads)

        if edge_index.size(1) == 0:
            return F.relu(self.out_proj(h_node.view(N, self.hidden_dim)))

        src, dst = edge_index[0], edge_index[1]
        h_edge = self.edge_proj(edge_attr).view(-1, self.heads, self.hidden_dim // self.heads)

        score_src = (h_node[src] * self.attn_src).sum(dim=-1)
        score_dst = (h_node[dst] * self.attn_dst).sum(dim=-1)
        score_e = (h_edge * self.attn_edge).sum(dim=-1)

        scores = F.leaky_relu(score_src + score_dst + score_e, negative_slope=0.2)
        alpha = torch.softmax(scores, dim=0)

        out = torch.zeros_like(h_node)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * h_node[src])

        return F.relu(self.out_proj(out.view(N, self.hidden_dim)))


class ActorCriticNet(nn.Module):
    """Multi-Agent PPO Actor-Critic Network."""

    def __init__(self, obs_dim: int = 32, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean + std log
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.actor(obs)
        mean, log_std = params.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-2.0, max=0.5)
        value = self.critic(obs)
        return mean, log_std.exp(), value
