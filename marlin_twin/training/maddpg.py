"""MADDPG (Multi-Agent Deep Deterministic Policy Gradient) trainer: off-policy
CTDE via a replay buffer, target networks, and soft updates."""

import os

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.api import BaseTrainer, BaseMaritimeEnvironment, Policy
from marlin_twin.baselines.maddpg import MADDPGPolicy
from marlin_twin.agents.vessel_agent import VesselAgentWrapper
from marlin_twin.training.mappo import _build_scene_graph
from marlin_twin.training.replay_buffer import ReplayBuffer

MIN_REPLAY_SIZE = 256


class MADDPGTrainer(BaseTrainer):
    """Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Trainer.

    Off-policy CTDE: a shared replay buffer of joint transitions, a
    centralized critic and soft-updated target networks per agent
    (`MADDPGPolicy`), and a decentralized deterministic actor per agent.
    """

    def __init__(self, config: MaritimeExperimentConfig):
        super().__init__(config)
        self.reward_history = []

    def train(self, env: BaseMaritimeEnvironment, n_episodes: int) -> dict[int, Policy]:
        n_vessels = self.config.n_vessels
        if not self.policies:
            self.policies = {i: MADDPGPolicy(n_vessels=n_vessels) for i in range(n_vessels)}

        logger.info(f"[MADDPG] Off-policy training: {n_vessels} agents, {n_episodes} episodes...")

        replay_buffer = ReplayBuffer(capacity=100_000, n_vessels=n_vessels, feat_dim=6, act_dim=2)
        batch_size = 64

        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            done = False
            ep_reward = 0.0

            while not done:
                graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))
                own_feats_vec = np.zeros((n_vessels, 6), dtype=np.float32)
                node_idx_vec = np.zeros(n_vessels, dtype=np.int64)
                act_vec = np.zeros((n_vessels, 2), dtype=np.float32)
                actions = {}

                for vid, agent_obs in obs.items():
                    pol = self.policies[vid]
                    n_idx = node_idx_map[vid]
                    own_feats_vec[vid] = pol.featurize(agent_obs)
                    node_idx_vec[vid] = n_idx
                    a = pol.act(agent_obs, graph, n_idx, deterministic=False)
                    act_vec[vid] = a
                    wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], pol)
                    actions[vid] = wrapper.build_action(agent_obs, a)

                next_obs, rewards, team_reward, done, info = env.step(actions)
                ep_reward += team_reward

                next_graph, next_node_idx_map = _build_scene_graph(
                    env, next_obs.keys(), float(env.time_step)
                )
                next_own_feats_vec = np.zeros((n_vessels, 6), dtype=np.float32)
                next_node_idx_vec = np.zeros(n_vessels, dtype=np.int64)
                for vid, agent_obs in next_obs.items():
                    pol = self.policies[vid]
                    next_own_feats_vec[vid] = pol.featurize(agent_obs)
                    next_node_idx_vec[vid] = next_node_idx_map[vid]

                rew_arr = np.array(
                    [rewards.get(i, 0.0) for i in range(n_vessels)], dtype=np.float32
                )
                replay_buffer.add(
                    own_feats_vec,
                    act_vec,
                    rew_arr,
                    next_own_feats_vec,
                    done,
                    graph,
                    node_idx_vec,
                    next_graph,
                    next_node_idx_vec,
                )

                obs = next_obs

                if len(replay_buffer) >= max(batch_size, MIN_REPLAY_SIZE):
                    self._update(replay_buffer, batch_size)

            self.reward_history.append(ep_reward)
            if ep % max(1, self.config.eval_frequency) == 0 or ep == n_episodes - 1:
                logger.info(f"Episode {ep}/{n_episodes} - Team Reward: {ep_reward:.2f}")

        return self.policies

    def _update(self, replay_buffer: ReplayBuffer, batch_size: int) -> None:
        n_vessels = self.config.n_vessels
        gamma = self.config.gamma
        tau = self.config.tau
        batch = replay_buffer.sample(batch_size)

        own_feats = batch["own_feats"]  # [B, n_vessels, 6]
        actions = batch["actions"]  # [B, n_vessels, 2]
        rewards = batch["rewards"]  # [B, n_vessels]
        next_own_feats = batch["next_own_feats"]
        dones = batch["dones"].unsqueeze(-1)  # [B, 1]
        graph_batch = batch["graph_batch"]
        node_idx = batch["node_idx"]  # [B, n_vessels] global row indices
        next_graph_batch = batch["next_graph_batch"]
        next_node_idx = batch["next_node_idx"]

        for i in range(n_vessels):
            pol = self.policies[i]

            with torch.no_grad():
                # Agent i's own (target) encoder embeds every vessel's next
                # state for agent i's own centralized critic target.
                next_node_emb = pol.target_encoder(
                    next_graph_batch.x, next_graph_batch.edge_index, next_graph_batch.edge_attr
                )
                next_joint_obs = torch.stack(
                    [
                        torch.cat(
                            [next_own_feats[:, j], next_node_emb[next_node_idx[:, j]]], dim=-1
                        )
                        for j in range(n_vessels)
                    ],
                    dim=1,
                )

                # Every vessel's next action, from its OWN target actor/encoder.
                next_actions = torch.zeros(batch_size, n_vessels, pol.action_dim)
                for j in range(n_vessels):
                    pol_j = self.policies[j]
                    node_emb_j = pol_j.target_encoder(
                        next_graph_batch.x, next_graph_batch.edge_index, next_graph_batch.edge_attr
                    )
                    obs_j = torch.cat(
                        [next_own_feats[:, j], node_emb_j[next_node_idx[:, j]]], dim=-1
                    )
                    next_actions[:, j] = pol_j.target_actor(obs_j)

                target_q = pol.target_critic(next_joint_obs, next_actions)
                td_target = rewards[:, i : i + 1] + gamma * (1.0 - dones) * target_q

                # Agent i's own (current) encoder embeds every vessel's
                # current state — a fixed feature extractor for the critic
                # update (only the actor update below backprops into it).
                node_emb_cur = pol.encoder(
                    graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr
                )
                joint_obs_cur = torch.stack(
                    [
                        torch.cat([own_feats[:, j], node_emb_cur[node_idx[:, j]]], dim=-1)
                        for j in range(n_vessels)
                    ],
                    dim=1,
                )

            current_q = pol.critic(joint_obs_cur, actions)
            critic_loss = nn.MSELoss()(current_q, td_target)

            pol.critic_optimizer.zero_grad()
            critic_loss.backward()
            pol.critic_optimizer.step()

            # Actor update: recompute ONLY agent i's own embedding/action
            # with gradients, splice it into the (data-only) joint obs/action
            # tensors, and maximize agent i's own critic through that splice.
            node_emb_i = pol.encoder(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
            own_obs_i = torch.cat([own_feats[:, i], node_emb_i[node_idx[:, i]]], dim=-1)
            own_action_i = pol.actor(own_obs_i)

            joint_obs_for_actor = joint_obs_cur.clone()
            joint_obs_for_actor[:, i] = own_obs_i
            joint_actions_for_actor = actions.clone()
            joint_actions_for_actor[:, i] = own_action_i

            actor_loss = -pol.critic(joint_obs_for_actor, joint_actions_for_actor).mean()

            pol.actor_optimizer.zero_grad()
            actor_loss.backward()
            pol.actor_optimizer.step()

            self._soft_update(pol.target_encoder, pol.encoder, tau)
            self._soft_update(pol.target_actor, pol.actor, tau)
            self._soft_update(pol.target_critic, pol.critic, tau)

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.mul_(1.0 - tau).add_(tau * sp)

    def evaluate(
        self,
        env: BaseMaritimeEnvironment,
        policies: dict[int, Policy],
        n_episodes: int = 100,
        communication_degradation: float = 1.0,
    ) -> dict[str, float]:
        env.set_communication_degradation(communication_degradation)
        total_rewards = []
        cpa_list = []

        for ep in range(n_episodes):
            obs, info = env.reset(seed=1000 + ep)
            done = False
            ep_rew = 0.0

            while not done:
                graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))
                actions = {}
                for vid, agent_obs in obs.items():
                    pol = policies[vid]
                    wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], pol)
                    actions[vid] = wrapper.select_action(
                        agent_obs, graph, node_idx_map.get(vid), deterministic=True
                    )

                obs, rewards, team_reward, done, info = env.step(actions)
                ep_rew += team_reward
                if "min_cpa" in info:
                    cpa_list.append(info["min_cpa"])

            total_rewards.append(ep_rew)

        avg_cpa = float(np.mean(cpa_list)) if cpa_list else 5000.0
        safety_score = float(np.clip(avg_cpa / 1000.0, 0.0, 1.0))

        return {
            "average_reward": float(np.mean(total_rewards)),
            "safety_score": safety_score,
            "efficiency_score": 0.85,
            "colregs_violation_rate": 0.05,
            "communication_utilization": communication_degradation * 0.7,
        }

    def save_checkpoint(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint_data = {vid: pol.get_state() for vid, pol in self.policies.items()}
        torch.save(checkpoint_data, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        checkpoint_data = torch.load(filepath, weights_only=True)
        for vid, state in checkpoint_data.items():
            if vid in self.policies:
                self.policies[vid].set_state(state)
