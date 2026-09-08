"""Multi-Agent SAC (MASAC) trainer: maximum-entropy off-policy CTDE via a
shared replay buffer, twin centralized critics, and automatic entropy
tuning."""

import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.api import BaseTrainer, BaseMaritimeEnvironment, Policy
from marlin_twin.baselines.sac import SACPolicy
from marlin_twin.agents.vessel_agent import VesselAgentWrapper
from marlin_twin.training.mappo import _build_scene_graph, _evaluate_policies
from marlin_twin.training.replay_buffer import ReplayBuffer

MIN_REPLAY_SIZE = 256  # matches MADDPGTrainer's warmup threshold
BATCH_SIZE = 64  # matches MADDPGTrainer
RANDOM_WARMUP_STEPS = 500  # one full episode of uniform-random actions before the actor takes over
GRAD_CLIP_NORM = 1.0  # None disables
# `MaritimeCoordEnv.step`'s `done` is a pure step-count timeout
# (`self.time_step >= self.config.episode_length`), never a true
# terminal/collision state -- so the vessel's value beyond the episode
# horizon is NOT zero. `MADDPGTrainer._update` multiplies by `(1 - dones)`
# anyway, which systematically under-estimates Q at every episode boundary;
# SAC's entropy-augmented target is more sensitive to that same bias, so
# bootstrap through the timeout by default. Flip to False only to A/B this
# choice against MADDPG's convention.
BOOTSTRAP_ON_TIMEOUT = True


class MASACTrainer(BaseTrainer):
    """Multi-Agent Soft Actor-Critic (MASAC) Trainer.

    Off-policy CTDE, structurally mirroring `MADDPGTrainer`: a shared replay
    buffer of joint transitions and a centralized twin-critic pair per
    agent (`SACPolicy`), but with a stochastic squashed-Gaussian actor
    (its own exploration mechanism -- no separate noise process), no target
    actor, no target encoder, and an automatically tuned entropy
    coefficient.
    """

    def __init__(
        self,
        config: MaritimeExperimentConfig,
        batch_size: int = BATCH_SIZE,
        min_replay_size: int = MIN_REPLAY_SIZE,
        update_every: int = 1,
        updates_per_step: int = 1,
        random_warmup_steps: int = RANDOM_WARMUP_STEPS,
        buffer_capacity: int = 100_000,
    ):
        super().__init__(config)
        self.reward_history: list[float] = []
        self.alpha_history: list[float] = []
        self.replay_buffer: ReplayBuffer | None = None
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.update_every = update_every
        self.updates_per_step = updates_per_step
        self.random_warmup_steps = random_warmup_steps
        self.buffer_capacity = buffer_capacity
        self._env_steps = 0

    def train(
        self,
        env: BaseMaritimeEnvironment,
        n_episodes: int,
        seed_offset: int = 0,
        on_episode_end: Callable[[int, "MASACTrainer"], None] | None = None,
    ) -> dict[int, Policy]:
        n_vessels = self.config.n_vessels
        if not self.policies:
            self.policies = {i: SACPolicy(n_vessels=n_vessels) for i in range(n_vessels)}

        logger.info(f"[MASAC] Off-policy training: {n_vessels} agents, {n_episodes} episodes...")

        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_capacity, n_vessels=n_vessels, feat_dim=6, act_dim=2
        )

        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed_offset + ep)
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
                    if self._env_steps < self.random_warmup_steps:
                        a = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
                    else:
                        a = pol.act(agent_obs, graph, n_idx, deterministic=False)
                    act_vec[vid] = a
                    wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], pol)
                    actions[vid] = wrapper.build_action(agent_obs, a)

                next_obs, rewards, team_reward, done, info = env.step(actions)
                ep_reward += team_reward
                self._env_steps += 1

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
                self.replay_buffer.add(
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

                if (
                    len(self.replay_buffer) >= max(self.batch_size, self.min_replay_size)
                    and self._env_steps % self.update_every == 0
                ):
                    for _ in range(self.updates_per_step):
                        self._update(self.replay_buffer, self.batch_size)

            self.reward_history.append(ep_reward)
            self.alpha_history.append(
                float(np.mean([float(p.alpha.detach()) for p in self.policies.values()]))
            )
            if ep % max(1, self.config.eval_frequency) == 0 or ep == n_episodes - 1:
                logger.info(
                    f"Episode {ep}/{n_episodes} - Team Reward: {ep_reward:.2f} - "
                    f"alpha={self.alpha_history[-1]:.3f}"
                )

            if on_episode_end is not None:
                on_episode_end(ep, self)

        return self.policies

    @staticmethod
    def _min_target_q(pol: SACPolicy, next_joint_obs, next_joint_actions) -> torch.Tensor:
        """Twin-Q pessimism: elementwise minimum of the two TARGET critics.
        Using only one critic (or the max) reintroduces the Q-overestimation
        bias the twin critics exist to remove."""
        return torch.min(
            pol.target_critic1(next_joint_obs, next_joint_actions),
            pol.target_critic2(next_joint_obs, next_joint_actions),
        )

    @staticmethod
    def _td_target(reward_i: torch.Tensor, soft_target_q: torch.Tensor, gamma: float, dones):
        if BOOTSTRAP_ON_TIMEOUT:
            return reward_i + gamma * soft_target_q
        return reward_i + gamma * (1.0 - dones) * soft_target_q

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
        node_idx = batch["node_idx"]
        next_graph_batch = batch["next_graph_batch"]
        next_node_idx = batch["next_node_idx"]

        with torch.no_grad():
            # Each agent's own encoder applied to the whole batched graph
            # ONCE (n forwards, not the O(n^2) nested loop MADDPG's target
            # encoder needs) -- SAC has no target encoder, so this is exact,
            # not an approximation, and is the reason twin critics don't
            # cost 2-3x MADDPG's per-update encoder work despite computing
            # more Q-values.
            cur_emb = [
                self.policies[j].encoder(
                    graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr
                )
                for j in range(n_vessels)
            ]
            next_emb = [
                self.policies[j].encoder(
                    next_graph_batch.x, next_graph_batch.edge_index, next_graph_batch.edge_attr
                )
                for j in range(n_vessels)
            ]

            # Each agent's next action + log-prob, from its OWN CURRENT
            # stochastic actor (SAC has no target actor -- this is the
            # entropy-regularized next-action distribution the soft Bellman
            # backup is defined against).
            next_actions_list, next_logps = [], []
            for j in range(n_vessels):
                obs_j = torch.cat([next_own_feats[:, j], next_emb[j][next_node_idx[:, j]]], dim=-1)
                a_j, lp_j = self.policies[j].sample_action(obs_j, deterministic=False)
                next_actions_list.append(a_j)
                next_logps.append(lp_j)
            next_joint_actions = torch.stack(next_actions_list, dim=1)  # [B, n, 2]

        for i in range(n_vessels):
            pol = self.policies[i]

            # ---- critic update -----------------------------------------
            with torch.no_grad():
                next_joint_obs = torch.stack(
                    [
                        torch.cat([next_own_feats[:, j], next_emb[i][next_node_idx[:, j]]], dim=-1)
                        for j in range(n_vessels)
                    ],
                    dim=1,
                )
                target_q = self._min_target_q(pol, next_joint_obs, next_joint_actions)
                # Entropy bonus uses AGENT i's OWN next log-prob -- consistent
                # with the per-agent actor loss below, which maximizes only
                # agent i's own entropy. Summing every agent's log-prob into
                # every agent's target would couple the agents' entropy
                # budgets and make per-agent alpha tuning ill-posed.
                soft_target_q = target_q - pol.alpha.detach() * next_logps[i]
                td_target = self._td_target(rewards[:, i : i + 1], soft_target_q, gamma, dones)

                joint_obs_cur = torch.stack(
                    [
                        torch.cat([own_feats[:, j], cur_emb[i][node_idx[:, j]]], dim=-1)
                        for j in range(n_vessels)
                    ],
                    dim=1,
                )

            q1 = pol.critic1(joint_obs_cur, actions)
            q2 = pol.critic2(joint_obs_cur, actions)
            critic_loss = nn.MSELoss()(q1, td_target) + nn.MSELoss()(q2, td_target)

            pol.critic_optimizer.zero_grad()
            critic_loss.backward()
            if GRAD_CLIP_NORM:
                nn.utils.clip_grad_norm_(
                    list(pol.critic1.parameters()) + list(pol.critic2.parameters()),
                    GRAD_CLIP_NORM,
                )
            pol.critic_optimizer.step()

            # ---- actor update -------------------------------------------
            # Recompute ONLY agent i's own embedding/action WITH gradients
            # and splice it into the (data-only) joint tensors, mirroring
            # MADDPGTrainer's actor update.
            node_emb_i = pol.encoder(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
            own_obs_i = torch.cat([own_feats[:, i], node_emb_i[node_idx[:, i]]], dim=-1)
            own_action_i, own_logp_i = pol.sample_action(own_obs_i, deterministic=False)

            obs_list = [joint_obs_cur[:, j] for j in range(n_vessels)]
            obs_list[i] = own_obs_i
            act_list = [actions[:, j] for j in range(n_vessels)]
            act_list[i] = own_action_i
            joint_obs_for_actor = torch.stack(obs_list, dim=1)
            joint_act_for_actor = torch.stack(act_list, dim=1)

            q1_pi = pol.critic1(joint_obs_for_actor, joint_act_for_actor)
            actor_loss = (pol.alpha.detach() * own_logp_i - q1_pi).mean()

            pol.actor_optimizer.zero_grad()
            actor_loss.backward()
            if GRAD_CLIP_NORM:
                nn.utils.clip_grad_norm_(
                    list(pol.actor.parameters()) + list(pol.encoder.parameters()), GRAD_CLIP_NORM
                )
            pol.actor_optimizer.step()

            # ---- alpha (automatic entropy tuning) ------------------------
            alpha_loss = -(pol.log_alpha * (own_logp_i.detach() + pol.target_entropy)).mean()
            pol.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            pol.alpha_optimizer.step()

            # ---- soft updates: ONLY the two target critics ---------------
            self._soft_update(pol.target_critic1, pol.critic1, tau)
            self._soft_update(pol.target_critic2, pol.critic2, tau)

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
        return _evaluate_policies(env, policies, n_episodes, communication_degradation)

    def save_checkpoint(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint_data = {vid: pol.get_state() for vid, pol in self.policies.items()}
        torch.save(checkpoint_data, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")
        checkpoint_data = torch.load(filepath, weights_only=True)
        n_vessels = self.config.n_vessels
        if not self.policies:
            self.policies = {i: SACPolicy(n_vessels=n_vessels) for i in range(n_vessels)}
        for vid, state in checkpoint_data.items():
            self.policies[vid].set_state(state)
