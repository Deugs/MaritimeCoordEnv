"""Multi-Agent PPO (MAPPO) trainer with centralized critic, decentralized actors."""

import os

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from marlin_twin.data_classes import MaritimeExperimentConfig, VoyageEpisode
from marlin_twin.api import BaseTrainer, BaseMaritimeEnvironment, Policy
from marlin_twin.agents.policies import GATPolicy
from marlin_twin.agents.vessel_agent import VesselAgentWrapper
from marlin_twin.envs.encounters import EncounterManager
from marlin_twin.training.rollout_buffer import RolloutBuffer
from marlin_twin.utils.scoring import (
    compute_safety_score,
    compute_efficiency_score,
    compute_colregs_violation_rate,
)


def _build_scene_graph(env: BaseMaritimeEnvironment, vessel_ids, timestamp: float):
    """One shared graph per step, built from the true scene state (not each
    vessel's own possibly-degraded observation) — every vessel's own policy
    indexes its own node out of this same graph via `node_idx_map`.

    Uses the env's current visibility range (defaults to the 5000m CLEAR
    baseline if the env doesn't expose one) as the proximity-edge cutoff, so
    a policy trained/evaluated under degraded visibility doesn't "see
    through" fog while the reward/metrics it's being judged against don't."""
    states = {vid: env.get_scene().vessels[vid].current_state for vid in vessel_ids}
    max_range = getattr(env, "_visibility_range", 5000.0)
    graph = EncounterManager.build_encounter_graph(
        states, timestamp, max_range=max_range
    ).to_pyg_data()
    node_idx_map = {vid: i for i, vid in enumerate(sorted(states.keys()))}
    return graph, node_idx_map


def _evaluate_policies(
    env: BaseMaritimeEnvironment,
    policies: dict[int, Policy],
    n_episodes: int,
    communication_degradation: float,
    return_episodes: bool = False,
    seed_offset: int = 1000,
):
    """Shared deterministic-rollout evaluation loop for `MAPPOTrainer`,
    `MADDPGTrainer`, and `experiment_matrix.run_experiment_matrix`: computes
    real safety/efficiency/COLREGs-violation metrics from the episode data
    rather than the hardcoded stand-ins this used to return
    (`efficiency_score=0.85`, `colregs_violation_rate=0.05`).

    Returns just the aggregate summary dict by default. With
    `return_episodes=True`, also returns a `list[VoyageEpisode]` (one real,
    non-placeholder entry per episode — `transitions` is left empty since
    capturing full per-step scene history is a separate, heavier feature
    this doesn't attempt) as `(summary, episodes)`."""
    env.set_communication_degradation(communication_degradation)
    total_rewards = []
    cpa_list = []
    progress_ratios = []
    fuel_per_step = []
    violation_count = 0
    step_vessel_pairs = 0
    episodes: list[VoyageEpisode] = []
    uses_graph = any(getattr(pol, "USES_GRAPH", False) for pol in policies.values())

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        done = False
        ep_rew = 0.0
        n_steps = 0
        ep_cpa_list: list[float] = []
        ep_violations = 0
        initial_dist = {
            vid: env.get_scene()
            .vessels[vid]
            .current_route.remaining_distance(env.get_scene().vessels[vid].current_state)
            for vid in obs
        }
        fuel_sum = {vid: 0.0 for vid in obs}

        while not done:
            actions = {}
            if uses_graph:
                graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))
            else:
                graph, node_idx_map = None, {}

            for vid, agent_obs in obs.items():
                pol = policies[vid]
                wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], pol)
                actions[vid] = wrapper.select_action(
                    agent_obs, graph, node_idx_map.get(vid), deterministic=True
                )

            for vid, act in actions.items():
                fuel_sum[vid] += abs(act.propeller_rpm) ** 3

            obs, rewards, team_reward, done, info = env.step(actions)
            ep_rew += team_reward
            n_steps += 1
            if "min_cpa" in info:
                cpa_list.append(info["min_cpa"])
                ep_cpa_list.append(info["min_cpa"])
            violation_count += info.get("colregs_violations", 0)
            ep_violations += info.get("colregs_violations", 0)

        total_rewards.append(ep_rew)
        step_vessel_pairs += n_steps * len(fuel_sum)

        for vid, init_dist in initial_dist.items():
            final_dist = (
                env.get_scene()
                .vessels[vid]
                .current_route.remaining_distance(env.get_scene().vessels[vid].current_state)
            )
            ratio = 1.0 - (final_dist / init_dist) if init_dist > 1e-6 else 1.0
            progress_ratios.append(float(np.clip(ratio, 0.0, 1.0)))
            fuel_per_step.append(fuel_sum[vid] / max(n_steps, 1))

        if return_episodes:
            episodes.append(
                VoyageEpisode(
                    episode_id=f"ep{ep}",
                    transitions=[],
                    total_reward=ep_rew,
                    length=n_steps,
                    min_cpa=float(min(ep_cpa_list)) if ep_cpa_list else 5000.0,
                    colregs_violations=ep_violations,
                    fuel_consumed=float(sum(fuel_sum.values())),
                    time_to_destination=float(n_steps),
                    communication_success_rate=float(communication_degradation),
                )
            )

    summary = {
        "average_reward": float(np.mean(total_rewards)),
        "safety_score": compute_safety_score(cpa_list),
        "efficiency_score": compute_efficiency_score(progress_ratios, fuel_per_step),
        "colregs_violation_rate": compute_colregs_violation_rate(
            violation_count, step_vessel_pairs
        ),
        "communication_utilization": communication_degradation * 0.7,
    }
    return (summary, episodes) if return_episodes else summary


class MAPPOTrainer(BaseTrainer):
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) Trainer.
    Supports Centralized Critic with Decentralized Actors (CTDE) and PPO gradient updates.
    """

    def __init__(self, config: MaritimeExperimentConfig):
        super().__init__(config)
        self.reward_history = []

    def train(self, env: BaseMaritimeEnvironment, n_episodes: int) -> dict[int, Policy]:
        n_vessels = self.config.n_vessels
        if not self.policies:
            self.policies = {i: GATPolicy() for i in range(n_vessels)}

        logger.info(f"[MAPPO] PPO training: {n_vessels} agents, {n_episodes} episodes...")

        sample_pol = next(iter(self.policies.values()))
        uses_graph = getattr(sample_pol, "USES_GRAPH", False)
        feat_dim = getattr(sample_pol, "FEAT_DIM", 6)
        buffer = RolloutBuffer(
            buffer_size=self.config.episode_length, n_vessels=n_vessels, feat_dim=feat_dim
        )

        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            done = False
            ep_reward = 0.0
            buffer.clear()

            while not done:
                actions = {}
                feat_vecs = []
                act_vecs = []
                val_vecs = []
                logp_vecs = []
                node_idx_vec = np.zeros(n_vessels, dtype=np.int64)

                if uses_graph:
                    graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))
                else:
                    graph, node_idx_map = None, {}

                for vid, agent_obs in obs.items():
                    pol = self.policies[vid]
                    wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], pol)
                    n_idx = node_idx_map.get(vid)

                    tanh_action, raw_action, val, logp = pol.get_action_and_val(
                        agent_obs, graph, n_idx
                    )
                    act = wrapper.build_action(agent_obs, tanh_action)
                    actions[vid] = act

                    feat_vecs.append(pol.featurize(agent_obs))
                    act_vecs.append(raw_action)
                    val_vecs.append(val)
                    logp_vecs.append(logp)
                    if n_idx is not None:
                        node_idx_vec[vid] = n_idx

                obs, rewards, team_reward, done, info = env.step(actions)
                ep_reward += team_reward

                feat_arr = np.array(feat_vecs, dtype=np.float32)
                act_arr = np.array(act_vecs, dtype=np.float32)
                rew_arr = np.array(
                    [rewards.get(i, 0.0) for i in range(n_vessels)], dtype=np.float32
                )
                val_arr = np.array(val_vecs, dtype=np.float32)
                logp_arr = np.array(logp_vecs, dtype=np.float32)

                buffer.add(
                    feat_arr,
                    act_arr,
                    rew_arr,
                    val_arr,
                    logp_arr,
                    graph=graph,
                    node_idx=node_idx_vec,
                )

            buffer.compute_returns_and_advantages(last_values=np.zeros(n_vessels, dtype=np.float32))
            self.reward_history.append(ep_reward)

            # PPO Gradient Update step
            for vid in range(n_vessels):
                pol = self.policies[vid]
                if hasattr(pol, "optimizer") and hasattr(pol, "evaluate_tensors"):
                    feat_t = torch.tensor(
                        buffer.own_feats_buf[: buffer.ptr, vid], dtype=torch.float32
                    )
                    act_t = torch.tensor(buffer.act_buf[: buffer.ptr, vid], dtype=torch.float32)
                    ret_t = torch.tensor(
                        buffer.ret_buf[: buffer.ptr, vid], dtype=torch.float32
                    ).unsqueeze(-1)
                    adv_t = torch.tensor(
                        buffer.adv_buf[: buffer.ptr, vid], dtype=torch.float32
                    ).unsqueeze(-1)
                    old_logp_t = torch.tensor(
                        buffer.logp_buf[: buffer.ptr, vid], dtype=torch.float32
                    ).unsqueeze(-1)

                    if getattr(pol, "USES_GRAPH", False):
                        batch, local_idx = buffer.batched_graph(vid)

                    for _ in range(4):  # PPO update epochs
                        if getattr(pol, "USES_GRAPH", False):
                            values, log_probs, entropy = pol.evaluate_tensors(
                                feat_t, batch, local_idx, act_t
                            )
                        else:
                            values, log_probs, entropy = pol.evaluate_tensors(feat_t, act_t)
                        ratios = torch.exp(log_probs - old_logp_t)

                        surr1 = ratios * adv_t
                        surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * adv_t
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = nn.MSELoss()(values, ret_t)
                        entropy_loss = -entropy.mean()

                        loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

                        pol.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(pol.net.parameters(), max_norm=0.5)
                        pol.optimizer.step()

            if ep % max(1, self.config.eval_frequency) == 0 or ep == n_episodes - 1:
                logger.info(f"Episode {ep}/{n_episodes} - Team Reward: {ep_reward:.2f}")

        return self.policies

    def evaluate(
        self,
        env: BaseMaritimeEnvironment,
        policies: dict[int, Policy],
        n_episodes: int = 100,
        communication_degradation: float = 1.0,
    ) -> dict[str, float]:
        return _evaluate_policies(env, policies, n_episodes, communication_degradation)

    def save_checkpoint(self, filepath: str) -> None:
        """Saves PyTorch model state dicts for all policies to a checkpoint file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint_data = {vid: pol.get_state() for vid, pol in self.policies.items()}
        torch.save(checkpoint_data, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Loads PyTorch model state dicts from a checkpoint file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        checkpoint_data = torch.load(filepath, weights_only=True)
        for vid, state in checkpoint_data.items():
            if vid in self.policies:
                self.policies[vid].set_state(state)
