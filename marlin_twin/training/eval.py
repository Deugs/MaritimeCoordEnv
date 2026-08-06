"""Multi-scenario policy evaluation across encounter types and communication levels."""

import numpy as np
from marlin_twin.api import BaseMaritimeEnvironment, Policy
from marlin_twin.agents.vessel_agent import VesselAgentWrapper
from marlin_twin.training.mappo import _build_scene_graph
from marlin_twin.utils.scoring import compute_safety_score


class MultiScenarioEvaluator:
    """Evaluates policies across multi-vessel encounters, channels, and port approach scenarios."""

    @staticmethod
    def evaluate_scenario(
        env: BaseMaritimeEnvironment,
        policies: dict[int, Policy],
        scenario_name: str = "channel",
        n_episodes: int = 50,
        comms_level: float = 1.0,
    ) -> dict[str, float]:
        env.set_communication_degradation(comms_level)
        rewards = []
        cpas = []
        uses_graph = any(getattr(pol, "USES_GRAPH", False) for pol in policies.values())

        for ep in range(n_episodes):
            obs, info = env.reset(scenario_type=scenario_name, seed=5000 + ep)
            done = False
            ep_rew = 0.0

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

                obs, _, team_reward, done, info = env.step(actions)
                ep_rew += team_reward
                if "min_cpa" in info:
                    cpas.append(info["min_cpa"])

            rewards.append(ep_rew)

        avg_cpa = float(np.mean(cpas)) if cpas else 5000.0
        return {
            "scenario": scenario_name,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_cpa": avg_cpa,
            "safety_score": compute_safety_score(cpas),
        }
