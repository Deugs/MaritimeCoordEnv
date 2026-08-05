"""Shared communication-degradation sweep helper for the ablation/full-evaluation scripts."""

from pathlib import Path
from typing import Callable

import numpy as np

from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_degradation_sweep(
    config: MaritimeExperimentConfig,
    policies_factory: Callable[[], dict],
    degradation_levels: list[float],
    eval_seeds: list[int],
    select_action: Callable,
    disable_digital_twin: bool = False,
) -> list[list[float]]:
    """Evaluate a policy set across communication degradation levels.

    For each level, runs one episode per seed with a freshly built environment
    and policy set, tracks the minimum inter-vessel distance observed, and
    converts it into a safety score via ``clip(min_dist / 500.0, 0.05, 1.0)``.
    Returns the per-seed safety scores for each degradation level (callers
    aggregate mean/std as needed).
    """
    scores_per_level = []
    for lam in degradation_levels:
        seed_scores = []
        for seed in eval_seeds:
            env = MaritimeCoordEnv(config)
            env.set_communication_degradation(lam)
            if disable_digital_twin:
                env.dt_estimator.enabled = False

            policies = policies_factory()
            obs, _ = env.reset(seed=seed)
            done = False
            min_dist = 5000.0

            while not done:
                actions = {}
                for vid, agent_obs in obs.items():
                    actions[vid] = select_action(env, vid, policies[vid], agent_obs)

                obs, _, _, done, _ = env.step(actions)

                v_ids = list(env.get_scene().vessels.keys())
                if len(v_ids) >= 2:
                    p1 = env.get_scene().vessels[v_ids[0]].current_state.position()
                    p2 = env.get_scene().vessels[v_ids[1]].current_state.position()
                    dist = float(np.linalg.norm(p1 - p2))
                    if dist < min_dist:
                        min_dist = dist

            seed_scores.append(float(np.clip(min_dist / 500.0, 0.05, 1.0)))

        scores_per_level.append(seed_scores)

    return scores_per_level
