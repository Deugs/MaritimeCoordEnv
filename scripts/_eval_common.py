"""Shared communication-degradation sweep helper for the ablation/full-evaluation scripts."""

from pathlib import Path
from typing import Callable

from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.training.mappo import _build_scene_graph
from marlin_twin.utils.scoring import compute_safety_score

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
    and policy set, reduces the episode to its true minimum pairwise vessel
    separation (`info["true_min_pairwise_distance"]`, the real Euclidean
    distance between every vessel pair's actual position each step -- not
    `info["min_cpa"]`, which is a per-step *projected* CPA from a linear
    velocity extrapolation and reads near-zero in the instant just before a
    rudder command actually changes heading, regardless of how safe the
    real, curving trajectory turns out to be), and converts that single
    per-episode value into the canonical `compute_safety_score` — the same
    formula used by every other evaluator, and one that generalizes past 2
    vessels (unlike the raw pairwise-distance-between-the-first-two-vessels
    this used to track). `compute_safety_score`'s own docstring calls for
    one real per-episode value, not a per-step time series -- averaging
    every step (including the many steps with no nearby vessel, which
    default to 5000m) into one number dilutes a real close encounter into
    insignificance, which is what made every degradation level look
    identically "safe" regardless of the actual encounter outcome.
    Returns the per-seed safety scores for each degradation level (callers
    aggregate mean/std as needed).

    `select_action(env, vid, policy, agent_obs, graph, node_idx)` — `graph`/
    `node_idx` are the shared per-step scene graph and this vessel's node
    index within it, built once per step if any policy in the set is
    graph-based (`USES_GRAPH`); both are `None` otherwise.
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
            uses_graph = any(getattr(p, "USES_GRAPH", False) for p in policies.values())
            obs, _ = env.reset(seed=seed)
            done = False
            episode_min_distance = 5000.0

            while not done:
                if uses_graph:
                    graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))
                else:
                    graph, node_idx_map = None, {}

                actions = {}
                for vid, agent_obs in obs.items():
                    actions[vid] = select_action(
                        env, vid, policies[vid], agent_obs, graph, node_idx_map.get(vid)
                    )

                obs, _, _, done, info = env.step(actions)
                if "true_min_pairwise_distance" in info:
                    episode_min_distance = min(
                        episode_min_distance, info["true_min_pairwise_distance"]
                    )

            seed_scores.append(compute_safety_score([episode_min_distance]))

        scores_per_level.append(seed_scores)

    return scores_per_level
