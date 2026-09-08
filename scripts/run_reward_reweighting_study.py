#!/usr/bin/env python3
"""
Phase 2 -- Reward Reweighting Study:
Sweeps the reward's three weights (safety, colregs, efficiency) for
marlin_twin only, 3 seeds, 500 episodes (matching Phase 1's primary budget
so this study's A0 control row is comparable to Phase 1's e500 marlin_twin
row) -- purely a config sweep, no source changes needed for arms A0-A5.

Motivation (see marlin_twin/envs/maritime_coord_env.py's reward block):
`r_safety = -exp(-min_cpa/200)` is effectively zero for CPA > ~1000m and
only meaningfully negative inside ~500m -- active for maybe 15-25 steps of
a 500-step episode. `r_efficiency = -dist/5000` pays a FLAT +1.0 every step
once the 2-waypoint route completes (radius 50m); in the 300m-gap head_on
scenario at 8 m/s that's ~step 37 of 500, so ~92% of the episode pays a
constant efficiency bonus that dwarfs the safety term's peak magnitude.
A1-A5 probe whether that asymmetry, not the algorithm, is driving "RL goes
straight, ignores safety."

A6/A7 are a separate, more invasive arm: `use_true_separation_for_safety_reward`
(marlin_twin/data_classes.py) changes what r_safety MEASURES (true per-vessel
separation, matching J(lambda)) rather than just its weight -- reported in
its own table section, not folded into the A0-A5 weight sweep, since
conflating "reweight" with "redefine" would misrepresent which change did
what.

Every arm's completion metrics (route-completion rate, mean final distance,
mean speed, sub-100m rate) are tracked alongside J(lambda) -- an arm that
raises J while collapsing task completion is a negative result and must be
reported as one, not cherry-picked out.

Usage:
    python scripts/run_reward_reweighting_study.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _experiment_common import (  # noqa: E402
    TrainingJob,
    checkpoint_exists,
    checkpoint_path,
    run_pool,
    save_results_json,
)

SEEDS = [42, 100, 200]
N_EPISODES = 500
DEGRADATION_LEVELS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
EVAL_SEEDS = list(range(100, 110))

# arm -> (safety_weight, colregs_weight, efficiency_weight, use_true_separation)
ARMS = {
    "A0": (2.0, 1.0, 1.0, False),  # control -- published config
    "A1": (2.0, 1.0, 0.0, False),  # zero the efficiency term entirely
    "A2": (2.0, 1.0, 0.1, False),  # A1 + weak progress signal (anti-loitering)
    "A3": (8.0, 1.0, 1.0, False),  # 4x safety weight
    "A4": (2.0, 4.0, 1.0, False),  # amplify the multiplicative colregs term
    "A5": (8.0, 1.0, 0.1, False),  # maximally realigned corner
    "A6": (2.0, 1.0, 0.1, True),  # A2 + true-separation safety reward
    "A7": (2.0, 1.0, 1.0, True),  # A0 + true-separation safety reward
}


def _train_arm(arm: str, seed: int, n_episodes: int) -> None:
    from marlin_twin.data_classes import MaritimeExperimentConfig
    from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
    from marlin_twin.training.curriculum import TwoStageCurriculumTrainer
    from marlin_twin.agents.policies import GATPolicy
    from marlin_twin.utils.seeding import seed_everything

    seed_everything(seed)
    safety_w, colregs_w, eff_w, use_true_sep = ARMS[arm]
    config = MaritimeExperimentConfig(
        scenario_type="head_on",
        n_vessels=2,
        n_episodes=n_episodes,
        episode_length=500,
        eval_frequency=100,
        safety_reward_weight=safety_w,
        colregs_reward_weight=colregs_w,
        efficiency_reward_weight=eff_w,
        use_true_separation_for_safety_reward=use_true_sep,
    )
    env = MaritimeCoordEnv(config)
    trainer = TwoStageCurriculumTrainer(config)
    trainer.policies = {i: GATPolicy() for i in range(2)}
    trainer.train_curriculum(env, total_episodes=n_episodes)
    trainer.save_checkpoint(checkpoint_path(f"rw{arm}", "marlin_twin", seed))


def _job_fn(job: TrainingJob):
    _train_arm(job.extra["arm"], job.seed, job.n_episodes)
    logger.info(f"[rw{job.extra['arm']}] marlin_twin seed {job.seed}: done")
    return (job.extra["arm"], job.seed)


def build_jobs() -> list[TrainingJob]:
    jobs = []
    for arm in ARMS:
        for seed in SEEDS:
            if checkpoint_exists(f"rw{arm}", "marlin_twin", seed):
                continue
            jobs.append(
                TrainingJob(
                    tag=f"rw{arm}",
                    variant="marlin_twin",
                    seed=seed,
                    n_episodes=N_EPISODES,
                    extra={"arm": arm},
                )
            )
    return jobs


def _run_episode_and_score(env, policies, use_graph: bool, deterministic: bool = True) -> dict:
    """One episode: returns J(lambda)-relevant true-min-distance plus the
    completion metrics (route completion, final distance, mean speed,
    sub-100m closest-approach flag) this study tracks alongside J."""
    from marlin_twin.agents.vessel_agent import VesselAgentWrapper
    from marlin_twin.training.mappo import _build_scene_graph

    obs, _ = env.reset(seed=0)
    done = False
    episode_min_distance = 5000.0
    speeds = []
    n_vessels = len(obs)

    while not done:
        if use_graph:
            graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))
        else:
            graph, node_idx_map = None, {}

        actions = {}
        for vid, agent_obs in obs.items():
            wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], policies[vid])
            actions[vid] = wrapper.select_action(
                agent_obs, graph, node_idx_map.get(vid), deterministic=deterministic
            )

        obs, _, _, done, info = env.step(actions)
        if "true_min_pairwise_distance" in info:
            episode_min_distance = min(episode_min_distance, info["true_min_pairwise_distance"])
        for vid in range(n_vessels):
            speeds.append(env.get_scene().vessels[vid].current_state.speed)

    completed = 0
    final_distances = []
    for vid in range(n_vessels):
        ag = env.get_scene().vessels[vid]
        wp = ag.current_route.current_waypoint()
        if wp is None:
            completed += 1
            final_distances.append(0.0)
        else:
            final_distances.append(float(wp.distance_to(ag.current_state)))

    return {
        "episode_min_distance": episode_min_distance,
        "route_completed_fraction": completed / max(n_vessels, 1),
        "mean_final_distance": float(np.mean(final_distances)),
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "sub_100m": episode_min_distance < 100.0,
    }


def evaluate_arms() -> dict:
    import torch
    from marlin_twin.data_classes import MaritimeExperimentConfig
    from marlin_twin.agents.policies import GATPolicy
    from marlin_twin.baselines.rule_based import RuleBasedCOLREGsController
    from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
    from marlin_twin.utils.metrics import compute_resilience_index

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _eval_common import run_degradation_sweep

    def select_action(env, vid, policy, agent_obs, graph, node_idx):
        from marlin_twin.agents.vessel_agent import VesselAgentWrapper

        wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], policy)
        return wrapper.select_action(agent_obs, graph, node_idx, deterministic=True)

    results = {}
    for arm in ARMS:
        per_seed_curves, per_seed_resilience = [], []
        per_seed_completion, per_seed_final_dist, per_seed_speed, per_seed_sub100 = (
            [],
            [],
            [],
            [],
        )
        eval_config = MaritimeExperimentConfig(
            scenario_type="head_on", n_vessels=2, episode_length=500
        )

        for seed in SEEDS:
            ckpt = checkpoint_path(f"rw{arm}", "marlin_twin", seed)
            if not os.path.exists(ckpt):
                continue

            def factory():
                pols = {i: GATPolicy() for i in range(2)}
                data = torch.load(ckpt, weights_only=True)
                for i in range(2):
                    if i in data:
                        pols[i].set_state(data[i])
                return pols

            scores_per_level = run_degradation_sweep(
                eval_config, factory, DEGRADATION_LEVELS, EVAL_SEEDS, select_action
            )
            curve = [float(np.mean(s)) for s in scores_per_level]
            per_seed_curves.append(curve)
            per_seed_resilience.append(compute_resilience_index(DEGRADATION_LEVELS, curve))

            # Completion metrics at full comms (lambda=1.0), one eval seed.
            env = MaritimeCoordEnv(eval_config)
            policies = factory()
            metrics = _run_episode_and_score(env, policies, use_graph=True)
            per_seed_completion.append(metrics["route_completed_fraction"])
            per_seed_final_dist.append(metrics["mean_final_distance"])
            per_seed_speed.append(metrics["mean_speed"])
            per_seed_sub100.append(metrics["sub_100m"])

        if not per_seed_curves:
            continue
        curves = np.array(per_seed_curves)
        results[arm] = {
            "weights": ARMS[arm],
            "j1_per_seed": curves[:, 0].tolist(),
            "j1_mean": float(curves[:, 0].mean()),
            "j1_std": float(curves[:, 0].std()),
            "resilience_mean": float(np.mean(per_seed_resilience)),
            "resilience_std": float(np.std(per_seed_resilience)),
            "route_completed_fraction_mean": float(np.mean(per_seed_completion)),
            "mean_final_distance_mean": float(np.mean(per_seed_final_dist)),
            "mean_speed_mean": float(np.mean(per_seed_speed)),
            "sub_100m_rate": float(np.mean(per_seed_sub100)),
        }

    # rule_based reference row -- reward-independent, needs no training.
    rb_policies = {i: RuleBasedCOLREGsController(i) for i in range(2)}
    eval_config = MaritimeExperimentConfig(scenario_type="head_on", n_vessels=2, episode_length=500)
    scores_per_level = run_degradation_sweep(
        eval_config, lambda: rb_policies, DEGRADATION_LEVELS, EVAL_SEEDS, select_action
    )
    curve = [float(np.mean(s)) for s in scores_per_level]
    env = MaritimeCoordEnv(eval_config)
    rb_metrics = _run_episode_and_score(env, rb_policies, use_graph=False)
    results["rule_based"] = {
        "weights": None,
        "j1_per_seed": [curve[0]],
        "j1_mean": curve[0],
        "j1_std": 0.0,
        "resilience_mean": compute_resilience_index(DEGRADATION_LEVELS, curve),
        "resilience_std": 0.0,
        "route_completed_fraction_mean": rb_metrics["route_completed_fraction"],
        "mean_final_distance_mean": rb_metrics["mean_final_distance"],
        "mean_speed_mean": rb_metrics["mean_speed"],
        "sub_100m_rate": float(rb_metrics["sub_100m"]),
    }
    return results


def main():
    jobs = build_jobs()
    print(f"=== Phase 2 Reward Reweighting Study: {len(jobs)} jobs remaining (Pool(4)) ===")
    if jobs:
        for result in run_pool(_job_fn, jobs, n_workers=4):
            print(f"  completed: {result}")
    else:
        print("  all checkpoints already exist -- skipping straight to evaluation")

    print("=== Evaluating all arms (A0-A7 + rule_based reference) ===")
    results = evaluate_arms()
    out_path = save_results_json(
        "reward_reweighting_study",
        {"seeds": SEEDS, "n_episodes": N_EPISODES, "eval_seeds": EVAL_SEEDS, "arms": results},
    )
    print(f"=== Results saved to {out_path} ===")

    print(f"\n{'arm':<12}{'weights':<20}{'J(1.0)':<18}{'R':<18}{'completion':<12}{'sub100m':<10}")
    for arm, r in results.items():
        w = str(r["weights"])
        print(
            f"{arm:<12}{w:<20}{r['j1_mean']:.4f}+/-{r['j1_std']:.4f}   "
            f"{r['resilience_mean']:.4f}+/-{r['resilience_std']:.4f}   "
            f"{r['route_completed_fraction_mean']:.2f}        {r['sub_100m_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
