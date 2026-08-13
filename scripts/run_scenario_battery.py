#!/usr/bin/env python3
"""
Phase 3 -- Harder/More Ambiguous Scenario Battery:
Step 3.0 screens Rule-Based COLREGs (which needs no training) plus an
untrained-random-policy floor across every candidate scenario -- the
existing N=4 multi_vessel_channel_convergence (never evaluated with
rule_based before), B1 (port_approach, N=6), B2 (head_on, N=6), and B3
(restricted_visibility_crossing, N=4 and N=6) -- all zero-new-code, already
implemented in scenarios.py. Rule-Based's structural weakness this battery
targets is confirmed by direct read of baselines/rule_based.py: it collapses
every neighbor to the single nearest-by-Euclidean-distance vessel and reacts
only to that one, so it cannot represent two simultaneous conflicting
encounter obligations -- which is exactly what B1's three 120-degree sectors
converging on the origin produce.

Step 3.1 escalates ONLY the scenario where rule_based's screened J(1.0) is
lowest (i.e. the scenario that most exercises rule_based's known structural
limitation) to full training: marlin_twin, independent_ppo,
ablation_flat_mlp, maddpg, and sac, each x 5 seeds x 500 episodes, evaluated
the same way as every other phase (_eval_common.run_degradation_sweep) with
rule_based included -- the first time rule_based is evaluated at N>2 in this
repo's history alongside a fully retrained variant set.

If the screen shows rule_based winning (or tied) everywhere, that is stated
plainly as a valid, publishable finding -- nothing here is designed to
force a different outcome; the screen only selects for scenarios that
exercise a confirmed, existing structural weakness, which is a different
and defensible criterion from "engineered until RL wins."

Usage:
    python scripts/run_scenario_battery.py                 # screen + escalate
    python scripts/run_scenario_battery.py --screen-only    # just Step 3.0
"""

import argparse
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

DEGRADATION_LEVELS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
SCREEN_EVAL_SEEDS = list(range(100, 106))  # 6 seeds -- screening is cheap, no training

# Candidate battery: name -> (scenario_type, n_vessels)
CANDIDATES = {
    "mv4_channel_convergence": ("multi_vessel_channel_convergence", 4),
    "B1_port_approach_n6": ("port_approach", 6),
    "B2_head_on_n6": ("head_on", 6),
    "B3_restricted_visibility_n4": ("restricted_visibility_crossing", 4),
    "B3_restricted_visibility_n6": ("restricted_visibility_crossing", 6),
}

# Full escalation variant set (per Phase 3.1) -- includes SAC and, for the
# first time in this repo's history, rule_based evaluated at N>2 alongside
# a fully retrained variant set.
ESCALATION_VARIANTS = [
    "marlin_twin",
    "independent_ppo",
    "ablation_flat_mlp",
    "maddpg",
    "sac",
]
ESCALATION_SEEDS = [42, 100, 200, 300, 400]
ESCALATION_EPISODES = 500
ESCALATION_EVAL_SEEDS = list(range(100, 110))


def _select_action(env, vid, policy, agent_obs, graph, node_idx):
    from marlin_twin.agents.vessel_agent import VesselAgentWrapper

    wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], policy)
    return wrapper.select_action(agent_obs, graph, node_idx, deterministic=True)


def screen_scenario(name: str, scenario_type: str, n_vessels: int) -> dict:
    """Rule-Based needs no training, so its J(lambda) on any scenario costs
    only evaluation. An untrained-random GATPolicy fleet is run alongside as
    a difficulty floor -- a scenario where even a random policy scores near
    the ceiling isn't discriminative."""
    from marlin_twin.data_classes import MaritimeExperimentConfig
    from marlin_twin.baselines.rule_based import RuleBasedCOLREGsController
    from marlin_twin.agents.policies import GATPolicy
    from marlin_twin.utils.metrics import compute_resilience_index

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _eval_common import run_degradation_sweep

    config = MaritimeExperimentConfig(
        scenario_type=scenario_type, n_vessels=n_vessels, episode_length=500
    )

    def rule_based_factory():
        return {i: RuleBasedCOLREGsController(i) for i in range(n_vessels)}

    def random_policy_factory():
        return {i: GATPolicy() for i in range(n_vessels)}

    rb_scores = run_degradation_sweep(
        config, rule_based_factory, DEGRADATION_LEVELS, SCREEN_EVAL_SEEDS, _select_action
    )
    rb_curve = [float(np.mean(s)) for s in rb_scores]

    random_scores = run_degradation_sweep(
        config, random_policy_factory, DEGRADATION_LEVELS, SCREEN_EVAL_SEEDS, _select_action
    )
    random_curve = [float(np.mean(s)) for s in random_scores]

    return {
        "scenario_type": scenario_type,
        "n_vessels": n_vessels,
        "rule_based_j1": rb_curve[0],
        "rule_based_j0": rb_curve[-1],
        "rule_based_resilience": compute_resilience_index(DEGRADATION_LEVELS, rb_curve),
        "rule_based_curve": rb_curve,
        "random_policy_j1": random_curve[0],
        "random_policy_curve": random_curve,
        # Lower rule_based J(1.0) = closer calls = rule_based struggling more
        # on this scenario's geometry -- the screen's selection criterion.
    }


def run_screening() -> dict:
    results = {}
    for name, (scenario_type, n_vessels) in CANDIDATES.items():
        logger.info(f"[screen] {name} ({scenario_type}, N={n_vessels})...")
        results[name] = screen_scenario(name, scenario_type, n_vessels)
        r = results[name]
        logger.info(
            f"[screen] {name}: rule_based J(1.0)={r['rule_based_j1']:.4f}, "
            f"random J(1.0)={r['random_policy_j1']:.4f}"
        )
    return results


def pick_most_discriminative(screen_results: dict) -> str:
    """Selects the scenario with the LOWEST rule_based J(1.0) among those
    where the random-policy floor is meaningfully worse (i.e. the scenario
    is not trivially easy for everyone) -- the scenario that most exercises
    rule_based's confirmed structural weakness (single-nearest-neighbor
    reduction, unable to represent simultaneous conflicting obligations)."""
    candidates = {
        name: r
        for name, r in screen_results.items()
        if r["rule_based_j1"]
        > r["random_policy_j1"] + 0.02  # rule_based is not merely at the floor
    }
    pool = candidates or screen_results
    return min(pool, key=lambda name: pool[name]["rule_based_j1"])


def _train_escalation_one(
    scenario_name: str, scenario_type: str, n_vessels: int, variant: str, seed: int, n_episodes: int
) -> None:
    from marlin_twin.data_classes import MaritimeExperimentConfig
    from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
    from marlin_twin.training.curriculum import TwoStageCurriculumTrainer
    from marlin_twin.training.maddpg import MADDPGTrainer
    from marlin_twin.training.sac import MASACTrainer
    from marlin_twin.agents.policies import GATPolicy, MLPPolicy
    from marlin_twin.baselines.independent_ppo import IndependentPPOPolicy
    from marlin_twin.baselines.maddpg import MADDPGPolicy
    from marlin_twin.baselines.sac import SACPolicy
    from marlin_twin.utils.seeding import seed_everything

    seed_everything(seed)
    config = MaritimeExperimentConfig(
        scenario_type=scenario_type,
        n_vessels=n_vessels,
        n_episodes=n_episodes,
        episode_length=500,
        eval_frequency=100,
    )
    env = MaritimeCoordEnv(config)
    tag = f"battery_{scenario_name}"

    if variant == "maddpg":
        trainer = MADDPGTrainer(config)
        trainer.policies = {i: MADDPGPolicy(n_vessels=n_vessels) for i in range(n_vessels)}
        trainer.train(env, n_episodes=n_episodes)
    elif variant == "sac":
        trainer = MASACTrainer(config, update_every=2)
        trainer.policies = {i: SACPolicy(n_vessels=n_vessels) for i in range(n_vessels)}
        trainer.train(env, n_episodes=n_episodes)
    else:
        trainer = TwoStageCurriculumTrainer(config)
        policy_cls = {
            "marlin_twin": GATPolicy,
            "independent_ppo": IndependentPPOPolicy,
            "ablation_flat_mlp": MLPPolicy,
        }[variant]
        trainer.policies = {i: policy_cls() for i in range(n_vessels)}
        trainer.train_curriculum(env, total_episodes=n_episodes)

    trainer.save_checkpoint(checkpoint_path(tag, variant, seed))


def _job_fn(job: TrainingJob):
    scenario_type, n_vessels = job.extra["scenario_type"], job.extra["n_vessels"]
    _train_escalation_one(
        job.extra["scenario_name"], scenario_type, n_vessels, job.variant, job.seed, job.n_episodes
    )
    logger.info(f"[battery:{job.extra['scenario_name']}] {job.variant} seed {job.seed}: done")
    return (job.extra["scenario_name"], job.variant, job.seed)


def build_escalation_jobs(scenario_name: str, scenario_type: str, n_vessels: int) -> list:
    jobs = []
    for variant in ESCALATION_VARIANTS:
        for seed in ESCALATION_SEEDS:
            tag = f"battery_{scenario_name}"
            if checkpoint_exists(tag, variant, seed):
                continue
            jobs.append(
                TrainingJob(
                    tag=tag,
                    variant=variant,
                    seed=seed,
                    n_episodes=ESCALATION_EPISODES,
                    extra={
                        "scenario_name": scenario_name,
                        "scenario_type": scenario_type,
                        "n_vessels": n_vessels,
                    },
                )
            )
    return jobs


def evaluate_escalation(scenario_name: str, scenario_type: str, n_vessels: int) -> dict:
    import torch
    from marlin_twin.data_classes import MaritimeExperimentConfig
    from marlin_twin.agents.policies import GATPolicy, MLPPolicy
    from marlin_twin.baselines.independent_ppo import IndependentPPOPolicy
    from marlin_twin.baselines.maddpg import MADDPGPolicy
    from marlin_twin.baselines.sac import SACPolicy
    from marlin_twin.baselines.rule_based import RuleBasedCOLREGsController
    from marlin_twin.utils.metrics import compute_resilience_index

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _eval_common import run_degradation_sweep

    tag = f"battery_{scenario_name}"
    policy_cls = {
        "marlin_twin": GATPolicy,
        "independent_ppo": IndependentPPOPolicy,
        "ablation_flat_mlp": MLPPolicy,
        "maddpg": MADDPGPolicy,
        "sac": SACPolicy,
    }
    config = MaritimeExperimentConfig(
        scenario_type=scenario_type, n_vessels=n_vessels, episode_length=500
    )
    results = {}
    for variant in ESCALATION_VARIANTS:
        per_seed_curves, per_seed_resilience = [], []
        for seed in ESCALATION_SEEDS:
            ckpt = checkpoint_path(tag, variant, seed)
            if not os.path.exists(ckpt):
                continue

            def factory():
                cls = policy_cls[variant]
                kwargs = {"n_vessels": n_vessels} if cls in (MADDPGPolicy, SACPolicy) else {}
                pols = {i: cls(**kwargs) for i in range(n_vessels)}
                data = torch.load(ckpt, weights_only=True)
                for i in range(n_vessels):
                    if i in data:
                        pols[i].set_state(data[i])
                return pols

            scores = run_degradation_sweep(
                config, factory, DEGRADATION_LEVELS, ESCALATION_EVAL_SEEDS, _select_action
            )
            curve = [float(np.mean(s)) for s in scores]
            per_seed_curves.append(curve)
            per_seed_resilience.append(compute_resilience_index(DEGRADATION_LEVELS, curve))
        if not per_seed_curves:
            continue
        curves = np.array(per_seed_curves)
        results[variant] = {
            "j1_per_seed": curves[:, 0].tolist(),
            "j1_mean": float(curves[:, 0].mean()),
            "j1_std": float(curves[:, 0].std()),
            "resilience_mean": float(np.mean(per_seed_resilience)),
            "resilience_std": float(np.std(per_seed_resilience)),
        }

    rb_policies = {i: RuleBasedCOLREGsController(i) for i in range(n_vessels)}
    rb_scores = run_degradation_sweep(
        config, lambda: rb_policies, DEGRADATION_LEVELS, ESCALATION_EVAL_SEEDS, _select_action
    )
    rb_curve = [float(np.mean(s)) for s in rb_scores]
    results["rule_based"] = {
        "j1_per_seed": [rb_curve[0]],
        "j1_mean": rb_curve[0],
        "j1_std": 0.0,
        "resilience_mean": compute_resilience_index(DEGRADATION_LEVELS, rb_curve),
        "resilience_std": 0.0,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()

    print("=== Phase 3 Step 3.0: Screening candidates (rule_based needs no training) ===")
    screen_results = run_screening()
    save_results_json("scenario_battery_screen", {"candidates": screen_results})

    print("\n--- Screen results ---")
    for name, r in screen_results.items():
        print(
            f"  {name}: rule_based J(1.0)={r['rule_based_j1']:.4f}  "
            f"random J(1.0)={r['random_policy_j1']:.4f}  "
            f"rule_based R={r['rule_based_resilience']:.4f}"
        )

    winner = pick_most_discriminative(screen_results)
    print(f"\n=== Most discriminative scenario: {winner} ===")

    if args.screen_only:
        print("--screen-only set, stopping after Step 3.0.")
        return

    scenario_type, n_vessels = CANDIDATES[winner]
    print(f"\n=== Phase 3 Step 3.1: Escalating {winner} ({scenario_type}, N={n_vessels}) ===")
    jobs = build_escalation_jobs(winner, scenario_type, n_vessels)
    print(f"  {len(jobs)} jobs remaining (Pool(4))")
    if jobs:
        for result in run_pool(_job_fn, jobs, n_workers=4):
            print(f"  completed: {result}")

    print("=== Evaluating escalation (including rule_based, first time at N>2) ===")
    eval_results = evaluate_escalation(winner, scenario_type, n_vessels)
    save_results_json(
        "scenario_battery_escalation",
        {
            "winner": winner,
            "scenario_type": scenario_type,
            "n_vessels": n_vessels,
            "seeds": ESCALATION_SEEDS,
            "eval_seeds": ESCALATION_EVAL_SEEDS,
            "results": eval_results,
        },
    )

    print(f"\n--- {winner} results ---")
    for variant, r in eval_results.items():
        print(
            f"  {variant}: J(1.0)={r['j1_mean']:.4f}+/-{r['j1_std']:.4f}  "
            f"R={r['resilience_mean']:.4f}+/-{r['resilience_std']:.4f}"
        )

    rb_j1 = eval_results.get("rule_based", {}).get("j1_mean")
    best_rl_j1 = max(
        (r["j1_mean"] for name, r in eval_results.items() if name != "rule_based"), default=None
    )
    if rb_j1 is not None and best_rl_j1 is not None and rb_j1 >= best_rl_j1:
        print(
            "\nNOTE: Rule-Based COLREGs still matches or beats every retrained RL variant "
            "on this scenario. That is a valid, reportable finding -- this battery selects "
            "scenarios that exercise Rule-Based's confirmed structural weakness, it does not "
            "guarantee RL wins."
        )


if __name__ == "__main__":
    main()
