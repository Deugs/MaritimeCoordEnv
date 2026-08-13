#!/usr/bin/env python3
"""
Phase 1 -- Realistic Training Budget Study:
Retrains the 6 baseline/ablation variants at a "fixed-150" re-baseline (the
paper's current published episode count, but WITH the P0.2 curriculum
Stage-2 seed-collapse fix applied) and at a primary "e500" budget (500
episodes, a 3.3x increase), across 5 seeds (42/100/200/300/400) each, plus
SAC at e500 (SAC never had a stale pre-fix checkpoint to re-baseline against)
and the 3 multi-vessel-generalization variants at e500. Milestone checkpoints
at episodes 150/300/500 are saved from each e500 run's on_episode_end hook,
so a single run also yields a training-progress diagnostic for free.

This directly supersedes scripts/run_extended_training.py (retired in Phase
0 -- it never seeded per-seed training, trained at episode_length=40, and
wrote to the same checkpoint filenames the benchmark figures load).

Every checkpoint is written under a NEW tag-prefixed filename
(checkpoints/e150_*.pt, checkpoints/e500_*.pt, checkpoints/e500m150_*.pt,
etc.) -- the currently-published checkpoints/{variant}_seed_{seed}.pt files
are never touched, so the existing figures remain reproducible as "pre-fix"
for the record until Phase 1's results are read and the paper is
regenerated from these new checkpoints.

Usage:
    python scripts/run_training_budget_study.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import time  # noqa: E402

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
    welch_ttest_and_cohens_d,
)

SEEDS = [42, 100, 200, 300, 400]
BASELINE_VARIANTS = [
    "marlin_twin",
    "ablation_mean_pooling",
    "ablation_flat_mlp",
    "ablation_no_digital_twin",
    "independent_ppo",
    "maddpg",
]
MV4_VARIANTS = ["marlin_twin", "independent_ppo", "ablation_flat_mlp"]
BUDGETS = {"e150": 150, "e500": 500}  # tag -> n_episodes
MILESTONE_EPISODES = {150, 300, 500}
EVAL_SEEDS_PRIMARY = list(range(100, 110))  # 10 eval seeds -- see module docstring
EVAL_SEEDS_LEGACY = [100, 101]  # matches the paper's currently-published sweep, for comparability
DEGRADATION_LEVELS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]


def _milestone_saver(tag: str, variant: str, seed: int):
    """Returns an `on_episode_end(ep, trainer)` callback saving a checkpoint
    at each episode count in MILESTONE_EPISODES that is < the run's total
    (the run's own final save covers the full-budget milestone)."""

    def _on_episode_end(ep: int, trainer) -> None:
        completed = ep + 1
        if completed in MILESTONE_EPISODES:
            path = checkpoint_path(f"{tag}m{completed}", variant, seed)
            trainer.save_checkpoint(path)

    return _on_episode_end


def _train_one(tag: str, variant: str, seed: int, n_episodes: int) -> None:
    from marlin_twin.data_classes import MaritimeExperimentConfig
    from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
    from marlin_twin.training.curriculum import TwoStageCurriculumTrainer
    from marlin_twin.training.maddpg import MADDPGTrainer
    from marlin_twin.training.sac import MASACTrainer
    from marlin_twin.agents.policies import GATPolicy, MeanPoolingPolicy, MLPPolicy
    from marlin_twin.baselines.independent_ppo import IndependentPPOPolicy
    from marlin_twin.baselines.maddpg import MADDPGPolicy
    from marlin_twin.baselines.sac import SACPolicy
    from marlin_twin.utils.seeding import seed_everything

    # Must precede ANY policy construction -- see run_retrain_all_baselines.py's
    # retrain_variant docstring for what silently breaks otherwise.
    seed_everything(seed)
    config = MaritimeExperimentConfig(
        scenario_type="head_on",
        n_vessels=2,
        n_episodes=n_episodes,
        episode_length=500,
        eval_frequency=100,
    )
    env = MaritimeCoordEnv(config)

    if variant == "ablation_no_digital_twin":
        env.dt_estimator.enabled = False

    on_episode_end = _milestone_saver(tag, variant, seed)

    if variant == "maddpg":
        trainer = MADDPGTrainer(config)
        trainer.policies = {i: MADDPGPolicy(n_vessels=config.n_vessels) for i in range(2)}
        trainer.train(env, n_episodes=n_episodes, on_episode_end=on_episode_end)
    elif variant == "sac":
        # update_every=2 (the escape hatch documented in training/sac.py's
        # module docstring): measured at ~8s/episode with the default
        # update_every=1, close to double MADDPG's ~4.46s/episode. Halving
        # the gradient-step frequency keeps this study's 5-seed x 500-episode
        # SAC arm from dominating the whole Pool(4) run's wall-clock.
        trainer = MASACTrainer(config, update_every=2)
        trainer.policies = {i: SACPolicy(n_vessels=config.n_vessels) for i in range(2)}
        trainer.train(env, n_episodes=n_episodes, on_episode_end=on_episode_end)
    else:
        trainer = TwoStageCurriculumTrainer(config)
        if variant in ("marlin_twin", "ablation_no_digital_twin"):
            trainer.policies = {i: GATPolicy() for i in range(2)}
        elif variant == "ablation_mean_pooling":
            trainer.policies = {i: MeanPoolingPolicy() for i in range(2)}
        elif variant == "ablation_flat_mlp":
            trainer.policies = {i: MLPPolicy() for i in range(2)}
        elif variant == "independent_ppo":
            trainer.policies = {i: IndependentPPOPolicy() for i in range(2)}
        trainer.train_curriculum(env, total_episodes=n_episodes, on_episode_end=on_episode_end)

    trainer.save_checkpoint(checkpoint_path(tag, variant, seed))
    # Reward-history JSON, so learning curves can be re-plotted without retraining.
    from _experiment_common import RESULTS_DIR
    import json

    os.makedirs(RESULTS_DIR, exist_ok=True)
    history_path = RESULTS_DIR / f"rewardhist_{tag}_{variant}_seed_{seed}.json"
    with open(history_path, "w") as f:
        json.dump({"reward_history": list(map(float, trainer.reward_history))}, f)


def _train_mv4_one(tag: str, variant: str, seed: int, n_episodes: int) -> None:
    from marlin_twin.data_classes import MaritimeExperimentConfig
    from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
    from marlin_twin.training.curriculum import TwoStageCurriculumTrainer
    from marlin_twin.agents.policies import GATPolicy, MLPPolicy
    from marlin_twin.baselines.independent_ppo import IndependentPPOPolicy
    from marlin_twin.utils.seeding import seed_everything

    seed_everything(seed)
    config = MaritimeExperimentConfig(
        scenario_type="multi_vessel_channel_convergence",
        n_vessels=4,
        n_episodes=n_episodes,
        episode_length=500,
        eval_frequency=200,
    )
    env = MaritimeCoordEnv(config)
    trainer = TwoStageCurriculumTrainer(config)
    policy_cls = {
        "marlin_twin": GATPolicy,
        "independent_ppo": IndependentPPOPolicy,
        "ablation_flat_mlp": MLPPolicy,
    }[variant]
    trainer.policies = {i: policy_cls() for i in range(4)}
    trainer.train_curriculum(
        env, total_episodes=n_episodes, on_episode_end=_milestone_saver(f"mv4{tag}", variant, seed)
    )
    trainer.save_checkpoint(checkpoint_path(f"mv4{tag}", variant, seed))


def _job_fn(job: TrainingJob):
    start = time.time()
    if job.extra.get("mv4"):
        _train_mv4_one(job.tag, job.variant, job.seed, job.n_episodes)
    else:
        _train_one(job.tag, job.variant, job.seed, job.n_episodes)
    elapsed = time.time() - start
    logger.info(f"[{job.tag}] {job.variant} seed {job.seed}: done in {elapsed:.1f}s")
    return (job.tag, job.variant, job.seed, elapsed)


def build_jobs() -> list[TrainingJob]:
    jobs = []
    for tag, n_episodes in BUDGETS.items():
        for variant in BASELINE_VARIANTS:
            for seed in SEEDS:
                if checkpoint_exists(tag, variant, seed):
                    continue
                jobs.append(TrainingJob(tag=tag, variant=variant, seed=seed, n_episodes=n_episodes))

    # SAC only at the primary e500 budget -- it never had a stale pre-Phase-0
    # checkpoint to re-baseline against, so no e150 arm is meaningful for it.
    for seed in SEEDS:
        if not checkpoint_exists("e500", "sac", seed):
            jobs.append(TrainingJob(tag="e500", variant="sac", seed=seed, n_episodes=500))

    # mv4 (4-vessel generalization) at e500 only.
    for variant in MV4_VARIANTS:
        for seed in SEEDS:
            if checkpoint_exists("mv4e500", variant, seed):
                continue
            jobs.append(
                TrainingJob(
                    tag="e500", variant=variant, seed=seed, n_episodes=500, extra={"mv4": True}
                )
            )
    return jobs


def evaluate_all(tag: str, variants: list[str], eval_seeds: list[int], mv4: bool = False) -> dict:
    """Runs every variant's `tag`-budget checkpoints through the canonical
    `_eval_common.run_degradation_sweep`, exactly the methodology used for
    every other figure in this repo -- returns per-seed raw J(lambda) curves
    (not just aggregates) plus the resilience index."""
    import torch
    from marlin_twin.data_classes import MaritimeExperimentConfig
    from marlin_twin.agents.policies import GATPolicy, MeanPoolingPolicy, MLPPolicy
    from marlin_twin.agents.vessel_agent import VesselAgentWrapper
    from marlin_twin.baselines.independent_ppo import IndependentPPOPolicy
    from marlin_twin.baselines.maddpg import MADDPGPolicy
    from marlin_twin.baselines.sac import SACPolicy
    from marlin_twin.baselines.rule_based import RuleBasedCOLREGsController
    from marlin_twin.utils.metrics import compute_resilience_index

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _eval_common import run_degradation_sweep

    n_vessels = 4 if mv4 else 2
    scenario = "multi_vessel_channel_convergence" if mv4 else "head_on"
    ckpt_tag = f"mv4{tag}" if mv4 else tag
    policy_cls = {
        "marlin_twin": GATPolicy,
        "ablation_mean_pooling": MeanPoolingPolicy,
        "ablation_flat_mlp": MLPPolicy,
        "ablation_no_digital_twin": GATPolicy,
        "independent_ppo": IndependentPPOPolicy,
        "maddpg": MADDPGPolicy,
        "sac": SACPolicy,
    }

    def select_action(env, vid, policy, agent_obs, graph, node_idx):
        wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], policy)
        return wrapper.select_action(agent_obs, graph, node_idx, deterministic=True)

    def make_factory(variant, seed):
        def factory():
            cls = policy_cls[variant]
            kwargs = {"n_vessels": n_vessels} if cls in (MADDPGPolicy, SACPolicy) else {}
            pols = {i: cls(**kwargs) for i in range(n_vessels)}
            ckpt_file = checkpoint_path(ckpt_tag, variant, seed)
            if os.path.exists(ckpt_file):
                data = torch.load(ckpt_file, weights_only=True)
                for i in range(n_vessels):
                    if i in data:
                        pols[i].set_state(data[i])
            return pols

        return factory

    def make_rule_based_factory():
        return {i: RuleBasedCOLREGsController(i) for i in range(n_vessels)}

    config = MaritimeExperimentConfig(
        scenario_type=scenario, n_vessels=n_vessels, episode_length=500
    )
    results = {}
    for variant in variants:
        per_seed_curves, per_seed_resilience = [], []
        for train_seed in SEEDS:
            if not os.path.exists(checkpoint_path(ckpt_tag, variant, train_seed)):
                continue
            scores_per_level = run_degradation_sweep(
                config,
                make_factory(variant, train_seed),
                DEGRADATION_LEVELS,
                eval_seeds,
                select_action,
            )
            curve = [float(np.mean(s)) for s in scores_per_level]
            per_seed_curves.append(curve)
            per_seed_resilience.append(compute_resilience_index(DEGRADATION_LEVELS, curve))
        if not per_seed_curves:
            continue
        curves = np.array(per_seed_curves)
        results[variant] = {
            "j1_per_seed": curves[:, 0].tolist(),
            "j0_per_seed": curves[:, -1].tolist(),
            "resilience_per_seed": per_seed_resilience,
            "j1_mean": float(curves[:, 0].mean()),
            "j1_std": float(curves[:, 0].std()),
            "resilience_mean": float(np.mean(per_seed_resilience)),
            "resilience_std": float(np.std(per_seed_resilience)),
        }

    # rule_based needs no training -- one constant reference row.
    scores_per_level = run_degradation_sweep(
        config, make_rule_based_factory, DEGRADATION_LEVELS, eval_seeds, select_action
    )
    curve = [float(np.mean(s)) for s in scores_per_level]
    results["rule_based"] = {
        "j1_per_seed": [curve[0]],
        "j0_per_seed": [curve[-1]],
        "resilience_per_seed": [compute_resilience_index(DEGRADATION_LEVELS, curve)],
        "j1_mean": curve[0],
        "j1_std": 0.0,
        "resilience_mean": compute_resilience_index(DEGRADATION_LEVELS, curve),
        "resilience_std": 0.0,
    }
    return results


def main():
    jobs = build_jobs()
    print(f"=== Phase 1 Training Budget Study: {len(jobs)} jobs remaining (Pool(4)) ===")
    if jobs:
        for result in run_pool(_job_fn, jobs, n_workers=4):
            print(f"  completed: {result}")
    else:
        print("  all checkpoints already exist -- skipping straight to evaluation")

    print("=== Evaluating e150 and e500 budgets (10 eval seeds) ===")
    eval_e150 = evaluate_all("e150", BASELINE_VARIANTS, EVAL_SEEDS_PRIMARY)
    eval_e500 = evaluate_all("e500", BASELINE_VARIANTS + ["sac"], EVAL_SEEDS_PRIMARY)
    eval_e500_legacy = evaluate_all("e500", ["marlin_twin"], EVAL_SEEDS_LEGACY)
    eval_mv4 = evaluate_all("e500", MV4_VARIANTS, EVAL_SEEDS_PRIMARY, mv4=True)

    significance = {}
    if "marlin_twin" in eval_e500 and "rule_based" in eval_e500:
        significance["marlin_twin_vs_rule_based_e500"] = welch_ttest_and_cohens_d(
            eval_e500["marlin_twin"]["j1_per_seed"], eval_e500["rule_based"]["j1_per_seed"] * 5
        )
    if "marlin_twin" in eval_e500 and "maddpg" in eval_e500:
        significance["marlin_twin_vs_maddpg_e500"] = welch_ttest_and_cohens_d(
            eval_e500["marlin_twin"]["j1_per_seed"], eval_e500["maddpg"]["j1_per_seed"]
        )
    if "sac" in eval_e500 and "rule_based" in eval_e500:
        significance["sac_vs_rule_based_e500"] = welch_ttest_and_cohens_d(
            eval_e500["sac"]["j1_per_seed"], eval_e500["rule_based"]["j1_per_seed"] * 5
        )

    payload = {
        "seeds": SEEDS,
        "eval_seeds_primary": EVAL_SEEDS_PRIMARY,
        "eval_seeds_legacy": EVAL_SEEDS_LEGACY,
        "degradation_levels": DEGRADATION_LEVELS,
        "e150": eval_e150,
        "e500": eval_e500,
        "e500_legacy_eval_seeds": eval_e500_legacy,
        "mv4_e500": eval_mv4,
        "significance": significance,
    }
    out_path = save_results_json("training_budget_study", payload)
    print(f"=== Results saved to {out_path} ===")

    for tag, results in (("e150", eval_e150), ("e500", eval_e500)):
        print(f"\n--- {tag} ---")
        for variant, r in results.items():
            print(
                f"  {variant}: J(1.0)={r['j1_mean']:.4f}+/-{r['j1_std']:.4f}  "
                f"R={r['resilience_mean']:.4f}+/-{r['resilience_std']:.4f}"
            )


if __name__ == "__main__":
    main()
