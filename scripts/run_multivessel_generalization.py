#!/usr/bin/env python3
"""
Multi-Vessel Scenario Generalization Script:
Retrains MARLIN-Twin (GAT), Independent PPO, and the flat-MLP ablation (Ablation 2)
from scratch on a denser 4-vessel channel-convergence scenario, across the same 4
seeds (42/100/200/300) used elsewhere in this paper, then evaluates each with the
same degradation-sweep safety-score/resilience-index methodology as
generate_ieee_figures.py's fig9/fig12. This is the script behind
Section "Multi-Vessel Scenario Generalization" in paper/main.tex (Table
tab:multivessel_table) -- it exists so that result is reproducible from a committed
script, not only from an ad-hoc analysis run.

Usage:
    python scripts/run_multivessel_generalization.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.training.curriculum import TwoStageCurriculumTrainer
from marlin_twin.agents.policies import GATPolicy, MLPPolicy
from marlin_twin.agents.vessel_agent import VesselAgentWrapper
from marlin_twin.baselines.independent_ppo import IndependentPPOPolicy
from marlin_twin.utils.metrics import compute_resilience_index

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_common import run_degradation_sweep  # noqa: E402

N_VESSELS = 4
SCENARIO = "multi_vessel_channel_convergence"
N_EPISODES = 150
TRAIN_SEEDS = [42, 100, 200, 300]
EVAL_SEEDS = [100, 101]
DEGRADATION_LEVELS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
VARIANTS = ["marlin_twin", "independent_ppo", "ablation_flat_mlp"]
POLICY_CLS = {
    "marlin_twin": GATPolicy,
    "independent_ppo": IndependentPPOPolicy,
    "ablation_flat_mlp": MLPPolicy,
}


def checkpoint_path(variant: str, seed: int) -> str:
    return os.path.join(REPO_ROOT, "checkpoints", f"mv4_{variant}_seed_{seed}.pt")


def train_variant(variant: str, seed: int) -> None:
    config = MaritimeExperimentConfig(
        scenario_type=SCENARIO,
        n_vessels=N_VESSELS,
        n_episodes=N_EPISODES,
        episode_length=500,
        eval_frequency=200,
    )
    env = MaritimeCoordEnv(config)
    trainer = TwoStageCurriculumTrainer(config)
    trainer.policies = {i: POLICY_CLS[variant]() for i in range(N_VESSELS)}
    trainer.train_curriculum(env, total_episodes=N_EPISODES)
    trainer.save_checkpoint(checkpoint_path(variant, seed))


def select_action(env, vid, policy, agent_obs, graph, node_idx):
    wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], policy)
    return wrapper.select_action(agent_obs, graph, node_idx, deterministic=True)


def evaluate_variant(variant: str) -> dict:
    config = MaritimeExperimentConfig(
        scenario_type=SCENARIO, n_vessels=N_VESSELS, episode_length=500
    )

    def make_factory(seed):
        def factory():
            pols = {i: POLICY_CLS[variant]() for i in range(N_VESSELS)}
            ckpt = torch.load(checkpoint_path(variant, seed), weights_only=True)
            for i in range(N_VESSELS):
                pols[i].set_state(ckpt[i])
            return pols

        return factory

    per_seed_curves, per_seed_resilience = [], []
    for seed in TRAIN_SEEDS:
        scores_per_level = run_degradation_sweep(
            config, make_factory(seed), DEGRADATION_LEVELS, EVAL_SEEDS, select_action
        )
        curve = [float(np.mean(s)) for s in scores_per_level]
        per_seed_curves.append(curve)
        per_seed_resilience.append(compute_resilience_index(DEGRADATION_LEVELS, curve))

    curves = np.array(per_seed_curves)
    return {
        "j1_mean": float(curves[:, 0].mean()),
        "j1_std": float(curves[:, 0].std()),
        "j0_mean": float(curves[:, -1].mean()),
        "j0_std": float(curves[:, -1].std()),
        "resilience_mean": float(np.mean(per_seed_resilience)),
        "resilience_std": float(np.std(per_seed_resilience)),
    }


def main():
    print(f"=== Multi-Vessel Generalization: {SCENARIO}, N={N_VESSELS} ===")
    for variant in VARIANTS:
        for seed in TRAIN_SEEDS:
            ckpt = checkpoint_path(variant, seed)
            if os.path.exists(ckpt):
                logger.info(f"[{variant}] seed {seed}: checkpoint exists, skipping training")
                continue
            print(f"Training {variant}, seed {seed}, {N_EPISODES} episodes...")
            train_variant(variant, seed)

    results = {}
    for variant in VARIANTS:
        print(f"Evaluating {variant} across {len(TRAIN_SEEDS)} seeds...")
        results[variant] = evaluate_variant(variant)
        r = results[variant]
        print(
            f"  J(lambda)={r['j1_mean']:.4f} +/- {r['j1_std']:.4f}, "
            f"R_resilience={r['resilience_mean']:.5f} +/- {r['resilience_std']:.4f}"
        )

    print("=== Multi-Vessel Generalization Complete ===")
    return results


if __name__ == "__main__":
    main()
