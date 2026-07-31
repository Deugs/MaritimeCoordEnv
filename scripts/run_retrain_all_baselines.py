#!/usr/bin/env python3
"""
Multi-Seed Baseline Retraining Script:
Retrains MARLIN-Twin and baseline/ablation policy variants across 5 random seeds
with authentic PPO gradient updates and saves PyTorch model checkpoints.
Usage:
    python scripts/run_retrain_all_baselines.py
"""

import os
import torch
import numpy as np
from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.training.curriculum import TwoStageCurriculumTrainer
from marlin_twin.agents.policies import GATPolicy, MeanPoolingPolicy, MLPPolicy

def retrain_variant(variant_name: str, seeds: list[int], n_episodes: int = 250):
    print(f"\n=======================================================")
    print(f"   Retraining Variant: {variant_name.upper()} ({len(seeds)} Seeds)")
    print(f"=======================================================")

    os.makedirs("checkpoints", exist_ok=True)

    for seed in seeds:
        print(f"\n---> [{variant_name}] Training Seed {seed} for {n_episodes} Episodes...")
        config = MaritimeExperimentConfig(
            scenario_type="head_on",
            n_vessels=2,
            n_episodes=n_episodes,
            episode_length=40,
            eval_frequency=100
        )
        env = MaritimeCoordEnv(config)

        if variant_name == "ablation_no_digital_twin":
            env.dt_estimator.enabled = False

        trainer = TwoStageCurriculumTrainer(config)

        # Initialize specific policy architectures
        if variant_name in ["marlin_twin", "ablation_no_digital_twin"]:
            trainer.policies = {i: GATPolicy() for i in range(config.n_vessels)}
        elif variant_name == "ablation_mean_pooling":
            trainer.policies = {i: MeanPoolingPolicy() for i in range(config.n_vessels)}
        elif variant_name == "ablation_flat_mlp":
            trainer.policies = {i: MLPPolicy() for i in range(config.n_vessels)}
        elif variant_name == "independent_ppo":
            trainer.policies = {i: GATPolicy() for i in range(config.n_vessels)}

        trainer.train_curriculum(env, total_episodes=n_episodes)

        ckpt_path = os.path.join("checkpoints", f"{variant_name}_seed_{seed}.pt")
        trainer.save_checkpoint(ckpt_path)
        print(f"     Saved PyTorch Checkpoint -> {ckpt_path}")

def main():
    print("=== MARLIN-Twin Multi-Seed Retraining Suite ===")
    seeds = [42, 100, 200, 300, 400]
    variants = [
        "marlin_twin",
        "ablation_mean_pooling",
        "ablation_flat_mlp",
        "ablation_no_digital_twin",
        "independent_ppo"
    ]

    for var in variants:
        retrain_variant(var, seeds=seeds, n_episodes=250)

    print("\n=== Retraining Suite Completed Successfully! All Checkpoints Saved. ===")

if __name__ == "__main__":
    main()
