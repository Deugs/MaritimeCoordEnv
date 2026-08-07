#!/usr/bin/env python3
"""
Phase 5 Validation Script:
Runs a 200-episode 2-Stage Curriculum MAPPO Training run and generates learning curves.
Usage:
    python scripts/phase5_validation.py
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.training.curriculum import TwoStageCurriculumTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=== MARLIN-Twin Phase 5 Validation Suite ===")

    total_episodes = 200
    config = MaritimeExperimentConfig(
        scenario_type="channel",
        n_vessels=3,
        n_episodes=total_episodes,
        episode_length=40,
        eval_frequency=20,
    )
    env = MaritimeCoordEnv(config)
    trainer = TwoStageCurriculumTrainer(config)

    print(f"\n1. Executing {total_episodes}-Episode 2-Stage Curriculum Training...")
    policies = trainer.train_curriculum(env, total_episodes=total_episodes)
    print(f"   Curriculum training completed successfully across {len(policies)} policies.")

    print("\n2. Evaluating Trained Policies under Full Comms vs 50% Degradation...")
    eval_full = trainer.evaluate(env, policies, n_episodes=10, communication_degradation=1.0)
    eval_deg = trainer.evaluate(env, policies, n_episodes=10, communication_degradation=0.5)

    print(
        f"   Full Comms Safety Score:    {eval_full['safety_score']:.3f} | "
        f"Mean Reward: {eval_full['average_reward']:.2f}"
    )
    print(
        f"   50% Comms Deg Safety Score: {eval_deg['safety_score']:.3f} | "
        f"Mean Reward: {eval_deg['average_reward']:.2f}"
    )

    print("\n3. Generating Curriculum Learning Curve Figure...")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Real per-episode team reward from the training run above -- trainer.reward_history
    # accumulates across both curriculum stages (TwoStageCurriculumTrainer.train_curriculum
    # calls self.train(), which appends every episode's reward, and never resets the list
    # between stages), not a hand-crafted exponential-decay illustration.
    all_rewards = np.array(trainer.reward_history[:total_episodes])
    episodes = np.arange(1, len(all_rewards) + 1)
    stage1_cutoff = int(total_episodes * 0.6)

    ax.plot(episodes, all_rewards, "b-", alpha=0.4, label="Raw Episode Reward")
    # Smooth moving average
    window = min(10, len(all_rewards))
    kernel = np.ones(window) / window
    smooth_rewards = np.convolve(all_rewards, kernel, mode="same")
    ax.plot(episodes, smooth_rewards, "b-", linewidth=2.5, label="Smoothed MAPPO (Curriculum)")

    ax.axvline(
        stage1_cutoff, color="r", linestyle="--", label=f"Stage Transition (Ep {stage1_cutoff})"
    )
    ax.set_title("2-Stage Curriculum MAPPO Learning Curve", fontweight="bold")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Team Mean Reward")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    os.makedirs(os.path.join(REPO_ROOT, "figures"), exist_ok=True)
    out_path = os.path.join(REPO_ROOT, "figures", "phase5_training_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\nValidation plots saved to: {out_path}")
    print("=== Phase 5 Validation Completed Successfully! ===")


if __name__ == "__main__":
    main()
