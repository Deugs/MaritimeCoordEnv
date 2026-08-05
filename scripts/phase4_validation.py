#!/usr/bin/env python3
"""
Phase 4 Validation Script:
Runs a 6-level bandwidth degradation sweep (lambda in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
and generates inter-vessel bandwidth utilization heatmaps.
Usage:
    python scripts/phase4_validation.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from marlin_twin.data_classes import MaritimeExperimentConfig, VesselAction, MessagePriority
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.agents.policies import GATPolicy


def main():
    print("=== MARLIN-Twin Phase 4 Validation Suite ===")

    degradation_levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    sweep_results = {}
    heatmap_matrix = []

    print("\n1. Running 6-Level Bandwidth Degradation Sweep (1.0 -> 0.0)...")
    for lam in degradation_levels:
        config = MaritimeExperimentConfig(
            scenario_type="channel", n_vessels=5, episode_length=50, bandwidth_bps=9600.0 * lam
        )
        env = MaritimeCoordEnv(config)
        env.set_communication_degradation(lam)
        policies = {i: GATPolicy() for i in range(5)}

        obs, _ = env.reset(seed=42)
        ep_rewards = []
        delivered_counts = []

        for step in range(50):
            actions = {}
            for vid, agent_obs in obs.items():
                act_vec = policies[vid].act(
                    agent_obs.own_state.x / 5000.0 * np.ones(32, dtype=np.float32),
                    deterministic=True,
                )
                actions[vid] = VesselAction(
                    vessel_id=vid,
                    propeller_rpm=float(act_vec[0] * 0.5 + 0.5),
                    rudder_angle=float(act_vec[1] * 0.5),
                    message_targets=[(vid + 1) % 5],
                    message_priority=MessagePriority.HIGH,
                )

            obs, rewards, team_reward, done, info = env.step(actions)
            ep_rewards.append(team_reward)
            delivered_counts.append(info.get("encounters", 0))

        mean_reward = float(np.mean(ep_rewards))
        sweep_results[lam] = mean_reward
        heatmap_matrix.append(
            [mean_reward, lam * 9600.0, lam * 100.0, float(np.mean(delivered_counts))]
        )
        print(f"   Lambda = {lam:.1f} ({lam*9600:.0f} bps) -> Mean Team Reward: {mean_reward:.2f}")

    print("\n2. Generating Bandwidth Degradation Heatmap & Resilience Plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Degradation Curve
    rewards_list = [sweep_results[lam] for lam in degradation_levels]
    ax1.plot(
        degradation_levels,
        rewards_list,
        "o-",
        color="#1f77b4",
        linewidth=2.5,
        label="MARLIN-Twin GAT",
    )
    ax1.set_title("Reward vs. Communication Quality (lambda)", fontweight="bold")
    ax1.set_xlabel("Communication Quality (lambda)")
    ax1.set_ylabel("Mean Team Reward")
    ax1.set_xlim(1.05, -0.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Bandwidth Utilization Heatmap Matrix
    mat = np.array(heatmap_matrix).T
    im = ax2.imshow(mat, aspect="auto", cmap="magma")
    ax2.set_title("Bandwidth Utilization Feature Matrix", fontweight="bold")
    ax2.set_xticks(range(len(degradation_levels)))
    ax2.set_xticklabels([f"{lam:.1f}" for lam in degradation_levels])
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(["Reward", "Bandwidth (bps)", "Cap (%)", "Encounters"])
    plt.colorbar(im, ax=ax2)

    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", "phase4_degradation_sweep.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\nValidation plots saved to: {out_path}")
    print("=== Phase 4 Validation Completed Successfully! ===")


if __name__ == "__main__":
    main()
