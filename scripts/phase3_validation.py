#!/usr/bin/env python3
"""
Phase 3 Validation Script:
Runs a 15-vessel port approach scenario and validates the multi-agent env loop & rewards.
Usage:
    python scripts/phase3_validation.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from marlin_twin.data_classes import MaritimeExperimentConfig, VesselAction
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv


def main():
    print("=== MARLIN-Twin Phase 3 Validation Suite ===")

    print("\n1. Initializing 15-vessel port approach scenario...")
    config = MaritimeExperimentConfig(
        scenario_type="port_approach", n_vessels=15, episode_length=150
    )
    env = MaritimeCoordEnv(config)

    obs, info = env.reset(seed=100)
    print(f"   Environment reset successfully. {len(obs)} active vessels in port approach.")

    steps = 100
    team_rewards = []
    min_cpas = []
    trajectories = {vid: [] for vid in obs}

    print("\n2. Simulating 100 timesteps of 15-vessel coordination...")
    for t in range(steps):
        actions = {}
        for vid, agent_obs in obs.items():
            trajectories[vid].append((agent_obs.own_state.x, agent_obs.own_state.y))

            # Rule-based simple action
            actions[vid] = VesselAction(
                vessel_id=vid, propeller_rpm=0.8, rudder_angle=0.0, message_targets=[]
            )

        obs, rewards, team_reward, done, step_info = env.step(actions)
        team_rewards.append(team_reward)
        min_cpas.append(step_info["min_cpa"])

    avg_team_reward = float(np.mean(team_rewards))
    min_fleet_cpa = float(np.min(min_cpas))

    print(f"   Mean Episode Team Reward: {avg_team_reward:.2f}")
    print(f"   Minimum Fleet CPA:        {min_fleet_cpa:.2f} m")

    print("\n3. Generating 15-Vessel Port Approach Fleet Trajectory Figure...")
    fig, ax = plt.subplots(figsize=(8, 8))

    for vid, traj in trajectories.items():
        xs = [pt[0] for pt in traj]
        ys = [pt[1] for pt in traj]
        ax.plot(xs, ys, "-", label=f"Vessel {vid}" if vid < 5 else "")
        ax.plot(xs[0], ys[0], "go", markersize=4)
        ax.plot(xs[-1], ys[-1], "rs", markersize=4)

    ax.plot(0, 0, "k*", markersize=15, label="Port Center Goal")
    ax.set_title("15-Vessel Port Approach Fleet Trajectories", fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.axis("equal")

    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", "phase3_port_approach.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\nValidation plots saved to: {out_path}")
    print("=== Phase 3 Validation Completed Successfully! ===")


if __name__ == "__main__":
    main()
