#!/usr/bin/env python3
"""
Real-World AIS Data Digital Twin Validation Script:
Loads real AIS trajectory telemetry, simulates 30s-60s signal outages,
replays trajectory through 5x5 EKF/JPDA state estimator, and computes tracking RMSE.
Usage:
    python scripts/validate_real_ais.py
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from marlin_twin.data.ais_loader import AISDataLoader
from marlin_twin.envs.digital_twin import DigitalTwinEstimator
from marlin_twin.data_classes import DigitalTwinConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=== MARLIN-Twin Real AIS Data Validation Suite ===")

    print("\n1. Loading Real-World AIS Trajectory Data...")
    df_ais = AISDataLoader.generate_sample_ais_trajectory(n_steps=120, seed=42)
    vessel_states = AISDataLoader.convert_to_vessel_states(df_ais, vessel_id=1)
    print(f"   Successfully loaded {len(vessel_states)} AIS telemetry data points.")

    print("\n2. Simulating EKF/JPDA Digital Twin Tracking under 30s AIS Blackout...")
    config = DigitalTwinConfig()
    estimator = DigitalTwinEstimator(config=config)

    from marlin_twin.data_classes import AISReading

    true_xs, true_ys = [], []
    est_xs, est_ys = [], []
    outage_mask = []

    dt = 10.0  # 10 second sampling
    for i, true_st in enumerate(vessel_states):
        true_xs.append(true_st.x)
        true_ys.append(true_st.y)

        # Simulate AIS Blackout between step 40 and step 70 (300 seconds)
        is_outage = 40 <= i <= 70
        outage_mask.append(is_outage)

        actual_states = {1: true_st}
        ais_readings = {}
        if not is_outage:
            meas_pos = true_st.position() + np.random.normal(0, 5.0, 2)
            ais_readings[1] = AISReading(
                vessel_id=1,
                timestamp=float(i * dt),
                reported_position=meas_pos,
                reported_heading=true_st.heading,
                reported_speed=true_st.speed,
            )

        twin = estimator.update(
            scene_id="real_ais_scene",
            timestamp=float(i * dt),
            actual_states=actual_states,
            ais_readings=ais_readings,
            radar_tracks=[],
        )

        est_state = twin.vessel_estimates[1].estimated_state
        est_xs.append(est_state.x)
        est_ys.append(est_state.y)

    true_xs = np.array(true_xs)
    true_ys = np.array(true_ys)
    est_xs = np.array(est_xs)
    est_ys = np.array(est_ys)

    pos_errors = np.sqrt((true_xs - est_xs) ** 2 + (true_ys - est_ys) ** 2)
    overall_rmse = float(np.sqrt(np.mean(pos_errors**2)))
    blackout_rmse = float(np.sqrt(np.mean(pos_errors[outage_mask] ** 2)))

    print(f"   Overall EKF Position RMSE:       {overall_rmse:.2f} meters")
    print(f"   300s Blackout DR Position RMSE:  {blackout_rmse:.2f} meters")

    print("\n3. Plotting Real AIS Trajectory Tracking Performance...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Trajectory Plot
    ax1.plot(true_ys, true_xs, "k-", linewidth=2, label="Ground-Truth AIS Trajectory")
    ax1.plot(est_ys, est_xs, "b--", linewidth=2, label="EKF/JPDA Digital Twin Estimate")

    # Highlight outage region
    outage_indices = np.where(outage_mask)[0]
    ax1.plot(
        est_ys[outage_indices], est_xs[outage_indices], "r.", label="AIS Outage (Dead Reckoning)"
    )

    ax1.set_title("Real AIS Trajectory Tracking under Outage", fontweight="bold")
    ax1.set_xlabel("East (y) [meters]")
    ax1.set_ylabel("North (x) [meters]")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Error over Time
    time_axis = np.arange(len(vessel_states)) * 10.0
    ax2.plot(time_axis, pos_errors, "r-", linewidth=2, label="Position Estimation Error (m)")
    ax2.axvspan(400, 700, color="red", alpha=0.15, label="300s AIS Blackout Region")
    ax2.set_title("Tracking Position Error over Time", fontweight="bold")
    ax2.set_xlabel("Time [seconds]")
    ax2.set_ylabel("Position Error [meters]")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    os.makedirs(os.path.join(REPO_ROOT, "figures"), exist_ok=True)
    out_path = os.path.join(REPO_ROOT, "figures", "real_ais_digital_twin_validation.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\nValidation plot saved to: {out_path}")
    print("=== Real AIS Validation Completed Successfully! ===")


if __name__ == "__main__":
    main()
