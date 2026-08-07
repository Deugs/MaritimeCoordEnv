#!/usr/bin/env python3
"""
Real-World AIS Data Digital Twin Validation Script:
Loads a real NOAA MarineCadastre AIS trajectory (marlin_twin/data/real_ais_sample.csv;
see marlin_twin/data/ais_loader.py's module docstring for full provenance), simulates a
300s communication blackout, replays the trajectory through the EKF/JPDA state
estimator at its real (irregular) AIS reporting intervals, and computes tracking RMSE.
This is a thin standalone wrapper around the same
scripts.generate_ieee_figures._run_real_ais_digital_twin() logic that produces the
paper's Figure 11, so the two never drift apart.

Usage:
    python scripts/validate_real_ais.py
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    from generate_ieee_figures import _run_real_ais_digital_twin

    print("=== MARLIN-Twin Real AIS Data Validation Suite ===")

    print("\n1. Loading real NOAA MarineCadastre AIS trajectory...")
    result = _run_real_ais_digital_twin()
    print(
        f"   Loaded {result['n_points']} real AIS reports for vessel "
        f"'{result['vessel_name']}' (MMSI {result['mmsi']})."
    )

    print("\n2. EKF/JPDA Digital Twin tracking against real positions, 300s blackout injected...")
    print(f"   Overall EKF Position RMSE:       {result['overall_rmse']:.2f} meters")
    print(f"   300s Blackout DR Position RMSE:  {result['blackout_rmse']:.2f} meters")

    print("\n3. Plotting real AIS trajectory tracking performance...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    om = result["outage_mask"]
    ax1.plot(result["true_ys"], result["true_xs"], "k-", linewidth=2, label="Real AIS Trajectory")
    ax1.plot(
        result["est_ys"],
        result["est_xs"],
        "b--",
        linewidth=2,
        label="EKF/JPDA Digital Twin Estimate",
    )
    ax1.plot(result["est_ys"][om], result["est_xs"][om], "r.", label="AIS Outage (Dead Reckoning)")
    ax1.set_title("Real AIS Trajectory Tracking under Outage", fontweight="bold")
    ax1.set_xlabel("East (y) [meters]")
    ax1.set_ylabel("North (x) [meters]")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(
        result["time_axis"], result["pos_errors"], "r-", linewidth=2, label="Position Error (m)"
    )
    ax2.axvspan(
        result["outage_start_s"],
        result["outage_end_s"],
        color="red",
        alpha=0.15,
        label="300s AIS Blackout Region",
    )
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
