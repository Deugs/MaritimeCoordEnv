#!/usr/bin/env python3
"""
Phase 1 Validation Script:
Runs sea trial maneuvers (Turning Circle & 10/10 Zig-zag) and generates validation plots.
Usage:
    python scripts/phase1_validation.py
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from marlin_twin.data_classes import VesselDynamics, VesselType
from marlin_twin.envs.vessel_dynamics import MMGDynamicsSolver

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=== MARLIN-Twin Phase 1 Validation Suite ===")

    dynamics = VesselDynamics(
        vessel_id=0,
        vessel_type=VesselType.CARGO,
        mass=15000000.0,
        moment_of_inertia=2e9,
        max_rpm=150.0,
        propeller_diameter=4.0,
    )
    solver = MMGDynamicsSolver(dynamics)

    print("\n1. Running Turning Circle Sea Trial (35 deg rudder)...")
    tc_results = solver.run_turning_circle_test(rudder_angle_deg=35.0, duration=400.0)
    print(f"   Tactical Diameter: {tc_results['tactical_diameter']:.2f} m")
    print(f"   Advance:          {tc_results['advance']:.2f} m")
    print(f"   Transfer:         {tc_results['transfer']:.2f} m")

    print("\n2. Running 10/10 Zig-zag Sea Trial...")
    # This vessel's yaw response to a 10 deg rudder is slow (empirically
    # confirmed to need ~5600s to complete both overshoots); a short
    # duration here would leave the maneuver non-convergent and
    # run_zigzag_test would honestly report None rather than a
    # placeholder number.
    zz_duration = 6000.0
    zz_results = solver.run_zigzag_test(angle_deg=10.0, duration=zz_duration)

    def _format_overshoot(angle: float | None, converged: bool) -> str:
        if not converged:
            return f"N/A (did not converge within {zz_duration:.0f}s)"
        return f"{angle:.2f} deg"

    print(
        "   First Overshoot Angle:  "
        + _format_overshoot(
            zz_results["first_overshoot_angle"], zz_results["first_overshoot_converged"]
        )
    )
    print(
        "   Second Overshoot Angle: "
        + _format_overshoot(
            zz_results["second_overshoot_angle"], zz_results["second_overshoot_converged"]
        )
    )

    # Generate Validation Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Turning Circle
    tc_traj = tc_results["trajectory"]
    x_tc = [s.x for s in tc_traj]
    y_tc = [s.y for s in tc_traj]
    ax1.plot(x_tc, y_tc, "b-", label="Trajectory (35 deg Rudder)")
    ax1.set_title("IMO Turning Circle Maneuver", fontweight="bold")
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    ax1.axis("equal")

    # Plot Zig-zag Heading
    zz_traj = zz_results["trajectory"]
    t_zz = np.arange(len(zz_traj))
    heading_deg = [np.degrees(s.heading) for s in zz_traj]
    ax2.plot(t_zz, heading_deg, "r-", label="Heading Angle (deg)")
    ax2.axhline(10.0, color="k", linestyle=":", label="Rudder Target (+10 deg)")
    ax2.axhline(-10.0, color="k", linestyle=":", label="Rudder Target (-10 deg)")
    ax2.set_title("10/10 Zig-zag Maneuver", fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Heading Angle (deg)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    os.makedirs(os.path.join(REPO_ROOT, "figures"), exist_ok=True)
    out_path = os.path.join(REPO_ROOT, "figures", "phase1_sea_trials.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\nValidation plots successfully saved to: {out_path}")
    print("=== Phase 1 Validation Completed Successfully! ===")


if __name__ == "__main__":
    main()
