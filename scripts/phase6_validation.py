#!/usr/bin/env python3
"""
Phase 6 Validation Script:
Runs a comparative degradation sweep across MARLIN-Twin, IPPO, MADDPG, and Rule-Based COLREGs,
computes the formal Resilience Index R_resilience, and generates paper-ready publication figures.
Usage:
    python scripts/phase6_validation.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from marlin_twin.data_classes import MaritimeExperimentConfig, VesselAction
from marlin_twin.agents.vessel_agent import VesselAgentWrapper
from marlin_twin.baselines.factory import BaselineFactory
from marlin_twin.utils.metrics import compute_resilience_index

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_common import run_degradation_sweep  # noqa: E402


def main():
    print("=== MARLIN-Twin Phase 6 Validation Suite ===")

    degradation_levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    algorithms = ["marlin_twin", "independent_ppo", "maddpg", "rule_based"]
    alg_labels = {
        "marlin_twin": "MARLIN-Twin (Proposed GAT)",
        "independent_ppo": "Independent PPO (IPPO)",
        "maddpg": "MADDPG Baseline",
        "rule_based": "Rule-Based COLREGs",
    }
    alg_colors = {
        "marlin_twin": "#1f77b4",
        "independent_ppo": "#ff7f0e",
        "maddpg": "#2ca02c",
        "rule_based": "#d62728",
    }

    sweep_results = {alg: [] for alg in algorithms}
    resilience_indices = {}

    print("\n1. Running Comparative Bandwidth Degradation Sweeps across 4 Algorithms...")
    # episode_length=500, not the original 30 -- this vessel class's realistic
    # yaw/thrust response (see VesselDynamics.thrust_coefficient/yaw_coefficient)
    # needs real time to develop any avoidance turn; 30 steps at ~8-9 m/s covers
    # a few hundred meters, nowhere near enough for this scenario's channel
    # crossing to actually occur.
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=3, episode_length=500)
    factory = BaselineFactory(config)
    # 5 seeds, not 1 -- these policies are freshly-initialized/untrained (no
    # checkpoint is loaded here), so a single seed's J(1.0) can land on a
    # near-zero outlier by chance alone; compute_resilience_index divides by
    # J(1.0), so a single unlucky seed can send the whole ratio to a
    # nonsensical value (e.g. R > 1 or, as observed, R > 25).
    eval_seeds = [42, 100, 200, 300, 400]

    def make_select_action(alg):
        def select_action(env, vid, policy, agent_obs, graph, node_idx):
            if alg == "rule_based":
                act_vec = policy.act(agent_obs, deterministic=True)
                return VesselAction(
                    vessel_id=vid,
                    propeller_rpm=float(act_vec[0]),
                    rudder_angle=float(act_vec[1]),
                    message_targets=[],
                )
            wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], policy)
            return wrapper.select_action(agent_obs, graph, node_idx, deterministic=True)

        return select_action

    for alg in algorithms:
        print(f"\n   Testing Algorithm: {alg_labels[alg]}")

        # Uses the same true-per-episode-minimum-pairwise-distance safety
        # metric as scripts/generate_ieee_figures.py's resilience benchmark
        # (see _eval_common.run_degradation_sweep's docstring) -- this used
        # to average every per-step *projected* CPA across the whole
        # episode instead, which reads near-zero in the instant just before
        # a rudder command actually changes heading regardless of how the
        # real trajectory turns out, saturating every algorithm's score.
        scores_per_level = run_degradation_sweep(
            config,
            lambda alg=alg: factory.create(alg),
            degradation_levels,
            eval_seeds,
            make_select_action(alg),
        )
        sweep_results[alg] = [float(np.mean(seed_scores)) for seed_scores in scores_per_level]
        for lam, score in zip(degradation_levels, sweep_results[alg]):
            print(f"      Lambda = {lam:.1f} -> Safety Score J(lambda): {score:.3f}")

        r_idx = compute_resilience_index(degradation_levels, sweep_results[alg])
        resilience_indices[alg] = r_idx
        print(f"   => Resilience Index R_resilience ({alg}): {r_idx:.4f}")

    print("\n2. Generating Publication-Quality Comparative Figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Safety Score vs Degradation Sweep
    for alg in algorithms:
        ax1.plot(
            degradation_levels,
            sweep_results[alg],
            "o-",
            linewidth=2.2,
            color=alg_colors[alg],
            label=f"{alg_labels[alg]} (R={resilience_indices[alg]:.2f})",
        )

    ax1.axhline(0.70, color="gray", linestyle=":", label="Target Sub-Linear Threshold (R >= 0.70)")
    ax1.set_title("Safety Score J(lambda) vs. Communication Degradation", fontweight="bold")
    ax1.set_xlabel("Communication Quality (lambda)")
    ax1.set_ylabel("Normalized Safety Score J(lambda)")
    ax1.set_xlim(1.05, -0.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Bar chart of Resilience Index
    bars = ax2.bar(
        [alg_labels[a] for a in algorithms],
        [resilience_indices[a] for a in algorithms],
        color=[alg_colors[a] for a in algorithms],
    )
    ax2.set_title("Coordination Resilience Index (R_resilience)", fontweight="bold")
    ax2.set_ylabel("Resilience Index Score")
    ax2.set_ylim(0.0, 1.1)
    plt.setp(ax2.get_xticklabels(), rotation=25, ha="right")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    os.makedirs(os.path.join(REPO_ROOT, "figures"), exist_ok=True)
    out_path = os.path.join(REPO_ROOT, "figures", "phase6_benchmark_resilience.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\nValidation plots saved to: {out_path}")
    print("=== Phase 6 Validation Completed Successfully! ===")


if __name__ == "__main__":
    main()
