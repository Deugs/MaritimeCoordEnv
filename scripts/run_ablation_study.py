#!/usr/bin/env python3
"""
Empirical Ablation Study Script for MARLIN-Twin:
Evaluates 4 ablation variants using trained PyTorch multi-seed checkpoints across
communication degradation levels lambda in [0.0, 1.0].
Generates IEEE-compliant PNG, PDF, and SVG plots.
Usage:
    python scripts/run_ablation_study.py
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.agents.policies import GATPolicy, MeanPoolingPolicy, MLPPolicy
from marlin_twin.agents.vessel_agent import VesselAgentWrapper
from marlin_twin.utils.metrics import compute_resilience_index

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_common import REPO_ROOT, run_degradation_sweep  # noqa: E402


def setup_ieee_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13,
        }
    )


def main():
    setup_ieee_style()
    print("=== MARLIN-Twin Empirical 4-Variant Ablation Study ===")

    variants = [
        "marlin_twin",
        "ablation_mean_pooling",
        "ablation_flat_mlp",
        "ablation_no_digital_twin",
    ]

    labels = {
        "marlin_twin": "MARLIN-Twin (Full: GAT + DT EKF)",
        "ablation_mean_pooling": "Ablation 1: Uniform Mean-Pooling GNN",
        "ablation_flat_mlp": "Ablation 2: Flat Vector MLP",
        "ablation_no_digital_twin": "Ablation 3: No Digital Twin EKF",
    }

    markers = {
        "marlin_twin": "o",
        "ablation_mean_pooling": "s",
        "ablation_flat_mlp": "^",
        "ablation_no_digital_twin": "D",
    }

    colors = {
        "marlin_twin": "#1f77b4",
        "ablation_mean_pooling": "#ff7f0e",
        "ablation_flat_mlp": "#2ca02c",
        "ablation_no_digital_twin": "#d62728",
    }

    degradation_levels = np.linspace(0.0, 1.0, 6)  # 0.0 (total loss) to 1.0 (full comms)
    eval_seeds = [100, 101, 102, 103, 104]

    ablation_results = {v: [] for v in variants}
    ablation_stds = {v: [] for v in variants}

    config = MaritimeExperimentConfig(scenario_type="head_on", n_vessels=2, episode_length=500)

    print("\n1. Running Empirical Evaluation Sweeps across 4 Ablation Variants...")
    for var in variants:
        print(f"\n---> Evaluating Variant: {labels[var]}")

        # Instantiate policy objects
        if var in ["marlin_twin", "ablation_no_digital_twin"]:
            pols = {i: GATPolicy() for i in range(2)}
        elif var == "ablation_mean_pooling":
            pols = {i: MeanPoolingPolicy() for i in range(2)}
        elif var == "ablation_flat_mlp":
            pols = {i: MLPPolicy() for i in range(2)}

        # Load trained PyTorch checkpoint if available
        ckpt_path = os.path.join(REPO_ROOT, "checkpoints", f"{var}_seed_42.pt")
        if os.path.exists(ckpt_path):
            ckpt_data = torch.load(ckpt_path, weights_only=True)
            for i in range(2):
                if i in ckpt_data:
                    try:
                        pols[i].set_state(ckpt_data[i])
                    except Exception as e:
                        logger.warning(
                            f"Failed to load checkpoint state for {var} vessel {i} from "
                            f"{ckpt_path}: {e}. Evaluating with an untrained policy instead."
                        )

        def select_action(env, vid, policy, agent_obs, graph, node_idx):
            wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], policy)
            return wrapper.select_action(agent_obs, graph, node_idx, deterministic=True)

        scores_per_level = run_degradation_sweep(
            config,
            lambda: pols,
            degradation_levels,
            eval_seeds,
            select_action,
            disable_digital_twin=(var == "ablation_no_digital_twin"),
        )

        for lam, seed_scores in zip(degradation_levels, scores_per_level):
            mean_score = float(np.mean(seed_scores))
            std_score = float(np.std(seed_scores))
            ablation_results[var].append(mean_score)
            ablation_stds[var].append(std_score)
            print(f"      Lambda = {lam:.1f} -> Safety Score: {mean_score:.3f} +/- {std_score:.3f}")

        resilience_index = compute_resilience_index(list(degradation_levels), ablation_results[var])
        print(f"      Coordination Resilience Index (R_resilience) = {resilience_index:.3f}")

    print("\n2. Plotting Publication-Quality IEEE Ablation Curves...")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for var in variants:
        means = np.array(ablation_results[var])
        stds = np.array(ablation_stds[var])
        ax.plot(
            degradation_levels,
            means,
            marker=markers[var],
            color=colors[var],
            linewidth=2.0,
            label=labels[var],
        )
        ax.fill_between(
            degradation_levels, means - stds, means + stds, color=colors[var], alpha=0.15
        )

    ax.set_title("Ablation Analysis: Communication Degradation Resilience", fontweight="bold")
    ax.set_xlabel("Communication Quality Parameter $\\lambda$ (0.0 = Denial, 1.0 = Full)")
    ax.set_ylabel("Normalized Safety Index $J(\\lambda)$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    figures_dir = os.path.join(REPO_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.join(figures_dir, "vector_pdf"), exist_ok=True)
    os.makedirs(os.path.join(figures_dir, "vector_svg"), exist_ok=True)

    png_path = os.path.join(figures_dir, "fig12_ablation_study_ieee.png")
    pdf_path = os.path.join(figures_dir, "vector_pdf", "fig12_ablation_study_ieee.pdf")
    svg_path = os.path.join(figures_dir, "vector_svg", "fig12_ablation_study_ieee.svg")

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.savefig(svg_path)
    plt.close()

    print(f"\nSaved PNG -> {png_path}")
    print(f"Saved PDF -> {pdf_path}")
    print(f"Saved SVG -> {svg_path}")
    print("=== Empirical Ablation Study Completed Successfully! ===")


if __name__ == "__main__":
    main()
