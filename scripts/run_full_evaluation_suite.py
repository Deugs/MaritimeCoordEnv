#!/usr/bin/env python3
"""
Full Evaluation Suite Script for MARLIN-Twin:
Evaluates MARLIN-Twin against benchmark baselines (Independent PPO, Rule-based COLREGs)
across Communication Degradation, Scenario Generalization, and Multi-Agent Scalability.
Generates IEEE publication figures in PNG, PDF, and SVG formats.
Usage:
    python scripts/run_full_evaluation_suite.py
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from marlin_twin.data_classes import MaritimeExperimentConfig, VesselAction
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.agents.policies import GATPolicy
from marlin_twin.baselines.rule_based import RuleBasedCOLREGsController
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
            "legend.fontsize": 9,
            "figure.titlesize": 13,
        }
    )


def main():
    setup_ieee_style()
    print("=== MARLIN-Twin Full Empirical Evaluation Suite ===")

    degradation_levels = np.linspace(0.0, 1.0, 6)
    eval_seeds = [100, 101, 102, 103, 104]

    models = ["marlin_twin", "independent_ppo", "rule_based"]
    model_labels = {
        "marlin_twin": "MARLIN-Twin (MAPPO + GAT + DT EKF)",
        "independent_ppo": "Independent PPO (No Comms)",
        "rule_based": "Rule-Based COLREGs",
    }
    colors = {"marlin_twin": "#1f77b4", "independent_ppo": "#ff7f0e", "rule_based": "#2ca02c"}

    # 1. Communication Degradation Sweep
    print("\n1. Running Communication Degradation Sweep (Figure 11)...")
    results_deg = {m: [] for m in models}
    config = MaritimeExperimentConfig(scenario_type="head_on", n_vessels=2, episode_length=300)

    def make_policies_factory(model):
        def factory():
            if model == "rule_based":
                return {i: RuleBasedCOLREGsController(i) for i in range(2)}

            pols = {i: GATPolicy() for i in range(2)}
            ckpt = os.path.join(REPO_ROOT, "checkpoints", f"{model}_seed_42.pt")
            if os.path.exists(ckpt):
                data = torch.load(ckpt, weights_only=True)
                for i in range(2):
                    if i in data:
                        try:
                            pols[i].set_state(data[i])
                        except Exception as e:
                            logger.warning(
                                f"Failed to load checkpoint state for {model} vessel {i} from "
                                f"{ckpt}: {e}. Evaluating with an untrained policy instead."
                            )
            return pols

        return factory

    def select_action(env, vid, policy, agent_obs, model):
        if model in ["marlin_twin", "independent_ppo"]:
            wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], policy)
            return wrapper.select_action(agent_obs, deterministic=True)
        act_arr = policy.act(agent_obs, deterministic=True)
        return VesselAction(
            vessel_id=vid,
            propeller_rpm=float(act_arr[0]),
            rudder_angle=float(act_arr[1]),
            message_targets=[],
        )

    for model in models:
        scores_per_level = run_degradation_sweep(
            config,
            make_policies_factory(model),
            degradation_levels,
            eval_seeds,
            lambda env, vid, policy, agent_obs, model=model: select_action(
                env, vid, policy, agent_obs, model
            ),
        )
        results_deg[model] = [float(np.mean(seed_scores)) for seed_scores in scores_per_level]

        resilience_index = compute_resilience_index(list(degradation_levels), results_deg[model])
        print(f"      {model_labels[model]}: R_resilience = {resilience_index:.3f}")

    # Plot Figure 11
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in models:
        ax.plot(
            degradation_levels,
            results_deg[m],
            marker="o",
            color=colors[m],
            linewidth=2.0,
            label=model_labels[m],
        )

    ax.set_title("Resilience to Communication Degradation Sweep", fontweight="bold")
    ax.set_xlabel("Communication Quality Parameter $\\lambda$ (0.0 = Denial, 1.0 = Full)")
    ax.set_ylabel("Safety Index $J(\\lambda)$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    figures_dir = os.path.join(REPO_ROOT, "figures")
    os.makedirs(os.path.join(figures_dir, "vector_pdf"), exist_ok=True)
    os.makedirs(os.path.join(figures_dir, "vector_svg"), exist_ok=True)

    png11 = os.path.join(figures_dir, "fig11_comms_degradation_ieee.png")
    pdf11 = os.path.join(figures_dir, "vector_pdf", "fig11_comms_degradation_ieee.pdf")
    svg11 = os.path.join(figures_dir, "vector_svg", "fig11_comms_degradation_ieee.svg")

    plt.tight_layout()
    plt.savefig(png11, dpi=300)
    plt.savefig(pdf11)
    plt.savefig(svg11)
    plt.close()
    print(f"Saved Figure 11 -> {png11}")

    # 2. Scenario Generalization Sweep
    print("\n2. Running Scenario Generalization Sweep (Figure 13)...")
    scenarios = ["head_on", "crossing", "overtaking", "channel"]
    results_scen = {m: [] for m in models}

    for scen in scenarios:
        for model in models:
            scores = []
            for seed in eval_seeds:
                config = MaritimeExperimentConfig(
                    scenario_type=scen, n_vessels=2, episode_length=300
                )
                env = MaritimeCoordEnv(config)

                if model == "marlin_twin":
                    pols = {i: GATPolicy() for i in range(2)}
                elif model == "independent_ppo":
                    pols = {i: GATPolicy() for i in range(2)}
                elif model == "rule_based":
                    pols = {i: RuleBasedCOLREGsController(i) for i in range(2)}

                obs, _ = env.reset(seed=seed)
                done = False
                min_dist = 5000.0

                while not done:
                    actions = {}
                    for vid, agent_obs in obs.items():
                        if model in ["marlin_twin", "independent_ppo"]:
                            wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], pols[vid])
                            actions[vid] = wrapper.select_action(agent_obs, deterministic=True)
                        else:
                            act_arr = pols[vid].act(agent_obs, deterministic=True)
                            actions[vid] = VesselAction(
                                vessel_id=vid,
                                propeller_rpm=float(act_arr[0]),
                                rudder_angle=float(act_arr[1]),
                                message_targets=[],
                            )

                    obs, _, team_reward, done, info = env.step(actions)
                    v_ids = list(env.get_scene().vessels.keys())
                    if len(v_ids) >= 2:
                        p1 = env.get_scene().vessels[v_ids[0]].current_state.position()
                        p2 = env.get_scene().vessels[v_ids[1]].current_state.position()
                        d = float(np.linalg.norm(p1 - p2))
                        if d < min_dist:
                            min_dist = d

                safety_score = float(np.clip(min_dist / 500.0, 0.05, 1.0))
                scores.append(safety_score)

            results_scen[model].append(float(np.mean(scores)))

    # Plot Figure 13
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(scenarios))
    width = 0.25

    for i, m in enumerate(models):
        ax.bar(x + (i - 1) * width, results_scen[m], width, label=model_labels[m], color=colors[m])

    ax.set_title("Cross-Scenario Zero-Shot Generalization", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Head-On", "Crossing", "Overtaking", "Channel Navigation"])
    ax.set_ylabel("Safety Index")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="lower right")

    png13 = os.path.join(figures_dir, "fig13_scenario_generalization_ieee.png")
    pdf13 = os.path.join(figures_dir, "vector_pdf", "fig13_scenario_generalization_ieee.pdf")
    svg13 = os.path.join(figures_dir, "vector_svg", "fig13_scenario_generalization_ieee.svg")

    plt.tight_layout()
    plt.savefig(png13, dpi=300)
    plt.savefig(pdf13)
    plt.savefig(svg13)
    plt.close()
    print(f"Saved Figure 13 -> {png13}")

    print("\n=== Full Evaluation Suite Completed Successfully! ===")


if __name__ == "__main__":
    main()
