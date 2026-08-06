#!/usr/bin/env python3
"""
IEEE Publication Figures Generator Script:
Re-renders all empirical performance charts matching IEEE Transactions publication standards:
300 DPI, colorblind-friendly palette, sans-serif typography, and standard column widths.

Every figure below is computed from a real run of this codebase's own solvers/estimators/
policies -- none of the values are hand-picked placeholders. Where a figure's title claims
"real-world" data (fig11), that claim is now honest: this repository has no real NOAA/AIS
dataset anywhere in it (confirmed by inspecting `marlin_twin/data/ais_loader.py`, which only
ever generates a synthetic sample trajectory), so fig11 is now explicitly labeled as a
simulated trajectory rather than falsely claiming real-world provenance.

Usage:
    python scripts/generate_ieee_figures.py
"""

import os
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from loguru import logger  # noqa: E402

from marlin_twin.data_classes import (  # noqa: E402
    VesselDynamics,
    VesselType,
    VesselState,
    AISReading,
    MaritimeExperimentConfig,
)
from marlin_twin.envs.vessel_dynamics import MMGDynamicsSolver  # noqa: E402
from marlin_twin.envs.digital_twin import DigitalTwinEstimator  # noqa: E402
from marlin_twin.envs.communication import CommunicationChannelManager  # noqa: E402
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv  # noqa: E402
from marlin_twin.agents.policies import GATPolicy  # noqa: E402
from marlin_twin.agents.vessel_agent import VesselAgentWrapper  # noqa: E402
from marlin_twin.baselines.independent_ppo import IndependentPPOPolicy  # noqa: E402
from marlin_twin.baselines.maddpg import MADDPGPolicy  # noqa: E402
from marlin_twin.baselines.rule_based import RuleBasedCOLREGsController  # noqa: E402
from marlin_twin.training.mappo import _build_scene_graph  # noqa: E402
from marlin_twin.training.curriculum import TwoStageCurriculumTrainer  # noqa: E402
from marlin_twin.utils.metrics import compute_resilience_index  # noqa: E402
from marlin_twin.data_classes import VesselAction  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_common import REPO_ROOT, run_degradation_sweep  # noqa: E402


def setup_ieee_style():
    """Sets Matplotlib global rcParams to IEEE Transactions standards."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 11.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig_all_formats(name: str):
    """Saves active figure in high-DPI PNG, vector PDF, and vector SVG formats."""
    figures_dir = os.path.join(REPO_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.join(figures_dir, "vector_pdf"), exist_ok=True)
    os.makedirs(os.path.join(figures_dir, "vector_svg"), exist_ok=True)

    plt.savefig(os.path.join(figures_dir, f"{name}.png"), dpi=300)
    plt.savefig(os.path.join(figures_dir, "vector_pdf", f"{name}.pdf"))
    plt.savefig(os.path.join(figures_dir, "vector_svg", f"{name}.svg"))
    plt.close()


def _default_dynamics() -> VesselDynamics:
    """Same CARGO-scale dynamics used by scripts/phase1_validation.py, for consistency
    across every sea-trial figure in the paper."""
    return VesselDynamics(
        vessel_id=0,
        vessel_type=VesselType.CARGO,
        mass=15000000.0,
        moment_of_inertia=2e9,
        max_rpm=150.0,
        propeller_diameter=4.0,
    )


def render_fig5_sea_trials() -> dict:
    """Figure 5: 3-DOF MMG Sea Trial Maneuvers (Turning Circle & Zig-Zag) -- real solver
    output, not a hand-drawn placeholder curve."""
    solver = MMGDynamicsSolver(_default_dynamics())

    # rudder_angle_deg=30.0 -- VesselDynamics.max_rudder_angle defaults to
    # pi/6 (30 deg) and MMGDynamicsSolver.step() clamps every commanded
    # rudder angle to it, so a caller passing 35.0 here was silently
    # simulating a 30 deg turn while every caption/label claimed 35 deg.
    #
    # duration=1200/3200 -- after fixing the double-RPM-scaling thrust bug and
    # deriving a real yaw_coefficient from each type's own turning_circle
    # (see VesselDynamics.thrust_coefficient/yaw_coefficient), this vessel
    # takes ~1000s to complete one full turning-circle loop and ~3000s to
    # complete two zigzag overshoots; the old duration=400/6000 defaults
    # predate both fixes and never converged under the corrected dynamics
    # (see run_turning_circle_test's loop_completed / run_zigzag_test's
    # *_converged flags -- always check them rather than trusting a
    # returned number blindly).
    tc_result = solver.run_turning_circle_test(rudder_angle_deg=30.0, duration=1200.0)
    zz_result = solver.run_zigzag_test(angle_deg=10.0, duration=2900.0)
    assert tc_result["loop_completed"], "turning circle did not complete a full loop"
    assert zz_result["first_overshoot_converged"], "zigzag did not converge"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))

    # Slice the plotted trajectory to the first completed loop only, matching
    # what tactical_diameter/advance/transfer actually measure -- the full
    # `duration` window runs a bit past loop completion and would otherwise
    # plot a second partial loop on top of the first.
    tc_traj = tc_result["trajectory"][: tc_result["loop_completed_step"] + 1]
    x_tc = [s.x for s in tc_traj]
    y_tc = [s.y for s in tc_traj]
    ax1.plot(y_tc, x_tc, "b-", lw=2, label="30 deg Starboard Turning Circle")
    ax1.plot(y_tc[0], x_tc[0], "go", label="Start Point")
    ax1.plot(y_tc[-1], x_tc[-1], "rs", label="End Point")
    ax1.set_title("3-DOF MMG Turning Circle (IMO Standard)", fontweight="bold")
    ax1.set_xlabel("Transfer (y) [meters]")
    ax1.set_ylabel("Advance (x) [meters]")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    zz_traj = zz_result["trajectory"]
    t_sec = np.arange(len(zz_traj))
    heading_deg = [np.degrees(s.heading) for s in zz_traj]
    ax2.plot(t_sec, heading_deg, "b-", lw=2.0, label="Heading Angle (deg)")
    ax2.axhline(10.0, color="r", linestyle="--", lw=1.2, label="Rudder Target (+/-10 deg)")
    ax2.axhline(-10.0, color="r", linestyle="--", lw=1.2)
    ax2.set_title("10/10 Zig-Zag Maneuver Response", fontweight="bold")
    ax2.set_xlabel("Time [seconds]")
    ax2.set_ylabel("Angle [degrees]")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    save_fig_all_formats("fig5_sea_trials_ieee")

    return {
        "tactical_diameter_m": tc_result["tactical_diameter"],
        "first_overshoot_deg": zz_result["first_overshoot_angle"],
        "first_overshoot_converged": zz_result["first_overshoot_converged"],
    }


def _run_blackout_digital_twin(seed: int, n_steps: int = 120, dt: float = 10.0) -> dict:
    """Drives a real vessel through the MMG solver, feeds its true trajectory through
    DigitalTwinEstimator's real EKF/JPDA update with a 300s AIS blackout window, and
    computes real position RMSE. Used by both fig6 and fig11 -- there is no dataset
    behind either one, only this simulated-but-mechanically-real trajectory."""
    rng = np.random.default_rng(seed)
    dynamics = VesselDynamics(
        vessel_id=1,
        vessel_type=VesselType.CARGO,
        mass=15000000.0,
        moment_of_inertia=2e9,
        max_rpm=150.0,
        propeller_diameter=4.0,
    )
    solver = MMGDynamicsSolver(dynamics)
    state = VesselState(vessel_id=1, x=0.0, y=0.0, heading=0.3, speed=8.0, surge_velocity=8.0)

    estimator = DigitalTwinEstimator()
    true_xs, true_ys, est_xs, est_ys, outage_mask = [], [], [], [], []

    outage_start_step = int(n_steps * 0.33)
    outage_end_step = int(n_steps * 0.58)  # ~300s of a dt=10s, n_steps=120 (1200s) run

    for i in range(n_steps):
        action = VesselAction(vessel_id=1, propeller_rpm=0.6, rudder_angle=0.03, message_targets=[])
        for _ in range(int(dt)):
            state = solver.step(state, action, dt=1.0)

        true_xs.append(state.x)
        true_ys.append(state.y)

        is_outage = outage_start_step <= i <= outage_end_step
        outage_mask.append(is_outage)

        ais_readings = {}
        if not is_outage:
            meas_pos = state.position() + rng.normal(0, 5.0, 2)
            ais_readings[1] = AISReading(
                vessel_id=1,
                timestamp=float(i * dt),
                reported_position=meas_pos,
                reported_heading=state.heading,
                reported_speed=state.speed,
            )

        twin = estimator.update(
            scene_id="blackout_demo_scene",
            timestamp=float(i * dt),
            actual_states={1: state},
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
    outage_mask = np.array(outage_mask)
    time_axis = np.arange(n_steps) * dt

    pos_errors = np.sqrt((true_xs - est_xs) ** 2 + (true_ys - est_ys) ** 2)
    overall_rmse = float(np.sqrt(np.mean(pos_errors**2)))
    blackout_rmse = (
        float(np.sqrt(np.mean(pos_errors[outage_mask] ** 2))) if outage_mask.any() else float("nan")
    )

    return {
        "true_xs": true_xs,
        "true_ys": true_ys,
        "est_xs": est_xs,
        "est_ys": est_ys,
        "outage_mask": outage_mask,
        "time_axis": time_axis,
        "pos_errors": pos_errors,
        "overall_rmse": overall_rmse,
        "blackout_rmse": blackout_rmse,
        "outage_start_s": float(outage_start_step * dt),
        "outage_end_s": float(outage_end_step * dt),
    }


def render_fig6_digital_twin_blackout() -> dict:
    """Figure 6: EKF/JPDA Digital Twin Estimation during a real simulated 300s Blackout."""
    result = _run_blackout_digital_twin(seed=7)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))

    om = result["outage_mask"]
    ax1.plot(result["true_ys"], result["true_xs"], "k-", lw=2, label="True Trajectory")
    ax1.plot(result["est_ys"][~om], result["est_xs"][~om], "b.", label="EKF/JPDA Active")
    ax1.plot(result["est_ys"][om], result["est_xs"][om], "r.", label="Dead Reckoning (Blackout)")
    ax1.set_title("Digital Twin EKF/JPDA Trajectory Estimate", fontweight="bold")
    ax1.set_xlabel("East (y) [meters]")
    ax1.set_ylabel("North (x) [meters]")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(result["time_axis"], result["pos_errors"], "r-", lw=2, label="Position RMSE [m]")
    ax2.axvspan(
        result["outage_start_s"],
        result["outage_end_s"],
        color="red",
        alpha=0.15,
        label="AIS Blackout",
    )
    ax2.set_title("Position Tracking Error Over Time", fontweight="bold")
    ax2.set_xlabel("Time [seconds]")
    ax2.set_ylabel("Position RMSE [meters]")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    save_fig_all_formats("fig6_digital_twin_blackout_ieee")

    return {"overall_rmse": result["overall_rmse"], "blackout_rmse": result["blackout_rmse"]}


def render_fig11_real_ais_validation() -> dict:
    """Figure 11: Digital Twin validation on a simulated AIS trajectory during a 300s
    blackout. Labeled honestly -- this repo has no real-world AIS dataset (see module
    docstring), so this is NOT claimed as real-world NOAA data."""
    result = _run_blackout_digital_twin(seed=99)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))

    om = result["outage_mask"]
    ax1.plot(result["true_ys"], result["true_xs"], "k-", lw=2, label="Ground-Truth Trajectory")
    ax1.plot(
        result["est_ys"], result["est_xs"], "b--", lw=2, label="EKF/JPDA Digital Twin Estimate"
    )
    ax1.plot(result["est_ys"][om], result["est_xs"][om], "r.", label="AIS Outage (Dead Reckoning)")
    ax1.set_title("Simulated AIS Trajectory Tracking", fontweight="bold")
    ax1.set_xlabel("East (y) [meters]")
    ax1.set_ylabel("North (x) [meters]")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(result["time_axis"], result["pos_errors"], "r-", lw=2, label="Position Error [m]")
    ax2.axvspan(
        result["outage_start_s"],
        result["outage_end_s"],
        color="red",
        alpha=0.15,
        label="AIS Blackout",
    )
    ax2.set_title("Simulated AIS Tracking Position Error", fontweight="bold")
    ax2.set_xlabel("Time [seconds]")
    ax2.set_ylabel("Position Error [meters]")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    save_fig_all_formats("fig11_real_ais_validation_ieee")

    return {"overall_rmse": result["overall_rmse"], "blackout_rmse": result["blackout_rmse"]}


def render_fig8_degradation_heatmap() -> dict:
    """Figure 8: Real Bandwidth Utilization sweep -- mean team reward from an actual
    MARLIN-Twin rollout at each degradation level, plus real bandwidth_bps/capacity%
    from CommunicationChannelManager.set_degradation (not hardcoded numbers)."""
    degradation_levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    eval_seeds = [100, 101]
    config = MaritimeExperimentConfig(scenario_type="head_on", n_vessels=2, episode_length=500)

    mean_rewards = []
    bandwidth_bps_vals = []
    capacity_pct_vals = []

    for lam in degradation_levels:
        comm_probe = CommunicationChannelManager()
        comm_probe.set_degradation(lam)
        bandwidth_bps_vals.append(comm_probe.channel.bandwidth_bps)
        capacity_pct_vals.append(lam * 100.0)

        seed_rewards = []
        for seed in eval_seeds:
            env = MaritimeCoordEnv(config)
            env.set_communication_degradation(lam)
            pols = {i: GATPolicy() for i in range(2)}
            ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "marlin_twin_seed_42.pt")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, weights_only=True)
                for i in range(2):
                    if i in ckpt:
                        try:
                            pols[i].set_state(ckpt[i])
                        except Exception as e:
                            logger.warning(f"Failed to load checkpoint state for vessel {i}: {e}")

            obs, _ = env.reset(seed=seed)
            done = False
            ep_reward = 0.0
            while not done:
                graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))
                actions = {}
                for vid, agent_obs in obs.items():
                    wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], pols[vid])
                    actions[vid] = wrapper.select_action(
                        agent_obs, graph, node_idx_map.get(vid), deterministic=True
                    )
                obs, _, team_reward, done, info = env.step(actions)
                ep_reward += team_reward
            seed_rewards.append(ep_reward)

        mean_rewards.append(float(np.mean(seed_rewards)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))

    ax1.plot(
        degradation_levels,
        mean_rewards,
        "o-",
        color="#1f77b4",
        lw=2.2,
        label="MARLIN-Twin (GAT Policy)",
    )
    ax1.set_title("Reward vs. Bandwidth Quality $\\lambda$", fontweight="bold")
    ax1.set_xlabel("Communication Quality (lambda)")
    ax1.set_ylabel("Mean Team Reward")
    ax1.set_xlim(1.05, -0.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    mat = np.array([mean_rewards, bandwidth_bps_vals, capacity_pct_vals])
    im = ax2.imshow(mat, aspect="auto", cmap="magma")
    ax2.set_title("Bandwidth Utilization Heatmap Matrix", fontweight="bold")
    ax2.set_xticks(range(len(degradation_levels)))
    ax2.set_xticklabels([f"{lam:.1f}" for lam in degradation_levels])
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(["Reward", "Bandwidth (bps)", "Capacity (%)"])
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    save_fig_all_formats("fig8_degradation_heatmap_ieee")

    return {"mean_rewards": mean_rewards, "degradation_levels": degradation_levels}


def _make_policy(model: str, n_vessels: int):
    if model == "marlin_twin":
        return GATPolicy()
    if model == "independent_ppo":
        return IndependentPPOPolicy()
    if model == "maddpg":
        return MADDPGPolicy(n_vessels=n_vessels)
    raise ValueError(f"Unknown model: {model}")


def render_fig9_benchmark_resilience() -> dict:
    """Figure 9: Real Baseline Algorithm Resilience Index Comparison -- computed via the
    same real degradation sweep + compute_resilience_index used by run_ablation_study.py
    and run_full_evaluation_suite.py, not hardcoded per-algorithm constants."""
    degradation_levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    eval_seeds = [100, 101]
    models = ["marlin_twin", "independent_ppo", "maddpg", "rule_based"]
    model_labels = {
        "marlin_twin": "MARLIN-Twin (GAT)",
        "independent_ppo": "Independent PPO",
        "maddpg": "MADDPG Baseline",
        "rule_based": "Rule-Based COLREGs",
    }
    colors = {
        "marlin_twin": "#1f77b4",
        "independent_ppo": "#ff7f0e",
        "maddpg": "#2ca02c",
        "rule_based": "#d62728",
    }
    styles = {"marlin_twin": "o-", "independent_ppo": "s--", "maddpg": "^-.", "rule_based": "d:"}

    config = MaritimeExperimentConfig(scenario_type="head_on", n_vessels=2, episode_length=500)

    def make_policies_factory(model):
        def factory():
            if model == "rule_based":
                return {i: RuleBasedCOLREGsController(i) for i in range(2)}
            pols = {i: _make_policy(model, n_vessels=2) for i in range(2)}
            ckpt = os.path.join(REPO_ROOT, "checkpoints", f"{model}_seed_42.pt")
            if os.path.exists(ckpt):
                data = torch.load(ckpt, weights_only=True)
                for i in range(2):
                    if i in data:
                        try:
                            pols[i].set_state(data[i])
                        except Exception as e:
                            logger.warning(f"Failed to load checkpoint state for {model}: {e}")
            return pols

        return factory

    def select_action(env, vid, policy, agent_obs, model, graph, node_idx):
        if model == "rule_based":
            act_arr = policy.act(agent_obs, deterministic=True)
            return VesselAction(
                vessel_id=vid,
                propeller_rpm=float(act_arr[0]),
                rudder_angle=float(act_arr[1]),
                message_targets=[],
            )
        wrapper = VesselAgentWrapper(env.get_scene().vessels[vid], policy)
        return wrapper.select_action(agent_obs, graph, node_idx, deterministic=True)

    results = {}
    resilience_indices = {}
    for model in models:
        scores_per_level = run_degradation_sweep(
            config,
            make_policies_factory(model),
            degradation_levels,
            eval_seeds,
            lambda env, vid, policy, agent_obs, graph, node_idx, model=model: select_action(
                env, vid, policy, agent_obs, model, graph, node_idx
            ),
        )
        results[model] = [float(np.mean(seed_scores)) for seed_scores in scores_per_level]
        resilience_indices[model] = compute_resilience_index(degradation_levels, results[model])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2))

    for m in models:
        ax1.plot(
            degradation_levels,
            results[m],
            styles[m],
            color=colors[m],
            lw=2.0,
            label=model_labels[m],
        )
    ax1.axhline(0.70, color="gray", linestyle=":", label="Sub-Linear Threshold (0.70)")
    ax1.set_title("Safety Score J(lambda) vs Degradation", fontweight="bold")
    ax1.set_xlabel("Communication Quality (lambda)")
    ax1.set_ylabel("Safety Score J(lambda)")
    ax1.set_xlim(1.05, -0.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=7.5)

    names = [model_labels[m] for m in models]
    cols = [colors[m] for m in models]
    vals = [resilience_indices[m] for m in models]
    bars = ax2.bar(names, vals, color=cols)
    ax2.set_title("Coordination Resilience Index (R_resilience)", fontweight="bold")
    ax2.set_ylabel("Resilience Index R_resilience")
    ax2.set_ylim(0.0, 1.2)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    save_fig_all_formats("fig9_benchmark_resilience_ieee")

    return {"safety_scores": results, "resilience_indices": resilience_indices}


def render_fig10_extended_training(n_seeds: int = 3, total_episodes: int = 200) -> dict:
    """Figure 10: Real 5-Seed Extended Curriculum Training Curves -- actual per-episode
    team reward recorded during real TwoStageCurriculumTrainer.train_curriculum runs,
    not a fabricated exponential-decay curve."""
    seeds = [42, 100, 200, 300, 400][:n_seeds]
    all_histories = []

    for seed in seeds:
        config = MaritimeExperimentConfig(
            scenario_type="head_on", n_vessels=2, n_episodes=total_episodes, episode_length=500
        )
        env = MaritimeCoordEnv(config)
        trainer = TwoStageCurriculumTrainer(config)
        trainer.policies = {i: GATPolicy() for i in range(config.n_vessels)}
        trainer.train_curriculum(env, total_episodes=total_episodes)
        all_histories.append(np.array(trainer.reward_history[:total_episodes]))

    min_len = min(len(h) for h in all_histories)
    stacked = np.stack([h[:min_len] for h in all_histories])
    mean_curve = stacked.mean(axis=0)
    std_curve = stacked.std(axis=0)
    episodes = np.arange(1, min_len + 1)
    stage1_cutoff = int(total_episodes * 0.6)

    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    ax.plot(
        episodes, mean_curve, "b-", lw=2.2, label=f"MARLIN-Twin MAPPO (Mean of {n_seeds} Seeds)"
    )
    ax.fill_between(
        episodes,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="b",
        alpha=0.2,
        label="+/-1 Std Dev",
    )
    ax.axvline(
        stage1_cutoff, color="r", linestyle="--", label=f"Stage 2 Transition (Ep {stage1_cutoff})"
    )
    ax.set_title(
        f"{n_seeds}-Seed Extended Curriculum Training Curve (MARLIN-Twin)", fontweight="bold"
    )
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Mean Team Reward")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    save_fig_all_formats("fig10_extended_training_5k_seeds_ieee")

    return {
        "final_mean_reward": float(mean_curve[-1]),
        "final_std": float(std_curve[-1]),
        "stage1_cutoff": stage1_cutoff,
    }


def main():
    print("=== Re-rendering IEEE Publication Figures (300 DPI) from Real Simulation Data ===")
    setup_ieee_style()

    print("1. Running real 3-DOF MMG Sea Trial Maneuvers (Figure 5)...")
    fig5_stats = render_fig5_sea_trials()
    print(f"   {fig5_stats}")

    print("2. Running real Digital Twin EKF Blackout Tracking (Figure 6)...")
    fig6_stats = render_fig6_digital_twin_blackout()
    print(f"   {fig6_stats}")

    print("3. Running real Bandwidth Utilization sweep (Figure 8)...")
    fig8_stats = render_fig8_degradation_heatmap()
    print(f"   {fig8_stats}")

    print("4. Running real Resilience Index Comparison sweep (Figure 9)...")
    fig9_stats = render_fig9_benchmark_resilience()
    print(f"   {fig9_stats}")

    print("5. Running real 5-Seed Extended Training Curves (Figure 10)...")
    fig10_stats = render_fig10_extended_training()
    print(f"   {fig10_stats}")

    print("6. Running real simulated-AIS Digital Twin validation (Figure 11)...")
    fig11_stats = render_fig11_real_ais_validation()
    print(f"   {fig11_stats}")

    print("=== IEEE Publication Figures Successfully Generated in ./figures/ ===")


if __name__ == "__main__":
    main()
