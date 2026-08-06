"""Multi-axis experimental sweep: scenario x fleet-size x weather x
communication-degradation-schedule x algorithm, composing existing building
blocks (`BaselineFactory`, `_evaluate_policies`) into one systematic
experiment matrix that persists structured, reusable `MaritimeExperimentResult`
output — instead of the one-off matplotlib scripts with inline
hand-recomputed formulas this repo otherwise accumulates (`run_ablation_study.py`,
`run_full_evaluation_suite.py`, `phase6_validation.py`).
"""

import os

from loguru import logger

from marlin_twin.data_classes import (
    CommsScheduleEvent,
    CoordinationResilienceMetrics,
    EnvironmentCondition,
    MaritimeExperimentConfig,
    MaritimeExperimentResult,
)
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.baselines.factory import BaselineFactory
from marlin_twin.training.mappo import _evaluate_policies

try:
    import torch
except ImportError:
    torch = None


def _try_load_checkpoint(policies: dict, algorithm: str, seed: int, checkpoint_dir: str) -> None:
    """Best-effort checkpoint load, mirroring the graceful-fallback pattern
    already used by `scripts/run_full_evaluation_suite.py` — most
    combinations run against brand-new scenario/weather/schedule
    configurations no checkpoint has ever been trained on, so a missing or
    shape-mismatched checkpoint just means "evaluate the untrained policy,"
    not a hard failure."""
    if torch is None:
        return
    ckpt_path = os.path.join(checkpoint_dir, f"{algorithm}_seed_{seed}.pt")
    if not os.path.exists(ckpt_path):
        return
    try:
        data = torch.load(ckpt_path, weights_only=True)
        for vid, pol in policies.items():
            if vid in data:
                pol.set_state(data[vid])
    except Exception as e:
        logger.warning(f"Failed to load checkpoint {ckpt_path} for {algorithm}: {e}")


def run_experiment_matrix(
    scenario_types: list[str],
    n_vessels_list: list[int],
    environment_conditions: list[EnvironmentCondition],
    comms_schedules: list[list[CommsScheduleEvent]],
    algorithms: list[str],
    n_episodes: int = 20,
    seeds: list[int] | None = None,
    maddpg_fixed_n_vessels: int | None = None,
    checkpoint_dir: str | None = None,
) -> MaritimeExperimentResult:
    """Run every combination of the given axes and aggregate into one
    `MaritimeExperimentResult`.

    `baseline_comparison[run_id]` holds each combination's aggregate summary
    metrics dict (`average_reward`/`safety_score`/`efficiency_score`/
    `colregs_violation_rate`/`communication_utilization`); `episodes` holds
    every combination's per-episode `VoyageEpisode` records (`episode_id`
    prefixed with `run_id` for traceability) across every seed in `seeds`.

    `algorithm == "maddpg"` is skipped for any `n_vessels !=
    maddpg_fixed_n_vessels` (when the latter is given) — `MADDPGPolicy`'s
    `CentralizedCritic` has a fixed input width baked in at construction, so
    it can't be meaningfully evaluated at a fleet size other than the one
    it would be trained/checkpointed for.

    If `checkpoint_dir` is given, each combination best-effort loads
    `{checkpoint_dir}/{algorithm}_seed_{seed}.pt` (the same naming
    convention `MAPPOTrainer.save_checkpoint`/`MADDPGTrainer.save_checkpoint`
    use) before evaluating, falling back to an untrained policy if none
    exists — most new scenario/weather/schedule combinations won't have one
    yet; this is a smoke-test/infrastructure driver, not a replacement for
    actually retraining on the new scenario mix.
    """
    if seeds is None:
        seeds = [42]
    if not comms_schedules:
        comms_schedules = [[]]

    all_episodes = []
    baseline_comparison: dict[str, dict[str, float]] = {}
    skipped = []
    representative_config = None

    for scenario_type in scenario_types:
        for n_vessels in n_vessels_list:
            for condition in environment_conditions:
                for schedule in comms_schedules:
                    for algorithm in algorithms:
                        if (
                            algorithm == "maddpg"
                            and maddpg_fixed_n_vessels is not None
                            and n_vessels != maddpg_fixed_n_vessels
                        ):
                            skipped.append((scenario_type, n_vessels, condition.name, algorithm))
                            continue

                        config = MaritimeExperimentConfig(
                            scenario_type=scenario_type,
                            n_vessels=n_vessels,
                            environment_condition=condition,
                            comms_schedule=schedule,
                        )
                        if representative_config is None:
                            representative_config = config

                        run_id = f"{scenario_type}|n{n_vessels}|{condition.name}|{algorithm}"

                        run_episodes = []
                        run_summaries = []
                        for seed in seeds:
                            env = MaritimeCoordEnv(config)
                            policies = BaselineFactory(config).create(algorithm)
                            if checkpoint_dir:
                                _try_load_checkpoint(policies, algorithm, seed, checkpoint_dir)

                            summary, episodes = _evaluate_policies(
                                env,
                                policies,
                                n_episodes,
                                communication_degradation=1.0,
                                return_episodes=True,
                                seed_offset=seed,
                            )
                            for ep in episodes:
                                ep.episode_id = f"{run_id}|seed{seed}|{ep.episode_id}"
                            run_episodes.extend(episodes)
                            run_summaries.append(summary)

                        all_episodes.extend(run_episodes)
                        baseline_comparison[run_id] = {
                            key: sum(s[key] for s in run_summaries) / len(run_summaries)
                            for key in run_summaries[0]
                        }

    if skipped:
        logger.info(
            f"Skipped {len(skipped)} maddpg combination(s) with n_vessels "
            f"!= maddpg_fixed_n_vessels={maddpg_fixed_n_vessels}: {skipped}"
        )

    return MaritimeExperimentResult(
        config=representative_config,
        episodes=all_episodes,
        baseline_comparison=baseline_comparison,
        resilience_metrics=CoordinationResilienceMetrics(),
    )
