#!/usr/bin/env python3
"""
CLI Script to run the multi-axis experimental scenario matrix (scenario type
x fleet size x weather x communication-degradation schedule x algorithm) and
persist a structured `MaritimeExperimentResult` for later analysis.

Usage:
    python scripts/run_experiment_matrix.py \
        --scenarios head_on,crossing_give_way,congested_port_approach \
        --n-vessels 4,6 --conditions CLEAR,FOG \
        --algorithms marlin_twin,independent_ppo,rule_based \
        --episodes 20 --seeds 42,43

Most scenario/weather/schedule combinations are brand new — no checkpoint
has been trained on them yet, so `--checkpoint-dir` loading (when a
checkpoint happens to exist for a given algorithm/seed) is best-effort, and
results for untrained policies mainly validate that the pipeline runs
end-to-end, not that the policies perform well. Rerun
`scripts/run_retrain_all_baselines.py`-style training on this scenario mix
separately to get meaningful comparative numbers.
"""

import argparse
from pathlib import Path

from marlin_twin.data_classes import CommsScheduleEvent, EnvironmentCondition
from marlin_twin.training.experiment_matrix import run_experiment_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SCENARIOS = [
    "head_on",
    "crossing_give_way",
    "overtaking",
    "multi_vessel_channel_convergence",
    "congested_port_approach",
    "restricted_visibility_crossing",
    "comms_blackout_transit",
]

# A concrete example of the scheduling primitive Part 5 adds: full comms for
# the first third of a run, a scripted blackout/jamming window mid-transit,
# then partial recovery — only applied when --with-comms-schedule-demo is set.
DEMO_COMMS_SCHEDULE = [
    CommsScheduleEvent(t_start=50, t_end=90, degradation_level=0.1),
    CommsScheduleEvent(t_start=100, t_end=150, degradation_level=0.0, jamming_zone=(0, 0, 1000)),
]


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run the MARLIN-Twin experiment matrix")
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--n-vessels", type=str, default="4,6")
    parser.add_argument("--conditions", type=str, default="CLEAR,FOG")
    parser.add_argument("--algorithms", type=str, default="marlin_twin,independent_ppo,rule_based")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument(
        "--maddpg-n-vessels",
        type=int,
        default=None,
        help="Only evaluate 'maddpg' at this exact n_vessels (skipped entirely if omitted "
        "while 'maddpg' is in --algorithms and --n-vessels has more than one value).",
    )
    parser.add_argument(
        "--with-comms-schedule-demo",
        action="store_true",
        help="Add a demo degradation-dip + jamming-window schedule as a second comms axis value.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=str(REPO_ROOT / "checkpoints"))
    parser.add_argument(
        "--output", type=str, default=str(REPO_ROOT / "results" / "experiment_matrix_result.pkl")
    )
    args = parser.parse_args()

    scenarios = _parse_list(args.scenarios)
    n_vessels_list = [int(v) for v in _parse_list(args.n_vessels)]
    conditions = [EnvironmentCondition[c] for c in _parse_list(args.conditions)]
    algorithms = _parse_list(args.algorithms)
    seeds = [int(v) for v in _parse_list(args.seeds)]
    comms_schedules = [[], DEMO_COMMS_SCHEDULE] if args.with_comms_schedule_demo else [[]]

    print("=== MARLIN-Twin Experiment Matrix ===")
    print(f"Scenarios:   {scenarios}")
    print(f"N-vessels:   {n_vessels_list}")
    print(f"Conditions:  {[c.name for c in conditions]}")
    print(f"Algorithms:  {algorithms}")
    print(f"Episodes/run: {args.episodes}, Seeds: {seeds}")

    result = run_experiment_matrix(
        scenario_types=scenarios,
        n_vessels_list=n_vessels_list,
        environment_conditions=conditions,
        comms_schedules=comms_schedules,
        algorithms=algorithms,
        n_episodes=args.episodes,
        seeds=seeds,
        maddpg_fixed_n_vessels=args.maddpg_n_vessels,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(
        f"\nCompleted {len(result.baseline_comparison)} combinations, "
        f"{len(result.episodes)} total episodes."
    )
    for run_id, summary in result.baseline_comparison.items():
        print(f"  {run_id}: {summary}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path))
    print(f"\nSaved structured result -> {output_path}")


if __name__ == "__main__":
    main()
