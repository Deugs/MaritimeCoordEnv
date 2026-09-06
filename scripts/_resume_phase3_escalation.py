#!/usr/bin/env python3
"""One-off resume driver: finishes B3_restricted_visibility_n6's escalation
(interrupted by a container restart, ~60% done) and separately runs the full
escalation for B1_port_approach_n6 (the scenario the corrected
pick_most_discriminative filter actually selects) -- see the Phase 3 fix
commit for why both are being kept rather than discarding B3's already-real
training. Not part of the normal run_scenario_battery.py entrypoint since
that script's main() only escalates one winner; this exists purely to
finish this session's specific two-scenario situation.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from run_scenario_battery import (  # noqa: E402
    CANDIDATES,
    build_escalation_jobs,
    evaluate_escalation,
)
from _experiment_common import run_pool, save_results_json  # noqa: E402
from run_scenario_battery import _job_fn  # noqa: E402


def run_one(scenario_name: str):
    scenario_type, n_vessels = CANDIDATES[scenario_name]
    print(f"=== Escalating {scenario_name} ({scenario_type}, N={n_vessels}) ===")
    jobs = build_escalation_jobs(scenario_name, scenario_type, n_vessels)
    print(f"  {len(jobs)} jobs remaining (Pool(4))")
    if jobs:
        for result in run_pool(_job_fn, jobs, n_workers=4):
            print(f"  completed: {result}")

    print(f"=== Evaluating {scenario_name} ===")
    eval_results = evaluate_escalation(scenario_name, scenario_type, n_vessels)
    out_path = save_results_json(
        f"scenario_battery_escalation_{scenario_name}",
        {
            "winner": scenario_name,
            "scenario_type": scenario_type,
            "n_vessels": n_vessels,
            "results": eval_results,
        },
    )
    print(f"Saved to {out_path}")
    for variant, r in eval_results.items():
        print(
            f"  {variant}: J(1.0)={r['j1_mean']:.4f}+/-{r['j1_std']:.4f}  "
            f"R={r['resilience_mean']:.4f}+/-{r['resilience_std']:.4f}"
        )


if __name__ == "__main__":
    run_one("B3_restricted_visibility_n6")
    run_one("B1_port_approach_n6")
