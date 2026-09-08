#!/usr/bin/env python3
"""
Compiles every Phase 1-3 results/*.json file into human-readable markdown
tables -- the single source every number in the paper should be read from,
rather than hand-copied from log output. Emits
results/compiled_results_report.md.

Usage:
    python scripts/compile_results_tables.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

MODEL_LABELS = {
    "marlin_twin": "MARLIN-Twin (GAT)",
    "ablation_mean_pooling": "Ablation: Mean-Pooling",
    "ablation_flat_mlp": "Ablation: Flat MLP",
    "ablation_no_digital_twin": "Ablation: No Digital Twin",
    "independent_ppo": "Independent PPO",
    "maddpg": "MADDPG",
    "sac": "MASAC (Multi-Agent SAC)",
    "rule_based": "Rule-Based COLREGs",
}


def load(name: str) -> dict:
    with open(RESULTS_DIR / f"{name}.json") as f:
        return json.load(f)


def fmt_row(label: str, r: dict) -> str:
    return (
        f"| {label} | {r['j1_mean']:.4f} +/- {r['j1_std']:.4f} "
        f"| {r['resilience_mean']:.4f} +/- {r['resilience_std']:.4f} |"
    )


def section_phase1() -> str:
    data = load("training_budget_study")
    lines = [
        "## Phase 1 -- Realistic Training Budget Study",
        "",
        f"Seeds: {data['seeds']}. Primary eval seeds: {len(data['eval_seeds_primary'])} "
        f"({data['eval_seeds_primary'][0]}-{data['eval_seeds_primary'][-1]}).",
        "",
        "### e150 (current published budget, WITH the Phase 0 curriculum seed fix)",
        "",
        "| Variant | J(1.0) | Resilience Index |",
        "|---|---|---|",
    ]
    for variant, r in data["e150"].items():
        lines.append(fmt_row(MODEL_LABELS.get(variant, variant), r))

    lines += [
        "",
        "### e500 (primary realistic budget, 3.3x increase)",
        "",
        "| Variant | J(1.0) | Resilience Index |",
        "|---|---|---|",
    ]
    for variant, r in data["e500"].items():
        lines.append(fmt_row(MODEL_LABELS.get(variant, variant), r))

    lines += ["", "### Significance (Welch's t-test + Cohen's d, e500)", ""]
    for comparison, sig in data["significance"].items():
        lines.append(
            f"- **{comparison}**: mean_a={sig['mean_a']:.4f}, mean_b={sig['mean_b']:.4f}, "
            f"p={sig['p_value']:.5f}, Cohen's d={sig['cohens_d']:.3f}, "
            f"significant at 0.05: {sig['significant_at_0.05']}"
        )

    lines += [
        "",
        "### mv4 (4-vessel multi_vessel_channel_convergence, e500 budget)",
        "",
        "| Variant | J(1.0) | Resilience Index |",
        "|---|---|---|",
    ]
    for variant, r in data.get("mv4_e500", {}).items():
        lines.append(fmt_row(MODEL_LABELS.get(variant, variant), r))

    return "\n".join(lines)


def section_phase2() -> str:
    data = load("reward_reweighting_study")
    lines = [
        "",
        "## Phase 2 -- Reward Reweighting Study",
        "",
        f"marlin_twin only, seeds {data['seeds']}, {data['n_episodes']} episodes/seed.",
        "",
        "| Arm | Weights (safety, colregs, efficiency, true_sep) | J(1.0) | Resilience | "
        "Route Completion | Sub-100m Rate | Mean Speed | Mean Final Dist |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, r in data["arms"].items():
        w = str(r["weights"]) if r["weights"] else "n/a"
        lines.append(
            f"| {arm} | {w} | {r['j1_mean']:.4f} +/- {r['j1_std']:.4f} "
            f"| {r['resilience_mean']:.4f} +/- {r['resilience_std']:.4f} "
            f"| {r['route_completed_fraction_mean']:.2f} | {r['sub_100m_rate']:.2f} "
            f"| {r['mean_speed_mean']:.2f} | {r['mean_final_distance_mean']:.1f} |"
        )
    return "\n".join(lines)


def section_phase3() -> str:
    screen = load("scenario_battery_screen")
    lines = [
        "",
        "## Phase 3 -- Harder Scenario Battery",
        "",
        "### Step 3.0: Screen (rule_based needs no training; random-policy floor for context)",
        "",
        "| Candidate | Scenario Type | N | rule_based J(1.0) | random J(1.0) | "
        "rule_based Resilience |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in screen["candidates"].items():
        lines.append(
            f"| {name} | {r['scenario_type']} | {r['n_vessels']} "
            f"| {r['rule_based_j1']:.4f} | {r['random_policy_j1']:.4f} "
            f"| {r['rule_based_resilience']:.4f} |"
        )

    for escalation_name, label in [
        ("scenario_battery_escalation_B3_restricted_visibility_n6", "B3_restricted_visibility_n6"),
        ("scenario_battery_escalation_B1_port_approach_n6", "B1_port_approach_n6"),
    ]:
        esc = load(escalation_name)
        lines += [
            "",
            f"### Step 3.1 Escalation: {label} ({esc['scenario_type']}, N={esc['n_vessels']})",
            "",
            "| Variant | J(1.0) | Resilience Index |",
            "|---|---|---|",
        ]
        for variant, r in esc["results"].items():
            lines.append(fmt_row(MODEL_LABELS.get(variant, variant), r))

    return "\n".join(lines)


def main():
    report = "\n".join(
        [
            "# MARLIN-Twin: Compiled Experimental Results (Phases 0-3)",
            "",
            "Generated by `scripts/compile_results_tables.py` from `results/*.json` "
            "-- every number below is read directly from a persisted experiment result, "
            "not hand-copied.",
            "",
            section_phase1(),
            section_phase2(),
            section_phase3(),
        ]
    )
    out_path = RESULTS_DIR / "compiled_results_report.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Wrote {out_path}")
    print(report)


if __name__ == "__main__":
    main()
