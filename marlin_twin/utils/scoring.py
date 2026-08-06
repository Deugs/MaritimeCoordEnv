"""Canonical episode-scoring formulas shared by every trainer/evaluator.

Replaces three previously inconsistent `safety_score` formulas and two
hardcoded stub constants (`efficiency_score=0.85`, `colregs_violation_rate=
0.05`) that were scattered across `training/mappo.py`, `training/maddpg.py`,
`training/eval.py`, and `scripts/`, none of which computed anything from the
actual episode.
"""

import numpy as np


def compute_safety_score(min_cpa_values: list[float]) -> float:
    """Mean closest-point-of-approach distance across episodes, normalized
    into [0, 1] against a 1000m reference. The canonical safety formula —
    every call site should feed it real per-episode `info["min_cpa"]`
    values rather than recomputing its own variant."""
    avg_cpa = float(np.mean(min_cpa_values)) if min_cpa_values else 5000.0
    return float(np.clip(avg_cpa / 1000.0, 0.0, 1.0))


def compute_efficiency_score(
    progress_ratios: list[float], mean_fuel_per_step: list[float], fuel_weight: float = 0.2
) -> float:
    """Goal-progress-per-effort: mean fraction of remaining route distance
    actually closed, penalized by fuel-proxy consumption intensity.

    `mean_fuel_per_step` is each vessel's average `|propeller_rpm|**3` per
    step over the episode — already bounded in [0, 1] since `propeller_rpm`
    is normalized to [-1, 1] (clipped to [0.2, 1.0] in practice by
    `VesselAgentWrapper.build_action`), so no further normalization is
    needed. The caller divides accumulated fuel-proxy by episode length
    before calling."""
    if not progress_ratios:
        return 0.0
    progress = float(np.mean(progress_ratios))
    fuel_penalty = float(np.mean(mean_fuel_per_step)) if mean_fuel_per_step else 0.0
    return float(np.clip(progress * (1.0 - fuel_weight * fuel_penalty), 0.0, 1.0))


def compute_colregs_violation_rate(violation_count: int, step_vessel_pairs: int) -> float:
    """Fraction of (step, vessel) pairs where `COLREGsEngine.evaluate_compliance`
    scored below 0.5 — i.e. an actual observed rule violation, not a
    constant guess. `step_vessel_pairs` is the total number of per-step
    per-vessel opportunities to violate, summed across every episode
    (`sum(n_steps * n_vessels)`)."""
    denom = max(step_vessel_pairs, 1)
    return float(np.clip(violation_count / denom, 0.0, 1.0))
