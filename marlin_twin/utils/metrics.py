"""Coordination resilience index computation via trapezoidal integration."""

import numpy as np


def compute_resilience_index(degradation_levels: list[float], safety_scores: list[float]) -> float:
    """
    Computes the normalized Coordination Resilience Index (R_resilience):
    R_resilience = integral_0^1 (J(lambda) / J(1.0)) d_lambda via trapezoidal integration.
    """
    if not degradation_levels or len(degradation_levels) < 2:
        return 0.0

    levels = np.array(degradation_levels, dtype=np.float64)
    scores = np.array(safety_scores, dtype=np.float64)

    # Sort levels ascending
    sort_idx = np.argsort(levels)
    levels = levels[sort_idx]
    scores = scores[sort_idx]

    # Normalize by baseline at full communication (lambda = 1.0)
    baseline = max(scores[-1], 1e-6)
    norm_scores = scores / baseline

    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = trapezoid(norm_scores, levels)
    span = levels[-1] - levels[0]
    return float(area / max(span, 1e-6))
