# ============================================================================
# FILE: marlin_twin/utils/metrics.py
# ============================================================================

import numpy as np
from marlin_twin.data_classes import CoordinationResilienceMetrics

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

    area = np.trapz(norm_scores, levels)
    span = levels[-1] - levels[0]
    return float(area / max(span, 1e-6))
