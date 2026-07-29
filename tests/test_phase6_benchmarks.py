# ============================================================================
# FILE: tests/test_phase6_benchmarks.py
# ============================================================================

import pytest
import numpy as np
from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.baselines.factory import BaselineFactory
from marlin_twin.utils.metrics import compute_resilience_index

def test_baseline_factory_creation():
    config = MaritimeExperimentConfig(n_vessels=3)
    factory = BaselineFactory(config)

    algorithms = ["marlin_twin", "flat_mlp", "rule_based", "independent_ppo", "maddpg"]
    for alg in algorithms:
        policies = factory.create(alg)
        assert len(policies) == 3
        assert 0 in policies

def test_rule_based_controller_action():
    config = MaritimeExperimentConfig(n_vessels=2)
    factory = BaselineFactory(config)
    rule_pol = factory.create("rule_based")[0]

    from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
    env = MaritimeCoordEnv(config)
    obs, _ = env.reset(seed=42)

    act = rule_pol.act(obs[0], deterministic=True)
    assert act.shape == (2,)
    assert act[0] > 0.0  # RPM positive

def test_resilience_index_calculation():
    degradation_levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    safety_scores = [1.0, 0.95, 0.90, 0.82, 0.75, 0.60]

    r_idx = compute_resilience_index(degradation_levels, safety_scores)
    assert isinstance(r_idx, float)
    assert 0.6 < r_idx <= 1.0
