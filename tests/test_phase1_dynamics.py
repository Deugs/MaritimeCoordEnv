# ============================================================================
# FILE: tests/test_phase1_dynamics.py
# ============================================================================

import pytest
import numpy as np
from marlin_twin.data_classes import VesselState, VesselDynamics, VesselType
from marlin_twin.envs.vessel_dynamics import MMGDynamicsSolver
from marlin_twin.envs.encounters import EncounterManager
from marlin_twin.envs.scenarios import ScenarioGenerator


def test_turning_circle_sea_trial():
    dynamics = VesselDynamics(
        vessel_id=0,
        vessel_type=VesselType.CARGO,
        mass=10000000.0,
        moment_of_inertia=1e9,
        max_rpm=150.0,
        propeller_diameter=4.0,
    )
    solver = MMGDynamicsSolver(dynamics)
    result = solver.run_turning_circle_test(rudder_angle_deg=35.0, duration=300.0)

    assert result["tactical_diameter"] > 0.0
    assert len(result["trajectory"]) > 100
    assert result["advance"] > 0.0


def test_zigzag_sea_trial():
    dynamics = VesselDynamics(
        vessel_id=0,
        vessel_type=VesselType.CARGO,
        mass=10000000.0,
        moment_of_inertia=1e9,
        max_rpm=150.0,
        propeller_diameter=4.0,
    )
    solver = MMGDynamicsSolver(dynamics)
    result = solver.run_zigzag_test(angle_deg=10.0, duration=200.0)

    assert "first_overshoot_angle" in result
    assert "second_overshoot_angle" in result
    assert isinstance(result["first_overshoot_angle"], float)


def test_cpa_tcpa_computation():
    v1 = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=10.0)  # Heading North
    v2 = VesselState(vessel_id=1, x=0.0, y=2000.0, heading=np.pi, speed=10.0)  # Heading South

    cpa, tcpa, dcpa = EncounterManager.compute_cpa(v1, v2)
    assert cpa == pytest.approx(0.0, abs=1e-3)
    assert tcpa == pytest.approx(100.0, abs=1.0)  # 2000m / (10 + 10) m/s = 100s


def test_preset_scenarios_creation():
    scenarios = ["head_on", "crossing_give_way", "overtaking"]
    for sc in scenarios:
        agents = ScenarioGenerator.create_scenario(scenario_type=sc, n_vessels=2)
        assert len(agents) == 2
        assert agents[0].current_state.speed > 0.0
