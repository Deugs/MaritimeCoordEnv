# ============================================================================
# FILE: tests/test_phase1_dynamics.py
# ============================================================================

import pytest
import numpy as np
from marlin_twin.data_classes import VesselState, VesselDynamics, VesselType
from marlin_twin.envs.vessel_dynamics import MMGDynamicsSolver
from marlin_twin.envs.encounters import EncounterManager
from marlin_twin.envs.scenarios import ScenarioGenerator
from marlin_twin.envs.vessel_profiles import VESSEL_PROFILES


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
    # A CARGO-scale vessel's yaw response to a 10 deg rudder is slow
    # (empirically confirmed to need ~19700s to complete both overshoots
    # at this duration/dt after fixing the double-RPM-scaling thrust bug,
    # see VesselDynamics.thrust_coefficient); USV converges within ~8000s
    # with its own correctly-calibrated thrust_coefficient, keeping this
    # test faster while still exercising a real completed maneuver.
    usv_profile = VESSEL_PROFILES[VesselType.USV]
    dynamics = VesselDynamics(
        vessel_id=0,
        vessel_type=VesselType.USV,
        mass=usv_profile["mass"],
        moment_of_inertia=usv_profile["moment_of_inertia"],
        X_u_dot=usv_profile["X_u_dot"],
        Y_v_dot=usv_profile["Y_v_dot"],
        N_r_dot=usv_profile["N_r_dot"],
        X_u=usv_profile["X_u"],
        Y_v=usv_profile["Y_v"],
        N_r=usv_profile["N_r"],
        max_rpm=usv_profile["max_rpm"],
        rudder_area=usv_profile["rudder_area"],
        propeller_diameter=usv_profile["propeller_diameter"],
        thrust_coefficient=usv_profile["thrust_coefficient"],
    )
    solver = MMGDynamicsSolver(dynamics)
    initial_state = VesselState(
        vessel_id=0,
        x=0.0,
        y=0.0,
        heading=0.0,
        speed=usv_profile["max_speed"],
        surge_velocity=usv_profile["max_speed"],
    )
    result = solver.run_zigzag_test(initial_state=initial_state, angle_deg=10.0, duration=8000.0)

    assert result["first_overshoot_converged"] is True
    assert result["second_overshoot_converged"] is True
    assert isinstance(result["first_overshoot_angle"], float)
    assert isinstance(result["second_overshoot_angle"], float)
    assert result["first_overshoot_angle"] > 0.0
    assert result["second_overshoot_angle"] > 0.0


def test_zigzag_sea_trial_reports_non_convergence_honestly():
    """A maneuver that doesn't complete within `duration` must report
    None + converged=False, not a fabricated placeholder overshoot angle
    (regression guard for a real bug: this used to silently return 2.5/3.1
    deg for any non-convergent run)."""
    dynamics = VesselDynamics(
        vessel_id=0,
        vessel_type=VesselType.CARGO,
        mass=15000000.0,
        moment_of_inertia=2e9,
        max_rpm=150.0,
        propeller_diameter=4.0,
    )
    solver = MMGDynamicsSolver(dynamics)
    result = solver.run_zigzag_test(angle_deg=10.0, duration=50.0)

    assert result["first_overshoot_converged"] is False
    assert result["second_overshoot_converged"] is False
    assert result["first_overshoot_angle"] is None
    assert result["second_overshoot_angle"] is None


def test_cpa_tcpa_computation():
    v1 = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=10.0)  # Heading North
    v2 = VesselState(vessel_id=1, x=0.0, y=2000.0, heading=np.pi, speed=10.0)  # Heading South

    tcpa, dcpa, cpa_distance = EncounterManager.compute_cpa(v1, v2)
    assert tcpa == pytest.approx(100.0, abs=1.0)  # 2000m / (10 + 10) m/s = 100s
    assert dcpa == pytest.approx(0.0, abs=1e-3)
    assert cpa_distance == pytest.approx(0.0, abs=1e-3)


def test_preset_scenarios_creation():
    scenarios = ["head_on", "crossing_give_way", "overtaking"]
    for sc in scenarios:
        agents = ScenarioGenerator.create_scenario(scenario_type=sc, n_vessels=2)
        assert len(agents) == 2
        assert agents[0].current_state.speed > 0.0
