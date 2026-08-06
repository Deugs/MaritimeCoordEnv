"""Verification tests for the extensive experimental scenario suite: real
(non-hardcoded) metrics, N-vessel-capable scenario geometry, heterogeneous
vessel hydrodynamics, weather/visibility effects, scheduled communication
degradation, the 4 new scenario types, and the multi-axis sweep driver.
"""

import numpy as np
import pytest

from marlin_twin.data_classes import (
    CommsScheduleEvent,
    EnvironmentCondition,
    MaritimeExperimentConfig,
    VesselAction,
    VesselType,
)
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.envs.scenarios import ScenarioGenerator
from marlin_twin.envs.vessel_dynamics import MMGDynamicsSolver
from marlin_twin.envs.vessel_profiles import VESSEL_PROFILES
from marlin_twin.baselines.rule_based import RuleBasedCOLREGsController
from marlin_twin.training.eval import MultiScenarioEvaluator
from marlin_twin.training.experiment_matrix import run_experiment_matrix
from marlin_twin.training.mappo import MAPPOTrainer
from marlin_twin.training.maddpg import MADDPGTrainer
from marlin_twin.utils.scoring import (
    compute_colregs_violation_rate,
    compute_efficiency_score,
    compute_safety_score,
)

# --- Part 1: real (non-hardcoded) metrics ------------------------------------


def test_compute_safety_score_matches_hand_computation():
    assert compute_safety_score([]) == pytest.approx(1.0)
    assert compute_safety_score([500.0, 1500.0]) == pytest.approx(1.0)
    assert compute_safety_score([200.0, 400.0]) == pytest.approx(0.3)


def test_compute_efficiency_score_penalizes_fuel_and_rewards_progress():
    no_progress = compute_efficiency_score([], [])
    assert no_progress == 0.0

    full_progress_no_fuel = compute_efficiency_score([1.0], [0.0])
    assert full_progress_no_fuel == pytest.approx(1.0)

    full_progress_full_fuel = compute_efficiency_score([1.0], [1.0], fuel_weight=0.5)
    assert full_progress_full_fuel == pytest.approx(0.5)


def test_compute_colregs_violation_rate_is_a_real_fraction():
    assert compute_colregs_violation_rate(0, 100) == 0.0
    assert compute_colregs_violation_rate(10, 100) == pytest.approx(0.1)
    # denom floors at 1, and the rate clips to 1.0 (can't exceed 100%)
    assert compute_colregs_violation_rate(5, 0) == pytest.approx(1.0)


def test_mappo_and_maddpg_evaluate_no_longer_return_hardcoded_stub_constants():
    config = MaritimeExperimentConfig(scenario_type="head_on", n_vessels=2, episode_length=10)
    env = MaritimeCoordEnv(config)
    policies = {0: RuleBasedCOLREGsController(0), 1: RuleBasedCOLREGsController(1)}

    mappo_results = MAPPOTrainer(config).evaluate(env, policies, n_episodes=2)
    assert mappo_results["efficiency_score"] != 0.85
    assert mappo_results["colregs_violation_rate"] != 0.05

    maddpg_results = MADDPGTrainer(config).evaluate(env, policies, n_episodes=2)
    assert maddpg_results["efficiency_score"] != 0.85
    assert maddpg_results["colregs_violation_rate"] != 0.05


def test_eval_py_uses_canonical_safety_score():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2, episode_length=10)
    env = MaritimeCoordEnv(config)
    policies = {0: RuleBasedCOLREGsController(0), 1: RuleBasedCOLREGsController(1)}
    results = MultiScenarioEvaluator.evaluate_scenario(
        env, policies, scenario_name="channel", n_episodes=2
    )
    assert results["safety_score"] == pytest.approx(
        float(np.clip(results["mean_cpa"] / 1000.0, 0.0, 1.0))
    )


# --- Part 2: N-vessel-capable scenario geometry + crossing alias -------------


@pytest.mark.parametrize("scenario_type", ["head_on", "crossing_give_way", "overtaking"])
@pytest.mark.parametrize("n_vessels", [4, 5, 6])
def test_pairwise_scenarios_scale_past_two_vessels(scenario_type, n_vessels):
    agents = ScenarioGenerator.create_scenario(scenario_type, n_vessels, seed=1)
    assert len(agents) == n_vessels
    for agent in agents.values():
        assert agent.current_state.speed > 0.0


def test_crossing_alias_matches_crossing_give_way_exactly():
    a1 = ScenarioGenerator.create_scenario("crossing", 4, seed=3)
    a2 = ScenarioGenerator.create_scenario("crossing_give_way", 4, seed=3)
    for vid in a1:
        assert a1[vid].current_state == a2[vid].current_state


def test_head_on_two_vessel_geometry_unchanged_from_before_generalization():
    agents = ScenarioGenerator.create_scenario("head_on", 2, seed=1)
    assert (agents[0].current_state.x, agents[0].current_state.y) == (0.0, -2000.0)
    assert (agents[1].current_state.x, agents[1].current_state.y) == (0.0, 2000.0)


# --- Part 3: real per-VesselType heterogeneity -------------------------------


def test_vessel_types_have_distinct_hydrodynamic_coefficients():
    cargo = VESSEL_PROFILES[VesselType.CARGO]
    tanker = VESSEL_PROFILES[VesselType.TANKER]
    usv = VESSEL_PROFILES[VesselType.USV]
    assert cargo["N_r"] != tanker["N_r"] != usv["N_r"]
    assert cargo["max_speed"] != tanker["max_speed"]


def test_tanker_is_less_yaw_responsive_than_usv():
    def yaw_rate_after(vtype, seconds=30):
        p = VESSEL_PROFILES[vtype]
        from marlin_twin.data_classes import VesselDynamics, VesselState

        dyn = VesselDynamics(
            vessel_id=0,
            vessel_type=vtype,
            mass=p["mass"],
            moment_of_inertia=p["moment_of_inertia"],
            X_u_dot=p["X_u_dot"],
            Y_v_dot=p["Y_v_dot"],
            N_r_dot=p["N_r_dot"],
            X_u=p["X_u"],
            Y_v=p["Y_v"],
            N_r=p["N_r"],
            propeller_diameter=4.0,
            max_rpm=p["max_rpm"],
            rudder_area=p["rudder_area"],
        )
        solver = MMGDynamicsSolver(dyn)
        state = VesselState(vessel_id=0, x=0, y=0, heading=0, speed=8.0, surge_velocity=8.0)
        action = VesselAction(
            vessel_id=0, propeller_rpm=0.8, rudder_angle=np.radians(20), message_targets=[]
        )
        for _ in range(seconds):
            state = solver.step(state, action, 1.0)
        return abs(state.yaw_rate)

    assert yaw_rate_after(VesselType.USV) > yaw_rate_after(VesselType.TANKER)


def test_heterogeneous_fleet_config_is_actually_wired_into_the_env():
    config = MaritimeExperimentConfig(
        scenario_type="channel",
        n_vessels=4,
        vessel_types=[VesselType.TANKER, VesselType.FISHING],
        heterogeneous=True,
    )
    env = MaritimeCoordEnv(config)
    env.reset(seed=1)
    types = [env.get_scene().vessels[i].specification.vessel_type for i in range(4)]
    assert types == [VesselType.TANKER, VesselType.FISHING, VesselType.TANKER, VesselType.FISHING]


def test_heterogeneous_false_falls_back_to_cargo_usv_alternation():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2, heterogeneous=False)
    env = MaritimeCoordEnv(config)
    env.reset(seed=1)
    assert env.get_scene().vessels[0].specification.vessel_type == VesselType.CARGO
    assert env.get_scene().vessels[1].specification.vessel_type == VesselType.USV


# --- Part 4: weather/visibility has a real mechanical effect -----------------


def test_fog_reduces_visibility_range_and_encounter_detection():
    def run(condition):
        config = MaritimeExperimentConfig(
            scenario_type="crossing_give_way",
            n_vessels=2,
            episode_length=15,
            environment_condition=condition,
        )
        env = MaritimeCoordEnv(config)
        obs, info = env.reset(seed=7)
        visibility = obs[0].visibility_range
        actions = {
            v: VesselAction(vessel_id=v, propeller_rpm=0.6, rudder_angle=0.0, message_targets=[])
            for v in obs
        }
        total_encounters = info["encounters"]
        done = False
        while not done:
            obs, rewards, team_reward, done, info = env.step(actions)
            total_encounters += info["encounters"]
        return visibility, total_encounters

    clear_visibility, clear_encounters = run(EnvironmentCondition.CLEAR)
    fog_visibility, fog_encounters = run(EnvironmentCondition.FOG)

    assert fog_visibility < clear_visibility
    assert fog_encounters <= clear_encounters


def test_fog_increases_sensor_noise():
    from marlin_twin.data_classes import VesselState
    from marlin_twin.envs.sensors import SensorSimulator

    state = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=8.0)
    np.random.seed(0)
    clear_errs = [
        abs(
            SensorSimulator.generate_ais(
                state, 0.0, drop_prob=0.0, noise_scale=1.0
            ).reported_position[0]
        )
        for _ in range(200)
    ]
    fog_errs = [
        abs(
            SensorSimulator.generate_ais(
                state, 0.0, drop_prob=0.0, noise_scale=6.25
            ).reported_position[0]
        )
        for _ in range(200)
    ]
    assert np.mean(fog_errs) > np.mean(clear_errs) * 2


# --- Part 5: scheduled/time-varying communication degradation ---------------


def test_comms_schedule_applies_and_reverts_correctly():
    schedule = [
        CommsScheduleEvent(t_start=3, t_end=6, degradation_level=0.2),
        CommsScheduleEvent(t_start=8, t_end=10, degradation_level=0.0, jamming_zone=(0, 0, 500)),
    ]
    config = MaritimeExperimentConfig(
        scenario_type="open_water", n_vessels=2, episode_length=12, comms_schedule=schedule
    )
    env = MaritimeCoordEnv(config)
    env.set_communication_degradation(1.0)
    env.reset(seed=1)
    assert env.comms_degradation_level == pytest.approx(1.0)

    actions = {
        v: VesselAction(vessel_id=v, propeller_rpm=0.6, rudder_angle=0.0, message_targets=[])
        for v in range(2)
    }
    levels_by_t = {}
    jamming_by_t = {}
    for t in range(1, 12):
        env.step(actions)
        levels_by_t[t] = env.comms_degradation_level
        jamming_by_t[t] = env.scene.communication_channel.jamming_active

    assert levels_by_t[2] == pytest.approx(1.0)
    assert levels_by_t[4] == pytest.approx(0.2)
    assert levels_by_t[7] == pytest.approx(1.0)
    assert levels_by_t[9] == pytest.approx(0.0)
    assert jamming_by_t[9] is True
    assert levels_by_t[11] == pytest.approx(1.0)
    assert jamming_by_t[11] is False


def test_empty_comms_schedule_is_fully_backward_compatible():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2, episode_length=5)
    env = MaritimeCoordEnv(config)
    env.set_communication_degradation(0.4)
    env.reset(seed=1)
    actions = {
        v: VesselAction(vessel_id=v, propeller_rpm=0.6, rudder_angle=0.0, message_targets=[])
        for v in range(2)
    }
    for _ in range(5):
        env.step(actions)
        assert env.comms_degradation_level == pytest.approx(0.4)


# --- Part 6: the 4 new scenario types run end-to-end -------------------------


@pytest.mark.parametrize(
    "scenario_type,n_vessels",
    [
        ("multi_vessel_channel_convergence", 6),
        ("congested_port_approach", 8),
        ("restricted_visibility_crossing", 4),
        ("comms_blackout_transit", 5),
    ],
)
def test_new_scenario_types_run_a_full_episode(scenario_type, n_vessels):
    kwargs = {}
    if scenario_type == "restricted_visibility_crossing":
        kwargs["environment_condition"] = EnvironmentCondition.FOG
    if scenario_type == "comms_blackout_transit":
        kwargs["comms_schedule"] = [CommsScheduleEvent(2, 4, 0.0)]

    config = MaritimeExperimentConfig(
        scenario_type=scenario_type, n_vessels=n_vessels, episode_length=8, **kwargs
    )
    env = MaritimeCoordEnv(config)
    obs, info = env.reset(seed=2)
    assert len(obs) == n_vessels

    actions = {
        v: VesselAction(vessel_id=v, propeller_rpm=0.6, rudder_angle=0.0, message_targets=[])
        for v in range(n_vessels)
    }
    done = False
    while not done:
        obs, rewards, team_reward, done, info = env.step(actions)
    assert isinstance(team_reward, float)


# --- Part 7: the multi-axis sweep driver -------------------------------------


def test_experiment_matrix_produces_populated_result_and_skips_mismatched_maddpg():
    result = run_experiment_matrix(
        scenario_types=["head_on", "crossing_give_way"],
        n_vessels_list=[2, 4],
        environment_conditions=[EnvironmentCondition.CLEAR],
        comms_schedules=[[]],
        algorithms=["rule_based", "maddpg"],
        n_episodes=1,
        seeds=[1],
        maddpg_fixed_n_vessels=2,
    )
    assert len(result.episodes) > 0
    assert len(result.baseline_comparison) > 0
    run_ids = list(result.baseline_comparison.keys())
    assert any("maddpg" in r and "n2" in r for r in run_ids)
    assert not any("maddpg" in r and "n4" in r for r in run_ids)
    assert any("rule_based" in r and "n4" in r for r in run_ids)
