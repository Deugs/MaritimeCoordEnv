import numpy as np
import pytest
from marlin_twin.data_classes import (
    MaritimeExperimentConfig,
    VesselState,
    VesselObservation,
    Route,
    Waypoint,
    EnvironmentCondition,
)
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.agents.observation_builder import ObservationBuilder
from marlin_twin.baselines.independent_ppo import IndependentPPOPolicy
from marlin_twin.baselines.maddpg import MADDPGPolicy
from marlin_twin.baselines.rule_based import RuleBasedCOLREGsController
from marlin_twin.baselines.factory import BaselineFactory


def _make_observation(own_state: VesselState, neighbor_states: dict) -> VesselObservation:
    route = Route(
        vessel_id=own_state.vessel_id,
        waypoints=[Waypoint(0, own_state.x, own_state.y, own_state.speed)],
    )
    return VesselObservation(
        vessel_id=own_state.vessel_id,
        own_state=own_state,
        own_route=route,
        neighbor_states=neighbor_states,
        neighbor_intents={},
        environment=EnvironmentCondition.CLEAR,
        visibility_range=10000.0,
        wind_speed=0.0,
        wind_direction=0.0,
        current_speed=0.0,
        current_direction=0.0,
        comm_link_quality={},
        last_message_timestamp={},
        estimated_neighbor_states={},
        estimation_confidence={},
        active_encounters=[],
        colregs_compliance_score=1.0,
    )


def test_independent_ppo_and_maddpg_act_like_gat_policy():
    config = MaritimeExperimentConfig(scenario_type="head_on", n_vessels=2)
    env = MaritimeCoordEnv(config)
    obs, _ = env.reset(seed=1)
    vec = ObservationBuilder.to_vector(obs[0])

    for policy_cls in [IndependentPPOPolicy, MADDPGPolicy]:
        action = policy_cls().act(vec, deterministic=True)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))


def test_rule_based_controller_alters_course_on_close_head_on_encounter():
    own_state = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=10.0)
    neighbor_state = VesselState(vessel_id=1, x=0.0, y=1000.0, heading=np.pi, speed=10.0)
    observation = _make_observation(own_state, {1: neighbor_state})

    controller = RuleBasedCOLREGsController(vessel_id=0)
    action = controller.act(observation, deterministic=True)

    assert action.shape == (2,)
    assert action[1] == pytest.approx(np.pi / 12)  # 15 deg alteration to starboard


def test_rule_based_controller_holds_course_with_no_nearby_traffic():
    own_state = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=10.0)
    observation = _make_observation(own_state, {})

    controller = RuleBasedCOLREGsController(vessel_id=0)
    action = controller.act(observation, deterministic=True)

    assert action[1] == pytest.approx(0.0)


def test_baseline_factory_unknown_algorithm_raises_value_error():
    config = MaritimeExperimentConfig(n_vessels=2)
    factory = BaselineFactory(config)

    with pytest.raises(ValueError):
        factory.create("nonexistent_algorithm")
