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
from marlin_twin.baselines.independent_ppo import IndependentPPOPolicy
from marlin_twin.baselines.maddpg import MADDPGPolicy
from marlin_twin.baselines.rule_based import RuleBasedCOLREGsController
from marlin_twin.baselines.factory import BaselineFactory
from marlin_twin.training.mappo import _build_scene_graph


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
    graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))

    ippo_action = IndependentPPOPolicy().act(obs[0], deterministic=True)
    assert ippo_action.shape == (2,)
    assert np.all(np.isfinite(ippo_action))

    maddpg_action = MADDPGPolicy(n_vessels=2).act(
        obs[0], graph, node_idx_map[0], deterministic=True
    )
    assert maddpg_action.shape == (2,)
    assert np.all(np.isfinite(maddpg_action))


def test_rule_based_controller_alters_course_on_close_head_on_encounter():
    own_state = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=10.0)
    neighbor_state = VesselState(vessel_id=1, x=0.0, y=1000.0, heading=np.pi, speed=10.0)
    observation = _make_observation(own_state, {1: neighbor_state})

    controller = RuleBasedCOLREGsController(vessel_id=0)
    action = controller.act(observation, deterministic=True)

    assert action.shape == (2,)
    # act() emits [-1,1] tanh-space, not physical radians -- 0.5 is the
    # exact inverse of build_action's rudder=clip(a*(pi/6),...) for a
    # physical 15 deg (pi/12) alteration to starboard.
    assert action[1] == pytest.approx(0.5)


def test_rule_based_controller_gives_way_when_overtaking():
    """Regression guard: the controller used to have no branch at all for
    OVERTAKING/OVERTAKEN, so it took zero avoidance action (rudder=0.0) for
    an encounter type it now can actually be asked to handle once the
    `overtaking` scenario produces a real speed differential (see
    test_experimental_scenarios.py). `classify_encounter`'s Rule 13 check
    is a bearing-of-the-other-vessel test: own (state_i) classifies as
    OVERTAKING when the neighbor is >112.5 deg abaft its beam (here, dead
    astern) and own is faster -- Rule 13 then requires the overtaking
    vessel to keep clear."""
    own_state = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=11.0)
    neighbor_state = VesselState(vessel_id=1, x=0.0, y=-200.0, heading=0.0, speed=4.0)
    observation = _make_observation(own_state, {1: neighbor_state})

    controller = RuleBasedCOLREGsController(vessel_id=0)
    action = controller.act(observation, deterministic=True)

    assert action[1] == pytest.approx(0.5)  # 15 deg give-way alteration, tanh-space


def test_rule_based_controller_holds_course_with_no_nearby_traffic():
    own_state = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=10.0)
    observation = _make_observation(own_state, {})

    controller = RuleBasedCOLREGsController(vessel_id=0)
    action = controller.act(observation, deterministic=True)

    assert action[1] == pytest.approx(0.0)


def test_rule_based_controller_action_round_trips_through_build_action():
    """The whole point of emitting tanh-space actions: build_action's
    physical remapping must invert exactly back to the maneuver
    RuleBasedCOLREGsController actually specifies (0.8 rpm fraction, 15 deg
    starboard alteration), not a different one. Regression guard for the bug
    where every evaluation path fed the pre-fix raw radians/rpm-fraction
    straight into build_action's tanh-space remapping and silently ran a
    different maneuver (always-full-throttle, ~half rudder angle)."""
    from types import SimpleNamespace
    from marlin_twin.agents.vessel_agent import VesselAgentWrapper

    own_state = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=10.0)
    neighbor_state = VesselState(vessel_id=1, x=0.0, y=1000.0, heading=np.pi, speed=10.0)
    observation = _make_observation(own_state, {1: neighbor_state})

    controller = RuleBasedCOLREGsController(vessel_id=0)
    action_vec = controller.act(observation, deterministic=True)

    # build_action only reads `self.agent.vessel_id` -- a bare namespace
    # avoids constructing an unrelated VesselSpecification/VesselDynamics
    # this test doesn't otherwise need.
    agent = SimpleNamespace(vessel_id=0)
    wrapper = VesselAgentWrapper(agent, controller)
    vessel_action = wrapper.build_action(observation, action_vec)

    assert vessel_action.propeller_rpm == pytest.approx(0.8)
    assert vessel_action.rudder_angle == pytest.approx(np.pi / 12)


def test_baseline_factory_unknown_algorithm_raises_value_error():
    config = MaritimeExperimentConfig(n_vessels=2)
    factory = BaselineFactory(config)

    with pytest.raises(ValueError):
        factory.create("nonexistent_algorithm")
