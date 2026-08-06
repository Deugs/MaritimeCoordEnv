import dataclasses

import numpy as np
import pytest
from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.agents.vessel_agent import VesselAgentWrapper
from marlin_twin.agents.policies import GATPolicy
from marlin_twin.training.mappo import _build_scene_graph


def test_vessel_agent_wrapper_build_action_applies_exact_rescale_formula():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2)
    env = MaritimeCoordEnv(config)
    obs, _ = env.reset(seed=42)

    agent = env.get_scene().vessels[0]
    wrapper = VesselAgentWrapper(agent, GATPolicy())

    # Build directly from a known tanh-squashed action vector, bypassing the
    # policy's own (stochastic) sampling, so the rescale formula itself is
    # what's under test rather than whatever clip() happens to guarantee.
    action = wrapper.build_action(obs[0], np.array([0.4, -0.5], dtype=np.float32))

    assert action.vessel_id == agent.vessel_id
    assert action.propeller_rpm == pytest.approx(np.clip(0.4 * 0.5 + 0.6, 0.2, 1.0))
    assert action.rudder_angle == pytest.approx(np.clip(-0.5 * (np.pi / 6), -np.pi / 6, np.pi / 6))

    # Saturating inputs must hit the documented rpm/rudder bounds exactly.
    saturated = wrapper.build_action(obs[0], np.array([10.0, -10.0], dtype=np.float32))
    assert saturated.propeller_rpm == pytest.approx(1.0)
    assert saturated.rudder_angle == pytest.approx(-np.pi / 6)


def test_vessel_agent_wrapper_selects_only_neighbors_within_3km():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2)
    env = MaritimeCoordEnv(config)
    obs, _ = env.reset(seed=42)

    agent = env.get_scene().vessels[0]
    wrapper = VesselAgentWrapper(agent, GATPolicy())

    observation = obs[0]
    own_x, own_y = observation.own_state.x, observation.own_state.y
    template = next(iter(observation.neighbor_states.values()))
    near_state = template.copy_with(x=own_x + 1000.0, y=own_y)
    far_state = template.copy_with(x=own_x + 100_000.0, y=own_y)
    observation = dataclasses.replace(observation, neighbor_states={10: near_state, 99: far_state})

    action = wrapper.build_action(observation, np.array([0.0, 0.0], dtype=np.float32))

    assert action.message_targets == [10]


def test_vessel_agent_wrapper_uses_default_policy_when_none_given():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2)
    env = MaritimeCoordEnv(config)
    obs, _ = env.reset(seed=1)

    agent = env.get_scene().vessels[0]
    wrapper = VesselAgentWrapper(agent)

    assert isinstance(wrapper.policy, GATPolicy)
    graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))
    action = wrapper.select_action(obs[0], graph, node_idx_map[0], deterministic=True)
    assert action.vessel_id == agent.vessel_id
