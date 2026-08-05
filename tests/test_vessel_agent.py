import numpy as np
from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.agents.vessel_agent import VesselAgentWrapper
from marlin_twin.agents.policies import GATPolicy


def test_vessel_agent_wrapper_select_action_produces_valid_action():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2)
    env = MaritimeCoordEnv(config)
    obs, _ = env.reset(seed=42)

    agent = env.get_scene().vessels[0]
    wrapper = VesselAgentWrapper(agent, GATPolicy())

    action = wrapper.select_action(obs[0], deterministic=True)

    assert action.vessel_id == agent.vessel_id
    assert 0.2 <= action.propeller_rpm <= 1.0
    assert -np.pi / 6 <= action.rudder_angle <= np.pi / 6
    assert len(action.message_targets) <= 3


def test_vessel_agent_wrapper_uses_default_policy_when_none_given():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2)
    env = MaritimeCoordEnv(config)
    obs, _ = env.reset(seed=1)

    agent = env.get_scene().vessels[0]
    wrapper = VesselAgentWrapper(agent)

    assert isinstance(wrapper.policy, GATPolicy)
    action = wrapper.select_action(obs[0], deterministic=True)
    assert action.vessel_id == agent.vessel_id
