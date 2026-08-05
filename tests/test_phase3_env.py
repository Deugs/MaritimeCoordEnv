# ============================================================================
# FILE: tests/test_phase3_env.py
# ============================================================================

import numpy as np
from marlin_twin.data_classes import (
    MaritimeExperimentConfig,
    VesselAction,
    EncounterType,
    COLREGsRule,
    Encounter,
)
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.agents.observation_builder import ObservationBuilder
from marlin_twin.agents.reward_shaping import COLREGsRewardShaper


def test_gymnasium_env_reset_and_step():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=3, episode_length=50)
    env = MaritimeCoordEnv(config)

    obs, info = env.reset(seed=42)
    assert len(obs) == 3
    assert 0 in obs
    assert obs[0].own_state.speed > 0.0

    actions = {
        i: VesselAction(vessel_id=i, propeller_rpm=0.8, rudder_angle=0.0, message_targets=[])
        for i in range(3)
    }
    next_obs, rewards, team_reward, done, step_info = env.step(actions)

    assert len(next_obs) == 3
    assert len(rewards) == 3
    assert isinstance(team_reward, float)
    assert done is False


def test_observation_builder_to_vector_and_graph():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=3)
    env = MaritimeCoordEnv(config)
    obs, _ = env.reset(seed=42)

    vec = ObservationBuilder.to_vector(obs[0], vector_dim=32)
    assert vec.shape == (32,)
    assert isinstance(vec[0], np.float32)

    states = {vid: o.own_state for vid, o in obs.items()}
    graph = ObservationBuilder.to_pyg_graph(states)
    if graph is not None:
        assert graph.x.shape[0] == 3
        assert graph.x.shape[1] == 6
        assert graph.edge_index.shape[0] == 2


def test_colregs_rule17_reward_shaping():
    from marlin_twin.data_classes import VesselState

    v0 = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=10.0)
    act_evasion = VesselAction(vessel_id=0, propeller_rpm=0.8, rudder_angle=0.2, message_targets=[])

    encounters = [
        Encounter(
            vessel_i=0,
            vessel_j=1,
            encounter_type=EncounterType.CROSSING_STAND_ON,
            colregs_rule=COLREGsRule.RULE_17_STAND_ON,
            cpa_distance=600.0,
            cpa_time=40.0,
            tcpa=40.0,
            dcpa=600.0,
            relative_bearing=0.2,
        )
    ]

    r = COLREGsRewardShaper.compute_reward(v0, act_evasion, encounters)
    assert isinstance(r, float)
    assert r > 0.0  # Evasion bonus applied without safety penalty offset
