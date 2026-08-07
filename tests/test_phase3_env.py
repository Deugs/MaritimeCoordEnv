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


def test_comms_degradation_reaches_digital_twin_estimate():
    """Regression guard for a real bug: radar used to be generated every
    step regardless of comms_degradation_level, silently backstopping every
    dropped AIS packet with a similarly-good estimate, so degraded comms
    had no observable effect on what the policy could see. Radar now has
    its own (smaller) drop probability tied to degradation, so full
    degradation should measurably shift the Digital Twin off kalman_ais
    more often than full communication does."""
    n_steps = 60
    kalman_counts = {}
    for lam in (1.0, 0.0):
        config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2, episode_length=200)
        env = MaritimeCoordEnv(config)
        env.set_communication_degradation(lam)
        env.reset(seed=7)
        kalman_steps = 0
        for _ in range(n_steps):
            actions = {
                i: VesselAction(
                    vessel_id=i, propeller_rpm=0.6, rudder_angle=0.0, message_targets=[]
                )
                for i in range(2)
            }
            env.step(actions)
            est0 = env.scene.digital_twin.vessel_estimates.get(0)
            if est0 and est0.estimation_method == "kalman_ais":
                kalman_steps += 1
        kalman_counts[lam] = kalman_steps

    assert kalman_counts[1.0] > kalman_counts[0.0]


def test_delivered_message_supplements_dropped_ais_reading():
    """A delivered V2V message (the sender reporting its own state) should
    be able to rescue the Digital Twin back onto kalman_ais even when AIS
    itself was dropped -- previously `comm_manager.process_step`'s
    `delivered` return value was discarded entirely, so this had no
    effect regardless of degradation level."""
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2, episode_length=200)
    env = MaritimeCoordEnv(config)
    env.set_communication_degradation(0.5)  # partial, independent AIS/message loss
    env.reset(seed=11)

    n_steps = 200
    kalman_steps = 0
    for _ in range(n_steps):
        actions = {
            i: VesselAction(
                vessel_id=i, propeller_rpm=0.6, rudder_angle=0.0, message_targets=[1 - i]
            )
            for i in range(2)
        }
        env.step(actions)
        est0 = env.scene.digital_twin.vessel_estimates.get(0)
        if est0 and est0.estimation_method == "kalman_ais":
            kalman_steps += 1

    # AIS alone succeeds ~50% of the time at this degradation level; with
    # messages independently backstopping dropped AIS, the combined rate
    # should be well above that.
    assert kalman_steps / n_steps > 0.60


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
