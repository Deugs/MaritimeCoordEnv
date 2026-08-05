from marlin_twin.data_classes import (
    MaritimeExperimentConfig,
    MaritimeScene,
    MaritimeCommunicationChannel,
    MaritimeDigitalTwin,
    EnvironmentCondition,
)
from marlin_twin.envs.base_env import BaseMaritimeEnvironment


class _MinimalEnv(BaseMaritimeEnvironment):
    """Concrete stub exercising only the base class's non-abstract behavior."""

    def reset(self, scenario_type=None, n_vessels=None, seed=None):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def get_scene(self):
        return self.scene


def _make_scene() -> MaritimeScene:
    channel = MaritimeCommunicationChannel(
        channel_id="test_channel", bandwidth_bps=9600.0, base_latency=0.5, packet_loss_rate=0.0
    )
    twin = MaritimeDigitalTwin(
        scene_id="test_scene",
        timestamp=0.0,
        vessel_estimates={},
        ais_readings=[],
        radar_tracks=[],
        detected_encounters=[],
        predicted_trajectories={},
        collision_risks={},
        sensor_health={},
        communication_health={},
    )
    return MaritimeScene(
        scene_id="test_scene",
        timestamp=0.0,
        vessels={},
        communication_channel=channel,
        digital_twin=twin,
        boundaries=(-5000.0, -5000.0, 5000.0, 5000.0),
        environment_condition=EnvironmentCondition.CLEAR,
    )


def test_set_communication_degradation_clamps_to_unit_interval():
    env = _MinimalEnv(MaritimeExperimentConfig())

    env.set_communication_degradation(1.5)
    assert env.comms_degradation_level == 1.0

    env.set_communication_degradation(-0.5)
    assert env.comms_degradation_level == 0.0

    env.set_communication_degradation(0.4)
    assert env.comms_degradation_level == 0.4


def test_set_communication_degradation_updates_scene_channel_and_jamming():
    env = _MinimalEnv(MaritimeExperimentConfig())
    env.scene = _make_scene()

    env.set_communication_degradation(0.3, jamming_zone=(0.0, 0.0, 500.0))

    channel = env.scene.communication_channel
    assert channel.weather_degradation == 0.7
    assert channel.jamming_active is True
    assert channel.jamming_zone == (0.0, 0.0, 500.0)


def test_render_and_close_are_safe_no_ops():
    env = _MinimalEnv(MaritimeExperimentConfig())
    assert env.render() is None
    env.close()
