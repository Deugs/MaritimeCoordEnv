import numpy as np
from marlin_twin.data_classes import (
    CoordinationResilienceMetrics,
    VoyageEpisode,
    SceneTransition,
    MaritimeScene,
    MaritimeCommunicationChannel,
    MaritimeDigitalTwin,
    MaritimeMessage,
    MessagePriority,
    MaritimeExperimentResult,
    MaritimeExperimentConfig,
    EnvironmentCondition,
)
from marlin_twin.envs.scenarios import ScenarioGenerator
from marlin_twin.visualization.plots import MaritimeVisualizer
from marlin_twin.visualization.episode_replayer import EpisodeReplayer


def _make_scene(timestamp: float, x_offset: float = 0.0) -> MaritimeScene:
    vessels = ScenarioGenerator.create_scenario("channel", n_vessels=2, seed=1)
    for agent in vessels.values():
        agent.current_state = agent.current_state.copy_with(x=agent.current_state.x + x_offset)
    channel = MaritimeCommunicationChannel(
        channel_id="test", bandwidth_bps=9600.0, base_latency=0.5, packet_loss_rate=0.0
    )
    twin = MaritimeDigitalTwin(
        scene_id="test",
        timestamp=timestamp,
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
        scene_id="test",
        timestamp=timestamp,
        vessels=vessels,
        communication_channel=channel,
        digital_twin=twin,
        boundaries=(-5000.0, -5000.0, 5000.0, 5000.0),
        environment_condition=EnvironmentCondition.CLEAR,
    )


def _make_episode() -> VoyageEpisode:
    message = MaritimeMessage(
        sender_id=0,
        receiver_id=1,
        content=np.zeros(4, dtype=np.float32),
        priority=MessagePriority.MEDIUM,
        timestamp=0.0,
        size_bits=256,
    )
    scene_0 = _make_scene(0.0, x_offset=0.0)
    scene_1 = _make_scene(1.0, x_offset=10.0)
    scene_2 = _make_scene(2.0, x_offset=20.0)
    transitions = [
        SceneTransition(
            scene=scene_0,
            observations={},
            actions={},
            messages=[message],
            next_scene=scene_1,
            rewards={0: 1.0, 1: 1.0},
            team_reward=1.0,
            done=False,
            info={},
        ),
        SceneTransition(
            scene=scene_1,
            observations={},
            actions={},
            messages=[],
            next_scene=scene_2,
            rewards={0: 1.0, 1: 1.0},
            team_reward=1.0,
            done=True,
            info={},
        ),
    ]
    return VoyageEpisode(episode_id="ep_1", transitions=transitions, total_reward=2.0, length=2)


def test_plot_trajectories_draws_real_vessel_positions(tmp_path):
    viz = MaritimeVisualizer()
    episode = _make_episode()
    output_path = tmp_path / "trajectories.png"

    frame = viz.plot_trajectories(episode=episode, output_path=str(output_path))

    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_trajectories_without_episode_still_renders_blank_axes(tmp_path):
    viz = MaritimeVisualizer()
    output_path = tmp_path / "blank.png"

    frame = viz.plot_trajectories(output_path=str(output_path))

    assert isinstance(frame, np.ndarray)
    assert output_path.exists()


def test_plot_resilience_curve_renders_the_provided_curve_not_the_default(tmp_path):
    viz = MaritimeVisualizer()
    default_frame = viz.plot_resilience_curve(CoordinationResilienceMetrics())

    custom_metrics = CoordinationResilienceMetrics(
        degradation_levels=[1.0, 0.5, 0.0], safety_scores=[1.0, 0.8, 0.4]
    )
    custom_frame = viz.plot_resilience_curve(custom_metrics)

    assert default_frame.shape == custom_frame.shape
    # Different underlying data must produce a different rendered image.
    assert not np.array_equal(default_frame, custom_frame)


def test_plot_communication_heatmap_uses_real_message_data_when_available():
    viz = MaritimeVisualizer()
    episode = _make_episode()
    result = MaritimeExperimentResult(
        config=MaritimeExperimentConfig(n_vessels=2), episodes=[episode]
    )

    real_frame = viz.plot_communication_heatmap(result=result)
    placeholder_frame = viz.plot_communication_heatmap(result=None)

    assert isinstance(real_frame, np.ndarray)
    assert isinstance(placeholder_frame, np.ndarray)


def test_plot_colregs_compliance_uses_results_resilience_metrics():
    viz = MaritimeVisualizer()
    metrics = CoordinationResilienceMetrics(
        degradation_levels=[1.0, 0.5, 0.0], safety_scores=[1.0, 0.8, 0.4]
    )
    result = MaritimeExperimentResult(
        config=MaritimeExperimentConfig(n_vessels=2), resilience_metrics=metrics
    )

    from_result = viz.plot_colregs_compliance(result=result)
    from_default = viz.plot_colregs_compliance(result=None)

    assert not np.array_equal(from_result, from_default)


def test_episode_replayer_play_does_not_raise(tmp_path):
    episode = VoyageEpisode(episode_id="ep_1", transitions=[], total_reward=12.5, length=0)
    replayer = EpisodeReplayer(speed=2.0)

    replayer.play(episode, save_video=True, output_path=str(tmp_path / "out.mp4"))
