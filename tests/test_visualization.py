import numpy as np
from marlin_twin.data_classes import CoordinationResilienceMetrics, VoyageEpisode
from marlin_twin.visualization.plots import MaritimeVisualizer
from marlin_twin.visualization.episode_replayer import EpisodeReplayer


def test_plot_trajectories_returns_rgb_array_and_writes_file(tmp_path):
    viz = MaritimeVisualizer()
    output_path = tmp_path / "trajectories.png"

    frame = viz.plot_trajectories(output_path=str(output_path))

    assert frame.shape == (400, 400, 3)
    assert output_path.exists()


def test_plot_resilience_curve_uses_provided_metrics(tmp_path):
    viz = MaritimeVisualizer()
    metrics = CoordinationResilienceMetrics(
        degradation_levels=[1.0, 0.5, 0.0], safety_scores=[1.0, 0.8, 0.4]
    )
    output_path = tmp_path / "resilience.png"

    frame = viz.plot_resilience_curve(metrics, output_path=str(output_path))

    assert frame.shape == (400, 400, 3)
    assert output_path.exists()


def test_plot_communication_heatmap_and_colregs_compliance_smoke(tmp_path):
    viz = MaritimeVisualizer()

    heatmap_frame = viz.plot_communication_heatmap(output_path=str(tmp_path / "heatmap.png"))
    compliance_frame = viz.plot_colregs_compliance(output_path=str(tmp_path / "compliance.png"))

    assert isinstance(heatmap_frame, np.ndarray)
    assert isinstance(compliance_frame, np.ndarray)


def test_episode_replayer_play_does_not_raise(tmp_path):
    episode = VoyageEpisode(episode_id="ep_1", transitions=[], total_reward=12.5, length=0)
    replayer = EpisodeReplayer(speed=2.0)

    replayer.play(episode, save_video=True, output_path=str(tmp_path / "out.mp4"))
