"""Time-scrubbable playback controller for recorded voyage episodes."""

from loguru import logger

from marlin_twin.data_classes import VoyageEpisode


class EpisodeReplayer:
    """Time-scrubbable replayer for voyage episodes."""

    def __init__(self, speed: float = 1.0):
        self.speed = speed

    def play(
        self, episode: VoyageEpisode, save_video: bool = False, output_path: str | None = None
    ) -> None:
        logger.info(f"[Replayer] Playing episode {episode.episode_id} at {self.speed}x speed...")
        if save_video and output_path:
            logger.info(f"[Replayer] Exporting video to {output_path}...")
