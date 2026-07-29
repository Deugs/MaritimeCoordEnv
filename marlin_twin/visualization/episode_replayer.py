# ============================================================================
# FILE: marlin_twin/visualization/episode_replayer.py
# ============================================================================

from marlin_twin.data_classes import VoyageEpisode

class EpisodeReplayer:
    """Time-scrubbable replayer for voyage episodes."""

    def __init__(self, speed: float = 1.0):
        self.speed = speed

    def play(self, episode: VoyageEpisode, save_video: bool = False, output_path: str | None = None) -> None:
        print(f"[Replayer] Playing episode {episode.episode_id} at {self.speed}x speed...")
        if save_video and output_path:
            print(f"[Replayer] Exporting video to {output_path}...")
