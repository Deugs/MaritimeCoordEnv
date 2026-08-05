"""Abstract base class for maritime coordination environments."""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from marlin_twin.data_classes import (
    MaritimeExperimentConfig,
    MaritimeScene,
    VesselObservation,
    VesselAction,
)


class BaseMaritimeEnvironment(ABC):
    """Abstract base for maritime coordination environment."""

    def __init__(self, config: MaritimeExperimentConfig):
        self.config = config
        self.scene: MaritimeScene | None = None
        self.time_step: int = 0
        self.dt: float = 1.0  # Simulation timestep (seconds)
        self.comms_degradation_level: float = 1.0
        self._initialized = False

    @abstractmethod
    def reset(
        self,
        scenario_type: str | None = None,
        n_vessels: int | None = None,
        seed: int | None = None,
    ) -> tuple[dict[int, VesselObservation], dict]:
        """Reset environment to initial maritime scene."""

    @abstractmethod
    def step(self, actions: dict[int, VesselAction]) -> tuple[
        dict[int, VesselObservation],
        dict[int, float],
        float,  # team reward
        bool,  # done
        dict,  # info
    ]:
        """Execute one environment step (1 second)."""

    @abstractmethod
    def get_scene(self) -> MaritimeScene:
        """Return current maritime scene."""

    def render(self, mode: str = "human") -> NDArray | None:
        """Render current scene."""
        return None

    def close(self) -> None:
        """Clean up resources."""

    def set_communication_degradation(
        self, level: float, jamming_zone: tuple[float, float, float] | None = None
    ) -> None:
        """Set communication degradation level (0.0 = complete loss, 1.0 = full capability)."""
        self.comms_degradation_level = max(0.0, min(1.0, level))
        if self.scene and self.scene.communication_channel:
            self.scene.communication_channel.weather_degradation = (
                1.0 - self.comms_degradation_level
            )
            if jamming_zone:
                self.scene.communication_channel.jamming_active = True
                self.scene.communication_channel.jamming_zone = jamming_zone
