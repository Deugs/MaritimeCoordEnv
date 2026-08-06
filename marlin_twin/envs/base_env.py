"""Abstract base class for maritime coordination environments."""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from marlin_twin.data_classes import (
    CommsScheduleEvent,
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
        self._comms_schedule: list[CommsScheduleEvent] = []
        self._comms_baseline_level: float = 1.0
        self._comms_baseline_jamming_zone: tuple[float, float, float] | None = None
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
        """Set communication degradation level (0.0 = complete loss, 1.0 = full capability).

        `jamming_zone=None` explicitly clears any previously-active jamming
        (not just "leave it as-is") — required so a scheduled jamming event
        (see `set_communication_schedule`) actually turns off once its
        window ends, rather than jamming persisting forever after the first
        time it's ever set."""
        self.comms_degradation_level = max(0.0, min(1.0, level))
        if self.scene and self.scene.communication_channel:
            self.scene.communication_channel.weather_degradation = (
                1.0 - self.comms_degradation_level
            )
            self.scene.communication_channel.jamming_active = jamming_zone is not None
            self.scene.communication_channel.jamming_zone = jamming_zone

    def set_communication_schedule(self, events: list[CommsScheduleEvent]) -> None:
        """Time-varying overrides on top of the static degradation level —
        e.g. `[CommsScheduleEvent(50, 90, 0.1), CommsScheduleEvent(100, 150,
        0.0, jamming_zone=(0, 0, 1000))]` to script a mid-transit
        degradation dip followed by a jamming window. An empty list (the
        default) leaves `set_communication_degradation`'s static behavior
        completely unchanged."""
        self._comms_schedule = list(events)

    def _apply_comms_schedule(self, t: float) -> None:
        """Called every step (and once at `t=0` in `reset`) with the current
        elapsed episode time. No-op when no schedule is set. When multiple
        events overlap at `t`, the last one in the list wins. When no event
        is active at `t`, reverts to the level/jamming-zone that was in
        effect before the schedule took over (captured by the caller, e.g.
        `MaritimeCoordEnv.reset`, as `_comms_baseline_level`/
        `_comms_baseline_jamming_zone`)."""
        if not self._comms_schedule:
            return
        active = None
        for event in self._comms_schedule:
            if event.active_at(t):
                active = event
        if active is not None:
            self.set_communication_degradation(active.degradation_level, active.jamming_zone)
        else:
            self.set_communication_degradation(
                self._comms_baseline_level, self._comms_baseline_jamming_zone
            )
