"""AIS and radar sensor simulation with realistic noise models."""

import numpy as np
from marlin_twin.data_classes import VesselState, AISReading, RadarTrack


class SensorSimulator:
    """Simulates AIS and marine radar sensor observations with realistic noise."""

    @staticmethod
    def generate_ais(
        state: VesselState, timestamp: float, drop_prob: float = 0.05
    ) -> AISReading | None:
        if np.random.rand() < drop_prob:
            return None  # Packet dropped

        # ITU-R M.1371 standard position noise ~ 5m
        pos_noise = np.random.normal(0, 5.0, size=2)
        heading_noise = np.random.normal(0, np.radians(0.5))
        speed_noise = np.random.normal(0, 0.2)

        return AISReading(
            vessel_id=state.vessel_id,
            timestamp=timestamp,
            reported_position=(float(state.x + pos_noise[0]), float(state.y + pos_noise[1])),
            reported_heading=float((state.heading + heading_noise + np.pi) % (2 * np.pi) - np.pi),
            reported_speed=float(max(0.0, state.speed + speed_noise)),
            confidence=0.95,
        )

    @staticmethod
    def generate_radar(state: VesselState, timestamp: float, track_id: int) -> RadarTrack:
        # Marine radar position noise ~ 15m
        pos_noise = np.random.normal(0, 15.0, size=2)
        vel_noise = np.random.normal(0, 0.5, size=2)

        vel_vec = state.velocity_vector()
        return RadarTrack(
            track_id=track_id,
            timestamp=timestamp,
            position=(float(state.x + pos_noise[0]), float(state.y + pos_noise[1])),
            velocity=(float(vel_vec[0] + vel_noise[0]), float(vel_vec[1] + vel_noise[1])),
            confidence=0.85,
            associated_vessel=state.vessel_id,
        )

    @classmethod
    def generate_scene_sensors(
        cls, vessels: dict[int, VesselState], timestamp: float, ais_drop_prob: float = 0.05
    ) -> tuple[dict[int, AISReading], list[RadarTrack]]:
        """Generate full sensor readings across all vessels in scene."""
        ais_readings = {}
        radar_tracks = []

        for vid, state in vessels.items():
            ais = cls.generate_ais(state, timestamp, drop_prob=ais_drop_prob)
            if ais is not None:
                ais_readings[vid] = ais

            radar = cls.generate_radar(state, timestamp, track_id=1000 + vid)
            radar_tracks.append(radar)

        return ais_readings, radar_tracks
