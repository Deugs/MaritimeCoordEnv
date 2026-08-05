"""Open AIS dataset loading and conversion to VesselState trajectories."""

import os
import numpy as np
import pandas as pd
from marlin_twin.data_classes import VesselState
from marlin_twin.utils.seeding import seed_everything


class AISDataLoader:
    """
    Open AIS Dataset Importer and Real-World Trajectory Converter.
    Parses USCG MarineCadastre / DMA AIS CSVs into MARLIN-Twin VesselStates.
    """

    @staticmethod
    def latlon_to_meters(
        lat: float, lon: float, ref_lat: float, ref_lon: float
    ) -> tuple[float, float]:
        """Converts Geographic (Lat, Lon) to Local Flat-Earth ENU coordinates in meters."""
        R = 6371000.0  # Earth radius in meters
        dlat = np.radians(lat - ref_lat)
        dlon = np.radians(lon - ref_lon)
        x = R * dlat
        y = R * dlon * np.cos(np.radians(ref_lat))
        return float(x), float(y)

    @staticmethod
    def generate_sample_ais_trajectory(n_steps: int = 100, seed: int = 42) -> pd.DataFrame:
        """Generates a realistic synthetic AIS trajectory simulating real vessel movement."""
        seed_everything(seed)
        times = np.arange(0, n_steps * 10, 10)  # 10s intervals
        ref_lat, ref_lon = 37.7749, -122.4194  # San Francisco Bay

        # Simulating realistic vessel turning and acceleration
        speed_knots = 12.0 + 1.5 * np.sin(times / 200.0)
        heading_deg = (45.0 + 15.0 * np.cos(times / 300.0)) % 360.0

        speeds_ms = speed_knots * 0.514444
        headings_rad = np.radians(heading_deg)

        dx = speeds_ms * np.cos(headings_rad) * 10.0
        dy = speeds_ms * np.sin(headings_rad) * 10.0

        xs = np.cumsum(dx)
        ys = np.cumsum(dy)

        # Convert back to lat/lon for dataset format
        R = 6371000.0
        lats = ref_lat + np.degrees(xs / R)
        lons = ref_lon + np.degrees(ys / (R * np.cos(np.radians(ref_lat))))

        df = pd.DataFrame(
            {
                "MMSI": 367123456,
                "BaseDateTime": pd.date_range("2026-01-01 12:00:00", periods=n_steps, freq="10s"),
                "LAT": lats,
                "LON": lons,
                "SOG": speed_knots,
                "COG": heading_deg,
                "Heading": heading_deg,
                "VesselName": "REAL_AIS_VESSEL_01",
            }
        )
        return df

    @classmethod
    def load_ais_csv(cls, filepath: str) -> pd.DataFrame:
        """Loads and cleans AIS CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"AIS dataset file not found: {filepath}")

        df = pd.read_csv(filepath)
        required_cols = ["MMSI", "LAT", "LON", "SOG", "COG"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in AIS dataset: {col}")

        df = df.dropna(subset=required_cols).sort_values("BaseDateTime").reset_index(drop=True)
        return df

    @classmethod
    def convert_to_vessel_states(cls, df: pd.DataFrame, vessel_id: int = 0) -> list[VesselState]:
        """Converts AIS DataFrame rows into a sequence of MARLIN-Twin VesselState objects."""
        ref_lat = df["LAT"].iloc[0]
        ref_lon = df["LON"].iloc[0]

        states = []
        for idx, row in df.iterrows():
            x, y = cls.latlon_to_meters(row["LAT"], row["LON"], ref_lat, ref_lon)
            speed = float(row["SOG"]) * 0.514444  # Knots to m/s
            heading = float(np.radians(row["COG"]))

            st = VesselState(vessel_id=vessel_id, x=x, y=y, heading=heading, speed=speed)
            states.append(st)

        return states
