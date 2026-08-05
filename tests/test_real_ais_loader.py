# ============================================================================
# FILE: tests/test_real_ais_loader.py
# ============================================================================

import pandas as pd
import pytest
from marlin_twin.data.ais_loader import AISDataLoader


def test_latlon_to_meters():
    x, y = AISDataLoader.latlon_to_meters(37.7750, -122.4190, 37.7749, -122.4194)
    assert isinstance(x, float)
    assert isinstance(y, float)
    assert x > 0.0
    assert y > 0.0


def test_generate_sample_ais_trajectory():
    df = AISDataLoader.generate_sample_ais_trajectory(n_steps=20, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
    assert "MMSI" in df.columns
    assert "LAT" in df.columns
    assert "LON" in df.columns


def test_convert_to_vessel_states():
    df = AISDataLoader.generate_sample_ais_trajectory(n_steps=10, seed=42)
    states = AISDataLoader.convert_to_vessel_states(df, vessel_id=1)
    assert len(states) == 10
    assert states[0].vessel_id == 1
    assert states[0].speed > 0.0


def test_load_ais_csv_missing_file_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        AISDataLoader.load_ais_csv(str(tmp_path / "does_not_exist.csv"))


def test_load_ais_csv_missing_required_column_raises_value_error(tmp_path):
    csv_path = tmp_path / "incomplete.csv"
    csv_path.write_text("MMSI,LAT,LON,SOG\n123,37.0,-122.0,10.0\n")  # missing COG

    with pytest.raises(ValueError):
        AISDataLoader.load_ais_csv(str(csv_path))
