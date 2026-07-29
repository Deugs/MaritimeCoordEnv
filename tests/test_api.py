import pytest
import marlin_twin

def test_minimal_api_creation():
    api = marlin_twin.create_minimal_api(scenario_type="channel", n_vessels=3)
    assert api is not None
    assert api.env is not None
    assert api.config.n_vessels == 3

def test_api_run():
    api = marlin_twin.create_minimal_api(scenario_type="open_water", n_vessels=2)
    result = api.train_and_evaluate(n_episodes=2)
    assert result is not None
    assert result.resilience_metrics is not None
