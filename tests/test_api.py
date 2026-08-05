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


def test_configure_unknown_key_raises_value_error():
    api = marlin_twin.MarlinTwinAPI()
    with pytest.raises(ValueError):
        api.configure(not_a_real_config_field=123)


def test_save_config_load_config_roundtrip(tmp_path):
    api = marlin_twin.MarlinTwinAPI()
    api.configure(n_vessels=7, scenario_type="port_approach")
    path = str(tmp_path / "config.yaml")

    api.save_config(path)

    reloaded = marlin_twin.MarlinTwinAPI()
    reloaded.load_config(path)
    assert reloaded.config.n_vessels == 7
    assert reloaded.config.scenario_type == "port_approach"
    assert isinstance(reloaded.config.boundaries, tuple)
