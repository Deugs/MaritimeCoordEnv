from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.agents.policies import GATPolicy
from marlin_twin.training.eval import MultiScenarioEvaluator


def test_evaluate_scenario_returns_expected_summary_fields():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2, episode_length=15)
    env = MaritimeCoordEnv(config)
    policies = {i: GATPolicy() for i in range(2)}

    results = MultiScenarioEvaluator.evaluate_scenario(
        env, policies, scenario_name="channel", n_episodes=2
    )

    assert results["scenario"] == "channel"
    assert 0.0 <= results["safety_score"] <= 1.0
    assert isinstance(results["mean_reward"], float)
    assert isinstance(results["std_reward"], float)
    assert results["mean_cpa"] > 0.0


def test_evaluate_scenario_respects_communication_degradation():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2, episode_length=15)
    env = MaritimeCoordEnv(config)
    policies = {i: GATPolicy() for i in range(2)}

    MultiScenarioEvaluator.evaluate_scenario(
        env, policies, scenario_name="channel", n_episodes=1, comms_level=0.0
    )

    assert env.comms_degradation_level == 0.0
