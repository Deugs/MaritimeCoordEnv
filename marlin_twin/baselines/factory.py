"""Factory for constructing benchmark baseline policy dictionaries."""

from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.api import Policy
from marlin_twin.baselines.rule_based import RuleBasedCOLREGsController
from marlin_twin.baselines.independent_ppo import IndependentPPOPolicy
from marlin_twin.baselines.maddpg import MADDPGPolicy


class BaselineFactory:
    """Factory creating benchmark baseline policy dictionaries."""

    def __init__(self, config: MaritimeExperimentConfig):
        self.config = config

    def create(self, algorithm: str) -> dict[int, Policy]:
        n_vessels = self.config.n_vessels
        if algorithm in ["marlin_twin", "gat"]:
            from marlin_twin.agents.policies import GATPolicy

            return {i: GATPolicy() for i in range(n_vessels)}
        elif algorithm in ["flat_mlp", "mlp"]:
            from marlin_twin.agents.policies import MLPPolicy

            return {i: MLPPolicy() for i in range(n_vessels)}
        elif algorithm == "rule_based":
            return {i: RuleBasedCOLREGsController(i) for i in range(n_vessels)}
        elif algorithm == "independent_ppo":
            return {i: IndependentPPOPolicy() for i in range(n_vessels)}
        elif algorithm == "maddpg":
            return {i: MADDPGPolicy(n_vessels=n_vessels) for i in range(n_vessels)}
        else:
            raise ValueError(f"Unknown baseline algorithm: {algorithm}")
