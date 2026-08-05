"""
API Design for MARLIN-Twin Maritime Framework

Composable, testable API surface for maritime multi-agent coordination
with digital twin, bandwidth-adaptive communication, and graceful degradation.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, TYPE_CHECKING
import json
import os
import pickle
import yaml
from loguru import logger
from numpy.typing import NDArray

from marlin_twin.data_classes import (
    VesselState,
    VesselAction,
    EnvironmentCondition,
    EncounterType,
    AISReading,
    RadarTrack,
    VesselStateEstimate,
    MaritimeDigitalTwin,
    DigitalTwinConfig,
    MaritimeMessage,
    MessagePriority,
    Route,
    MaritimeCommunicationChannel,
    Encounter,
    EncounterGraph,
    CoordinationResilienceMetrics,
    GracefulDegradationReport,
    MaritimeExperimentConfig,
    MaritimeExperimentResult,
    CommunicationStatus,
)
from marlin_twin.envs.base_env import BaseMaritimeEnvironment

if TYPE_CHECKING:
    pass


# =============================================================================
# PROTOCOLS (INTERFACES)
# =============================================================================


@runtime_checkable
class VesselDynamicsSolver(Protocol):
    """Protocol for vessel dynamics computation (MMG model)."""

    def step(
        self, state: VesselState, action: VesselAction, dt: float, environment: EnvironmentCondition
    ) -> VesselState:
        """Integrate vessel dynamics forward by dt seconds."""
        ...

    def compute_cpa(
        self, state_i: VesselState, state_j: VesselState, prediction_horizon: float = 600.0
    ) -> tuple[float, float]:
        """Compute CPA distance and time for two vessels."""
        ...

    def check_colregs_compliance(
        self,
        own_state: VesselState,
        own_action: VesselAction,
        other_state: VesselState,
        encounter_type: EncounterType,
    ) -> bool:
        """Check if action complies with COLREGs for encounter type."""
        ...


@runtime_checkable
class StateEstimator(Protocol):
    """Protocol for digital twin state estimation."""

    def estimate(
        self,
        ais_readings: list[AISReading],
        radar_tracks: list[RadarTrack],
        last_estimates: dict[int, VesselStateEstimate],
    ) -> MaritimeDigitalTwin:
        """Estimate full maritime scene from sensor data."""
        ...

    def predict_trajectory(
        self,
        vessel_id: int,
        current_estimate: VesselStateEstimate,
        horizon: float = 300.0,
        step: float = 10.0,
    ) -> list[VesselState]:
        """Predict future trajectory."""
        ...

    def detect_sensor_anomaly(self, reading: AISReading, estimate: VesselStateEstimate) -> float:
        """Return anomaly score for a sensor reading."""
        ...


@runtime_checkable
class CommunicationProtocol(Protocol):
    """Protocol for bandwidth-adaptive maritime communication."""

    def allocate_bandwidth(
        self, messages: list[MaritimeMessage], channel: MaritimeCommunicationChannel
    ) -> list[MaritimeMessage]:
        """Select and schedule messages within bandwidth constraints."""
        ...

    def encode_message(
        self, sender_state: VesselState, sender_intent: Route, priority: MessagePriority
    ) -> MaritimeMessage:
        """Encode vessel state and intent into message."""
        ...

    def decode_message(
        self, message: MaritimeMessage, receiver_estimate: VesselStateEstimate
    ) -> tuple[VesselState, Route]:
        """Decode message into state and intent estimate."""
        ...

    def should_communicate(
        self,
        own_state: VesselState,
        neighbor_states: dict[int, VesselState],
        encounters: list[Encounter],
        channel_quality: float,
    ) -> bool:
        """Decide whether communication is warranted."""
        ...


@runtime_checkable
class Policy(Protocol):
    """Protocol for RL policies."""

    def act(self, observation: NDArray, deterministic: bool = False) -> NDArray:
        """Select action given observation."""
        ...

    def evaluate(self, observations: NDArray, actions: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Evaluate actions: returns (values, log_probs, entropy)."""
        ...

    def get_state(self) -> dict:
        """Return serializable policy state."""
        ...

    def set_state(self, state: dict) -> None:
        """Restore policy from serialized state."""
        ...


@runtime_checkable
class GraphEncoder(Protocol):
    """Protocol for GNN-based scene encoding."""

    def encode(self, encounter_graph: EncounterGraph, vessel_id: int) -> NDArray:
        """Encode scene graph into vessel-centric embedding."""
        ...

    def update_edge_features(
        self,
        graph: EncounterGraph,
        communication_states: dict[tuple[int, int], CommunicationStatus],
    ) -> EncounterGraph:
        """Update graph edge features with communication quality."""
        ...


# =============================================================================
# ABSTRACT BASE CLASSES
# =============================================================================


class BaseTrainer(ABC):
    """Abstract base for MARL training."""

    def __init__(self, config: MaritimeExperimentConfig):
        self.config = config
        self.policies: dict[int, Policy] = {}
        self.history: list[dict] = []

    @abstractmethod
    def train(self, env: "BaseMaritimeEnvironment", n_episodes: int) -> dict[int, Policy]:
        """Train policies on environment."""

    @abstractmethod
    def evaluate(
        self,
        env: "BaseMaritimeEnvironment",
        policies: dict[int, Policy],
        n_episodes: int = 100,
        communication_degradation: float = 1.0,
    ) -> dict[str, float]:
        """Evaluate policies under specified conditions."""

    def run_resilience_sweep(
        self,
        env: "BaseMaritimeEnvironment",
        policies: dict[int, Policy],
        degradation_levels: list[float] | None = None,
    ) -> CoordinationResilienceMetrics:
        """Evaluate policies across communication degradation levels."""
        if degradation_levels is None:
            degradation_levels = self.config.test_degradation_levels

        metrics = CoordinationResilienceMetrics()
        metrics.degradation_levels = degradation_levels

        for level in degradation_levels:
            env.set_communication_degradation(level)
            results = self.evaluate(env, policies, n_episodes=50)
            metrics.safety_scores.append(results.get("safety_score", 0.0))
            metrics.efficiency_scores.append(results.get("efficiency_score", 0.0))

        env.set_communication_degradation(1.0)
        return metrics


class BaseDigitalTwin(ABC):
    """Abstract base for maritime digital twin."""

    def __init__(self, config: DigitalTwinConfig):
        self.config = config
        self.estimates: dict[int, VesselStateEstimate] = {}
        self.sensor_history: list[AISReading] = []

    @abstractmethod
    def update(
        self, ais_readings: list[AISReading], radar_tracks: list[RadarTrack]
    ) -> MaritimeDigitalTwin:
        """Update digital twin with new sensor data."""

    @abstractmethod
    def get_estimate(self, vessel_id: int) -> VesselStateEstimate:
        """Get current estimate for a vessel."""

    @abstractmethod
    def get_fallback_estimate(
        self, vessel_id: int, last_known: VesselState, dt: float
    ) -> VesselState:
        """Generate fallback estimate when sensors fail."""


# =============================================================================
# CONCRETE API CLASS
# =============================================================================


class MarlinTwinAPI:
    """
    Main API facade for the MARLIN-Twin Maritime framework.

    Provides a unified interface for:
    - Environment creation and configuration
    - Training with MAPPO and 2-Stage Curriculum
    - Digital twin integration (Kalman filter, ITU-R noise, JPDA track association)
    - Bandwidth-adaptive communication protocol management
    - Resilience evaluation and Coordination Resilience Index calculation
    - Result persistence and visualization
    """

    def __init__(self, config: MaritimeExperimentConfig | None = None):
        self.config = config or MaritimeExperimentConfig()
        self.env: BaseMaritimeEnvironment | None = None
        self.trainer: BaseTrainer | None = None
        self.digital_twin: BaseDigitalTwin | None = None
        self.result: MaritimeExperimentResult | None = None
        self.policies: dict[int, Policy] = {}

    def configure(self, **kwargs) -> "MarlinTwinAPI":
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
        return self

    def load_config(self, path: str) -> "MarlinTwinAPI":
        """Load configuration from YAML/JSON file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        self.config = MaritimeExperimentConfig(**config_dict)
        return self

    def save_config(self, path: str) -> None:
        """Save current configuration to file."""
        with open(path, "w") as f:
            yaml.dump(self.config.__dict__, f)

    def create_environment(
        self, env_class: type[BaseMaritimeEnvironment] | None = None
    ) -> BaseMaritimeEnvironment:
        """Create and initialize the maritime environment."""
        if env_class is None:
            from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv

            env_class = MaritimeCoordEnv

        self.env = env_class(self.config)
        return self.env

    def set_environment(self, env: BaseMaritimeEnvironment) -> "MarlinTwinAPI":
        """Set an existing environment."""
        self.env = env
        return self

    def set_trainer(self, trainer: BaseTrainer) -> "MarlinTwinAPI":
        """Set custom trainer implementation."""
        self.trainer = trainer
        return self

    def train_and_evaluate(
        self, n_episodes: int = 1000, eval_frequency: int = 25
    ) -> MaritimeExperimentResult:
        """Train policies and evaluate throughout."""
        if self.env is None:
            self.create_environment()

        if self.trainer is None:
            from marlin_twin.training.mappo import MAPPOTrainer

            self.trainer = MAPPOTrainer(self.config)

        logger.info(f"Starting training for {n_episodes} episodes...")
        self.policies = self.trainer.train(self.env, n_episodes)

        logger.info("Evaluating with full communication...")
        full_comms_results = self.trainer.evaluate(
            self.env, self.policies, n_episodes=100, communication_degradation=1.0
        )

        logger.info("Running resilience sweep...")
        resilience_metrics = self.trainer.run_resilience_sweep(self.env, self.policies)

        degradation_reports = self._generate_degradation_reports()

        self.result = MaritimeExperimentResult(
            config=self.config,
            episodes=[],
            final_policies=self.policies,
            resilience_metrics=resilience_metrics,
            graceful_degradation_reports=degradation_reports,
            training_rewards=[],
            eval_rewards=[full_comms_results.get("average_reward", 0.0)],
            colregs_violation_rates=[full_comms_results.get("colregs_violation_rate", 0.0)],
            communication_utilization=[full_comms_results.get("communication_utilization", 0.0)],
            baseline_comparison={},
        )

        return self.result

    def _generate_degradation_reports(self) -> list[GracefulDegradationReport]:
        """Generate per-vessel graceful degradation reports."""
        reports = []
        for vessel_id in range(self.config.n_vessels):
            scores = {}
            for level in [1.0, 0.5, 0.0]:
                self.env.set_communication_degradation(level)
                results = self.trainer.evaluate(self.env, self.policies, n_episodes=20)
                scores[level] = results.get("safety_score", 0.5)

            report = GracefulDegradationReport(
                vessel_id=vessel_id,
                full_comms_score=scores[1.0],
                half_comms_score=scores[0.5],
                no_comms_score=scores[0.0],
                fallback_strategy=self.config.dt_fallback_mode,
                fallback_effectiveness=(scores[0.0] / max(scores[1.0], 0.01)),
                colregs_compliance_under_degradation=scores[0.5],
                safety_margin_maintained=scores[0.0] > 0.3 * scores[1.0],
            )
            reports.append(report)

        self.env.set_communication_degradation(1.0)
        return reports

    def evaluate_resilience(
        self,
        policies: dict[int, Policy] | None = None,
        degradation_levels: list[float] | None = None,
    ) -> CoordinationResilienceMetrics:
        """Evaluate coordination resilience under communication degradation."""
        if self.env is None:
            self.create_environment()

        if policies is None:
            policies = self.policies

        if self.trainer is None:
            from marlin_twin.training.mappo import MAPPOTrainer

            self.trainer = MAPPOTrainer(self.config)

        return self.trainer.run_resilience_sweep(self.env, policies, degradation_levels)

    def compare_baselines(
        self, baseline_algorithms: list[str] | None = None
    ) -> dict[str, dict[str, float]]:
        """Compare trained policies against baseline algorithms."""
        if baseline_algorithms is None:
            baseline_algorithms = ["independent_ppo", "maddpg", "rule_based"]

        results = {}
        if self.trainer and self.env:
            trained_results = self.trainer.evaluate(self.env, self.policies, n_episodes=100)
            results["trained_mappo"] = trained_results

            for algo in baseline_algorithms:
                baseline_policies = self._load_baseline(algo)
                baseline_results = self.trainer.evaluate(
                    self.env, baseline_policies, n_episodes=100
                )
                results[algo] = baseline_results

        if self.result:
            self.result.baseline_comparison = results

        return results

    def _load_baseline(self, algorithm: str) -> dict[int, Policy]:
        from marlin_twin.baselines import BaselineFactory

        factory = BaselineFactory(self.config)
        return factory.create(algorithm)

    def save_result(self, path: str) -> None:
        if self.result is None:
            raise RuntimeError("No results to save.")
        self.result.save(path)

    def load_result(self, path: str) -> MaritimeExperimentResult:
        self.result = MaritimeExperimentResult.load(path)
        return self.result

    def export_results(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        if self.result:
            self.save_result(os.path.join(output_dir, "result.pkl"))

        self.save_config(os.path.join(output_dir, "config.yaml"))

        for vid, policy in self.policies.items():
            policy_state = policy.get_state()
            with open(os.path.join(output_dir, f"policy_vessel_{vid}.pkl"), "wb") as f:
                pickle.dump(policy_state, f)

        summary = self.get_summary()
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    def get_summary(self) -> dict:
        if self.result is None:
            return {"status": "No experiment run"}

        summary = self.result.get_summary()
        summary["config"] = {
            "n_vessels": self.config.n_vessels,
            "scenario_type": self.config.scenario_type,
            "algorithm": self.config.algorithm,
            "adaptive_bandwidth": self.config.adaptive_bandwidth,
            "digital_twin": self.config.dt_enabled,
        }
        return summary

    def print_summary(self) -> None:
        summary = self.get_summary()
        logger.info("=" * 60)
        logger.info("MARLIN-TWIN MARITIME EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        for key, value in summary.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("=" * 60)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_default_api(config_path: str | None = None) -> MarlinTwinAPI:
    """Create API with sensible defaults."""
    api = MarlinTwinAPI()
    if config_path:
        api.load_config(config_path)
    api.create_environment()
    return api


def create_minimal_api(scenario_type: str = "channel", n_vessels: int = 5) -> MarlinTwinAPI:
    """Create minimal API for quick testing."""
    config = MaritimeExperimentConfig(
        scenario_type=scenario_type,
        n_vessels=n_vessels,
        n_episodes=100,
        episode_length=200,
        adaptive_bandwidth=True,
        dt_enabled=True,
    )
    api = MarlinTwinAPI(config)
    api.create_environment()
    return api
