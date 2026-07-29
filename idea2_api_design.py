# ============================================================================
# FILE: marlin_twin/api_design.py
# ============================================================================

"""
API Design for MARLIN-Twin Maritime Framework

Composable, testable API surface for maritime multi-agent coordination
with digital twin, bandwidth-adaptive communication, and graceful degradation.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# PROTOCOLS (INTERFACES)
# =============================================================================

@runtime_checkable
class VesselDynamicsSolver(Protocol):
    """Protocol for vessel dynamics computation (MMG model)."""

    def step(
        self,
        state: "VesselState",
        action: "VesselAction",
        dt: float,
        environment: "EnvironmentCondition"
    ) -> "VesselState":
        """Integrate vessel dynamics forward by dt seconds."""
        ...

    def compute_cpa(
        self,
        state_i: "VesselState",
        state_j: "VesselState",
        prediction_horizon: float = 600.0
    ) -> tuple[float, float]:
        """Compute CPA distance and time for two vessels."""
        ...

    def check_colregs_compliance(
        self,
        own_state: "VesselState",
        own_action: "VesselAction",
        other_state: "VesselState",
        encounter_type: "EncounterType"
    ) -> bool:
        """Check if action complies with COLREGs for encounter type."""
        ...


@runtime_checkable
class StateEstimator(Protocol):
    """Protocol for digital twin state estimation."""

    def estimate(
        self,
        ais_readings: list["AISReading"],
        radar_tracks: list["RadarTrack"],
        last_estimates: dict[int, "VesselStateEstimate"]
    ) -> "MaritimeDigitalTwin":
        """Estimate full maritime scene from sensor data."""
        ...

    def predict_trajectory(
        self,
        vessel_id: int,
        current_estimate: "VesselStateEstimate",
        horizon: float = 300.0,
        step: float = 10.0
    ) -> list["VesselState"]:
        """Predict future trajectory."""
        ...

    def detect_sensor_anomaly(
        self,
        reading: "AISReading",
        estimate: "VesselStateEstimate"
    ) -> float:
        """Return anomaly score for a sensor reading."""
        ...


@runtime_checkable
class CommunicationProtocol(Protocol):
    """Protocol for bandwidth-adaptive maritime communication."""

    def allocate_bandwidth(
        self,
        messages: list["MaritimeMessage"],
        channel: "MaritimeCommunicationChannel"
    ) -> list["MaritimeMessage"]:
        """Select and schedule messages within bandwidth constraints."""
        ...

    def encode_message(
        self,
        sender_state: "VesselState",
        sender_intent: "Route",
        priority: "MessagePriority"
    ) -> "MaritimeMessage":
        """Encode vessel state and intent into message."""
        ...

    def decode_message(
        self,
        message: "MaritimeMessage",
        receiver_estimate: "VesselStateEstimate"
    ) -> tuple["VesselState", "Route"]:
        """Decode message into state and intent estimate."""
        ...

    def should_communicate(
        self,
        own_state: "VesselState",
        neighbor_states: dict[int, "VesselState"],
        encounters: list["Encounter"],
        channel_quality: float
    ) -> bool:
        """Decide whether communication is warranted."""
        ...


@runtime_checkable
class Policy(Protocol):
    """Protocol for RL policies."""

    def act(
        self,
        observation: NDArray,
        deterministic: bool = False
    ) -> NDArray:
        """Select action given observation."""
        ...

    def evaluate(
        self,
        observations: NDArray,
        actions: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
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

    def encode(
        self,
        encounter_graph: "EncounterGraph",
        vessel_id: int
    ) -> NDArray:
        """Encode scene graph into vessel-centric embedding."""
        ...

    def update_edge_features(
        self,
        graph: "EncounterGraph",
        communication_states: dict[tuple[int, int], "CommunicationStatus"]
    ) -> "EncounterGraph":
        """Update graph edge features with communication quality."""
        ...


# =============================================================================
# ABSTRACT BASE CLASSES
# =============================================================================

class BaseMaritimeEnvironment(ABC):
    """Abstract base for maritime coordination environment."""

    def __init__(self, config: "MaritimeExperimentConfig"):
        self.config = config
        self.scene: "MaritimeScene" = None
        self.time_step: int = 0
        self.dt: float = 1.0  # Simulation timestep (seconds)
        self._initialized = False

    @abstractmethod
    def reset(
        self,
        scenario_type: str | None = None,
        n_vessels: int | None = None,
        seed: int | None = None
    ) -> tuple[dict[int, "VesselObservation"], dict]:
        """Reset environment to initial maritime scene."""
        pass

    @abstractmethod
    def step(
        self,
        actions: dict[int, "VesselAction"]
    ) -> tuple[
        dict[int, "VesselObservation"],
        dict[int, float],
        float,   # team reward
        bool,    # done
        dict     # info
    ]:
        """Execute one environment step (1 second)."""
        pass

    @abstractmethod
    def get_scene(self) -> "MaritimeScene":
        """Return current maritime scene."""
        pass

    def render(self, mode: str = "human") -> NDArray | None:
        """Render current scene (optional)."""
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    def set_communication_degradation(
        self,
        level: float,  # 0.0 = full degradation, 1.0 = perfect
        jamming_zone: tuple[float, float, float] | None = None
    ) -> None:
        """Set communication degradation level for resilience testing."""
        pass


class BaseTrainer(ABC):
    """Abstract base for MARL training."""

    def __init__(self, config: "MaritimeExperimentConfig"):
        self.config = config
        self.policies: dict[int, Policy] = {}
        self.history: list[dict] = []

    @abstractmethod
    def train(
        self,
        env: BaseMaritimeEnvironment,
        n_episodes: int
    ) -> dict[int, Policy]:
        """Train policies on environment."""
        pass

    @abstractmethod
    def evaluate(
        self,
        env: BaseMaritimeEnvironment,
        policies: dict[int, Policy],
        n_episodes: int = 100,
        communication_degradation: float = 1.0
    ) -> dict[str, float]:
        """Evaluate policies under specified conditions."""
        pass

    def run_resilience_sweep(
        self,
        env: BaseMaritimeEnvironment,
        policies: dict[int, Policy],
        degradation_levels: list[float] | None = None
    ) -> "CoordinationResilienceMetrics":
        """Evaluate policies across communication degradation levels."""
        if degradation_levels is None:
            degradation_levels = self.config.test_degradation_levels

        metrics = CoordinationResilienceMetrics()
        metrics.degradation_levels = degradation_levels

        for level in degradation_levels:
            env.set_communication_degradation(level)
            results = self.evaluate(env, policies, n_episodes=50)
            metrics.safety_scores.append(results["safety_score"])
            metrics.efficiency_scores.append(results["efficiency_score"])

        # Reset to full communication
        env.set_communication_degradation(1.0)

        return metrics


class BaseDigitalTwin(ABC):
    """Abstract base for maritime digital twin."""

    def __init__(self, config: "DigitalTwinConfig"):
        self.config = config
        self.estimates: dict[int, "VesselStateEstimate"] = {}
        self.sensor_history: list["AISReading"] = []

    @abstractmethod
    def update(
        self,
        ais_readings: list["AISReading"],
        radar_tracks: list["RadarTrack"]
    ) -> "MaritimeDigitalTwin":
        """Update digital twin with new sensor data."""
        pass

    @abstractmethod
    def get_estimate(self, vessel_id: int) -> "VesselStateEstimate":
        """Get current estimate for a vessel."""
        pass

    @abstractmethod
    def get_fallback_estimate(
        self,
        vessel_id: int,
        last_known: "VesselState",
        dt: float
    ) -> "VesselState":
        """Generate fallback estimate when sensors fail."""
        pass

    def compute_encounter_predictions(
        self,
        prediction_horizon: float = 300.0
    ) -> list["Encounter"]:
        """Predict future encounters from estimated trajectories."""
        encounters = []
        vessel_ids = list(self.estimates.keys())

        for i, vid_i in enumerate(vessel_ids):
            est_i = self.estimates[vid_i]
            if not est_i.is_reliable():
                continue

            traj_i = self.predict_trajectory(vid_i, est_i, prediction_horizon)

            for vid_j in vessel_ids[i+1:]:
                est_j = self.estimates[vid_j]
                if not est_j.is_reliable():
                    continue

                traj_j = self.predict_trajectory(vid_j, est_j, prediction_horizon)

                # Find minimum CPA in predicted trajectories
                min_dist = float('inf')
                min_time = 0.0
                for t, (si, sj) in enumerate(zip(traj_i, traj_j)):
                    dist = np.sqrt((si.x - sj.x)**2 + (si.y - sj.y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        min_time = t * self.config.prediction_step

                if min_dist < 1000:  # Within 1km
                    encounter = Encounter(
                        vessel_i=vid_i,
                        vessel_j=vid_j,
                        encounter_type=EncounterType.NO_ENCOUNTER,  # Classify separately
                        colregs_rule=None,
                        cpa_distance=min_dist,
                        cpa_time=min_time,
                        tcpa=min_time,
                        dcpa=min_dist,
                        relative_bearing=0.0,
                        is_dangerous=min_dist < 500
                    )
                    encounters.append(encounter)

        return encounters

    def predict_trajectory(
        self,
        vessel_id: int,
        estimate: "VesselStateEstimate",
        horizon: float = 300.0,
        step: float = 10.0
    ) -> list["VesselState"]:
        """Default prediction: constant velocity model."""
        trajectory = []
        state = estimate.estimated_state
        n_steps = int(horizon / step)

        for _ in range(n_steps):
            state = VesselState(
                vessel_id=state.vessel_id,
                x=state.x + state.speed * np.sin(state.heading) * step,
                y=state.y + state.speed * np.cos(state.heading) * step,
                heading=state.heading,
                speed=state.speed,
            )
            trajectory.append(state)

        return trajectory


# =============================================================================
# CONCRETE API CLASSES
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

    Usage:
        api = MarlinTwinAPI(config)
        result = api.train_and_evaluate()

        # Test resilience
        resilience = api.evaluate_resilience(degradation_levels=[1.0, 0.5, 0.0])

        # Visualize
        api.plot_trajectories(result, output_path="trajectories.png")
        api.plot_resilience_curve(resilience, output_path="resilience.png")

        # Export
        api.export_results("experiment_results/")
    """

    def __init__(self, config: "MaritimeExperimentConfig" | None = None):
        self.config = config or MaritimeExperimentConfig()
        self.env: BaseMaritimeEnvironment | None = None
        self.trainer: BaseTrainer | None = None
        self.digital_twin: BaseDigitalTwin | None = None
        self.result: "MaritimeExperimentResult" | None = None
        self.policies: dict[int, Policy] = {}

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

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
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = MaritimeExperimentConfig(**config_dict)
        return self

    def save_config(self, path: str) -> None:
        """Save current configuration to file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.config.__dict__, f)

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------

    def create_environment(
        self,
        env_class: type[BaseMaritimeEnvironment] | None = None
    ) -> BaseMaritimeEnvironment:
        """Create and initialize the maritime environment."""
        if env_class is None:
            from .envs import MaritimeCoordEnv
            env_class = MaritimeCoordEnv

        self.env = env_class(self.config)
        return self.env

    def set_environment(self, env: BaseMaritimeEnvironment) -> "MarlinTwinAPI":
        """Set an existing environment."""
        self.env = env
        return self

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def set_trainer(self, trainer: BaseTrainer) -> "MarlinTwinAPI":
        """Set custom trainer implementation."""
        self.trainer = trainer
        return self

    def train_and_evaluate(
        self,
        n_episodes: int = 1000,
        eval_frequency: int = 25
    ) -> "MaritimeExperimentResult":
        """
        Train policies and evaluate throughout.

        Args:
            n_episodes: Total training episodes
            eval_frequency: Evaluate every N episodes

        Returns:
            Complete experiment results
        """
        if self.env is None:
            self.create_environment()

        if self.trainer is None:
            from .training import MAPPOTrainer
            self.trainer = MAPPOTrainer(self.config)

        print(f"Starting training for {n_episodes} episodes...")
        self.policies = self.trainer.train(self.env, n_episodes)

        # Final evaluation with full communication
        print("Evaluating with full communication...")
        full_comms_results = self.trainer.evaluate(
            self.env, self.policies, n_episodes=100, communication_degradation=1.0
        )

        # Resilience sweep
        print("Running resilience sweep...")
        resilience_metrics = self.trainer.run_resilience_sweep(
            self.env, self.policies
        )

        # Graceful degradation reports
        print("Generating graceful degradation reports...")
        degradation_reports = self._generate_degradation_reports()

        self.result = MaritimeExperimentResult(
            config=self.config,
            episodes=[],  # Populated by trainer
            final_policies=self.policies,
            resilience_metrics=resilience_metrics,
            graceful_degradation_reports=degradation_reports,
            training_rewards=[],  # Populated by trainer
            eval_rewards=[full_comms_results["average_reward"]],
            colregs_violation_rates=[full_comms_results["colregs_violation_rate"]],
            communication_utilization=[full_comms_results["communication_utilization"]],
            baseline_comparison={}
        )

        return self.result

    def _generate_degradation_reports(self) -> list["GracefulDegradationReport"]:
        """Generate per-vessel graceful degradation reports."""
        reports = []
        for vessel_id in range(self.config.n_vessels):
            # Evaluate at different degradation levels
            scores = {}
            for level in [1.0, 0.5, 0.0]:
                self.env.set_communication_degradation(level)
                results = self.trainer.evaluate(self.env, self.policies, n_episodes=20)
                scores[level] = results["safety_score"]

            report = GracefulDegradationReport(
                vessel_id=vessel_id,
                full_comms_score=scores[1.0],
                half_comms_score=scores[0.5],
                no_comms_score=scores[0.0],
                fallback_strategy=self.config.dt_fallback_mode,
                fallback_effectiveness=(scores[0.0] / max(scores[1.0], 0.01)),
                colregs_compliance_under_degradation=scores[0.5],
                safety_margin_maintained=scores[0.0] > 0.3 * scores[1.0]
            )
            reports.append(report)

        # Reset
        self.env.set_communication_degradation(1.0)
        return reports

    def evaluate_resilience(
        self,
        policies: dict[int, Policy] | None = None,
        degradation_levels: list[float] | None = None
    ) -> "CoordinationResilienceMetrics":
        """
        Evaluate coordination resilience under communication degradation.

        Args:
            policies: Policies to evaluate (None = use trained policies)
            degradation_levels: Comms quality levels to test

        Returns:
            Coordination resilience metrics
        """
        if self.env is None:
            self.create_environment()

        if policies is None:
            policies = self.policies

        if self.trainer is None:
            from .training import MAPPOTrainer
            self.trainer = MAPPOTrainer(self.config)

        return self.trainer.run_resilience_sweep(self.env, policies, degradation_levels)

    # -------------------------------------------------------------------------
    # Baseline Comparisons
    # -------------------------------------------------------------------------

    def compare_baselines(
        self,
        baseline_algorithms: list[str] = None
    ) -> dict[str, dict[str, float]]:
        """
        Compare trained policies against baseline algorithms.

        Args:
            baseline_algorithms: List of algorithm names to compare

        Returns:
            Dictionary: algorithm -> metric -> value
        """
        if baseline_algorithms is None:
            baseline_algorithms = ["independent_ppo", "maddpg", "rule_based"]

        results = {}

        # Evaluate trained policy
        trained_results = self.trainer.evaluate(self.env, self.policies, n_episodes=100)
        results["trained_mappo"] = trained_results

        # Evaluate baselines
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
        """Load or train baseline policies."""
        # Implementation depends on baseline library
        from .baselines import BaselineFactory
        factory = BaselineFactory(self.config)
        return factory.create(algorithm)

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_trajectories(
        self,
        result: "MaritimeExperimentResult" | None = None,
        episode_idx: int = -1,
        output_path: str | None = None
    ) -> NDArray:
        """Plot vessel trajectories for an episode."""
        if result is None:
            result = self.result

        from .visualization import MaritimeVisualizer
        viz = MaritimeVisualizer()
        return viz.plot_trajectories(result.episodes[episode_idx], output_path)

    def plot_resilience_curve(
        self,
        metrics: "CoordinationResilienceMetrics" | None = None,
        output_path: str | None = None
    ) -> NDArray:
        """Plot coordination resilience degradation curve."""
        if metrics is None:
            metrics = self.result.resilience_metrics if self.result else None

        if metrics is None:
            raise RuntimeError("No resilience metrics available.")

        from .visualization import MaritimeVisualizer
        viz = MaritimeVisualizer()
        return viz.plot_resilience_curve(metrics, output_path)

    def plot_communication_heatmap(
        self,
        result: "MaritimeExperimentResult" | None = None,
        output_path: str | None = None
    ) -> NDArray:
        """Plot communication link quality heatmap over time."""
        if result is None:
            result = self.result

        from .visualization import MaritimeVisualizer
        viz = MaritimeVisualizer()
        return viz.plot_communication_heatmap(result, output_path)

    def plot_encounter_graph(
        self,
        scene: "MaritimeScene" | None = None,
        output_path: str | None = None
    ) -> NDArray:
        """Plot encounter graph with vessel positions and edges."""
        if scene is None and self.env:
            scene = self.env.get_scene()

        from .visualization import MaritimeVisualizer
        viz = MaritimeVisualizer()
        return viz.plot_encounter_graph(scene, output_path)

    def plot_colregs_compliance(
        self,
        result: "MaritimeExperimentResult" | None = None,
        output_path: str | None = None
    ) -> NDArray:
        """Plot COLREGs compliance rates over time."""
        if result is None:
            result = self.result

        from .visualization import MaritimeVisualizer
        viz = MaritimeVisualizer()
        return viz.plot_colregs_compliance(result, output_path)

    def generate_paper_figures(
        self,
        output_dir: str = "./figures"
    ) -> list[str]:
        """Generate all figures for paper submission."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        figure_paths = []

        # Figure 1: Training curves
        path = os.path.join(output_dir, "fig1_training_curves.png")
        self._plot_training_curves(path)
        figure_paths.append(path)

        # Figure 2: Resilience curve
        path = os.path.join(output_dir, "fig2_resilience_curve.png")
        self.plot_resilience_curve(output_path=path)
        figure_paths.append(path)

        # Figure 3: Trajectory comparison
        path = os.path.join(output_dir, "fig3_trajectories.png")
        self.plot_trajectories(output_path=path)
        figure_paths.append(path)

        # Figure 4: Communication heatmap
        path = os.path.join(output_dir, "fig4_communication.png")
        self.plot_communication_heatmap(output_path=path)
        figure_paths.append(path)

        # Figure 5: Baseline comparison
        path = os.path.join(output_dir, "fig5_baseline_comparison.png")
        self._plot_baseline_comparison(path)
        figure_paths.append(path)

        # Figure 6: Graceful degradation
        path = os.path.join(output_dir, "fig6_graceful_degradation.png")
        self._plot_graceful_degradation(path)
        figure_paths.append(path)

        return figure_paths

    def _plot_training_curves(self, path: str) -> None:
        """Internal: plot training reward curves."""
        pass  # Implementation

    def _plot_baseline_comparison(self, path: str) -> None:
        """Internal: plot bar chart comparing baselines."""
        pass  # Implementation

    def _plot_graceful_degradation(self, path: str) -> None:
        """Internal: plot graceful degradation per vessel."""
        pass  # Implementation

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save_result(self, path: str) -> None:
        """Save experiment result to disk."""
        if self.result is None:
            raise RuntimeError("No results to save.")
        self.result.save(path)

    def load_result(self, path: str) -> "MaritimeExperimentResult":
        """Load experiment result from disk."""
        self.result = MaritimeExperimentResult.load(path)
        return self.result

    def export_results(self, output_dir: str) -> None:
        """Export all results to directory."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save result
        self.save_result(os.path.join(output_dir, "result.pkl"))

        # Save config
        self.save_config(os.path.join(output_dir, "config.yaml"))

        # Save policies
        for vid, policy in self.policies.items():
            policy_state = policy.get_state()
            import pickle
            with open(os.path.join(output_dir, f"policy_vessel_{vid}.pkl"), 'wb') as f:
                pickle.dump(policy_state, f)

        # Generate figures
        self.generate_paper_figures(os.path.join(output_dir, "figures"))

        # Generate summary
        summary = self.get_summary()
        import json
        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Get experiment summary statistics."""
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
        """Print formatted experiment summary."""
        summary = self.get_summary()
        print("=" * 60)
        print("MARLIN-TWIN MARITIME EXPERIMENT SUMMARY")
        print("=" * 60)
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        print("=" * 60)

    def replay_episode(
        self,
        episode: "VoyageEpisode" | int,
        speed: float = 1.0,
        save_video: bool = False,
        output_path: str | None = None
    ) -> None:
        """
        Replay an episode with visualization.

        Args:
            episode: Episode to replay (or index into result.episodes)
            speed: Playback speed multiplier
            save_video: Whether to save as video file
            output_path: Video output path
        """
        if isinstance(episode, int):
            if self.result is None:
                raise RuntimeError("No results available.")
            episode = self.result.episodes[episode]

        from .visualization import EpisodeReplayer
        replayer = EpisodeReplayer(speed=speed)
        replayer.play(episode, save_video=save_video, output_path=output_path)


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


def create_minimal_api(
    scenario_type: str = "channel",
    n_vessels: int = 5
) -> MarlinTwinAPI:
    """Create minimal API for quick testing."""
    config = MaritimeExperimentConfig(
        scenario_type=scenario_type,
        n_vessels=n_vessels,
        n_episodes=100,
        episode_length=200,
        adaptive_bandwidth=True,
        dt_enabled=True
    )
    api = MarlinTwinAPI(config)
    api.create_environment()
    return api


def load_pretrained_api(checkpoint_dir: str) -> MarlinTwinAPI:
    """Load API with pretrained policies."""
    config_path = f"{checkpoint_dir}/config.yaml"
    api = create_default_api(config_path)

    # Load policies
    import pickle
    import os
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("policy_vessel_") and filename.endswith(".pkl"):
            vessel_id = int(filename.split("_")[2].split(".")[0])
            with open(os.path.join(checkpoint_dir, filename), 'rb') as f:
                policy_state = pickle.load(f)
            # Restore policy (implementation depends on policy class)

    return api
