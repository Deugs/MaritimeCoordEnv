"""Dataclasses and enums for vessel state, scenes, communication, and experiment results."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray

# =============================================================================
# ENUMERATIONS
# =============================================================================


class VesselType(Enum):
    """Types of maritime vessels."""

    CARGO = auto()
    CONTAINER = auto()
    TANKER = auto()
    PASSENGER = auto()
    USV = auto()  # Unmanned Surface Vehicle
    FERRY = auto()
    FISHING = auto()


class COLREGsRule(Enum):
    """COLREGs rules for encounter classification."""

    RULE_13_OVERTAKING = auto()
    RULE_14_HEAD_ON = auto()
    RULE_15_CROSSING = auto()
    RULE_16_GIVE_WAY = auto()
    RULE_17_STAND_ON = auto()
    RULE_18_RESPONSIBILITIES = auto()


class EncounterType(Enum):
    """Encounter classification for maritime coordination."""

    HEAD_ON = auto()
    CROSSING_GIVE_WAY = auto()
    CROSSING_STAND_ON = auto()
    OVERTAKING = auto()
    OVERTAKEN = auto()
    NO_ENCOUNTER = auto()


class MessagePriority(Enum):
    """Communication message priority levels."""

    CRITICAL = 0  # Collision imminent
    HIGH = 1  # Action required
    MEDIUM = 2  # Information sharing
    LOW = 3  # Routine updates


class CommunicationStatus(Enum):
    """Status of communication link."""

    ACTIVE = auto()
    DEGRADED = auto()
    LOST = auto()
    JAMMED = auto()


class NavigationMode(Enum):
    """Navigation control mode."""

    AUTONOMOUS = auto()
    REMOTE_CONTROL = auto()
    RULE_BASED = auto()
    EMERGENCY = auto()


class EnvironmentCondition(Enum):
    """Environmental conditions affecting navigation."""

    CLEAR = auto()
    FOG = auto()
    RAIN = auto()
    HIGH_WIND = auto()
    HIGH_SEA = auto()
    ICE = auto()


class ExplanationType(Enum):
    """Types of causal explanations."""

    INTERVENTION = auto()
    COUNTERFACTUAL = auto()
    PATH_ANALYSIS = auto()
    COMMUNICATION_BOTTLENECK = auto()


# =============================================================================
# VESSEL DYNAMICS DOMAIN
# =============================================================================


@dataclass(frozen=True)
class VesselState:
    """Physical state of a vessel."""

    vessel_id: int
    x: float  # Easting (m)
    y: float  # Northing (m)
    heading: float  # Radians (0 = North)
    speed: float  # m/s
    yaw_rate: float = 0.0  # rad/s
    surge_velocity: float = 0.0  # m/s (body frame)
    sway_velocity: float = 0.0  # m/s (body frame)

    def position(self) -> NDArray:
        return np.array([self.x, self.y])

    def velocity_vector(self) -> NDArray:
        return np.array([self.speed * np.sin(self.heading), self.speed * np.cos(self.heading)])

    def copy_with(self, **kwargs) -> "VesselState":
        return VesselState(**{**self.__dict__, **kwargs})


@dataclass
class VesselDynamics:
    """MMG model parameters for vessel dynamics."""

    vessel_id: int
    vessel_type: VesselType

    # Mass and inertia
    mass: float  # kg
    moment_of_inertia: float  # kg*m^2

    # Hull hydrodynamics
    X_u_dot: float = -50000.0  # Added mass surge
    Y_v_dot: float = -100000.0  # Added mass sway
    N_r_dot: float = -500000.0  # Added mass yaw

    # Damping coefficients
    X_u: float = -1000.0  # Surge damping
    Y_v: float = -5000.0  # Sway damping
    N_r: float = -20000.0  # Yaw damping

    # Propeller
    propeller_diameter: float = 4.0  # m
    max_rpm: float = 150.0

    # Rudder
    rudder_area: float = 20.0  # m^2
    max_rudder_angle: float = np.pi / 6  # 30 degrees

    def compute_derivatives(
        self, state: VesselState, propeller_rpm: float, rudder_angle: float
    ) -> Tuple[float, float, float, float, float]:
        """Compute state derivatives (dx, dy, dheading, du, dr)."""
        u = state.surge_velocity
        v = state.sway_velocity
        r = state.yaw_rate

        # Surge equation (MMG added mass & propeller thrust)
        effective_mass = max(self.mass - self.X_u_dot, 1.0)
        n_rps = (propeller_rpm * self.max_rpm) / 60.0
        thrust = (n_rps**2) * (self.propeller_diameter**4) * 0.05
        drag = self.X_u * u * abs(u)
        du = (thrust + drag) / effective_mass

        # Sway equation
        Y = self.Y_v_dot * v + self.Y_v * v * abs(v) + rudder_angle * self.rudder_area * u * u * 0.5
        Y / max(self.mass, 1.0)

        # Yaw equation
        N = self.N_r_dot * r + self.N_r * r * abs(r) + rudder_angle * self.rudder_area * u * u * 0.3
        dr = N / max(self.moment_of_inertia, 1.0)

        # Kinematics (Nautical Convention: 0=North/+Y, pi/2=East/+X)
        dx = u * np.sin(state.heading) + v * np.cos(state.heading)
        dy = u * np.cos(state.heading) - v * np.sin(state.heading)
        dheading = r

        return dx, dy, dheading, du, dr


@dataclass(frozen=True)
class VesselSpecification:
    """Static vessel specifications."""

    vessel_id: int
    name: str
    vessel_type: VesselType
    length: float  # m
    beam: float  # m
    draft: float  # m
    max_speed: float  # m/s
    min_speed: float = 0.0
    turning_circle: float = 0.0  # m

    # Safety
    safety_domain_radius: float = 500.0  # m (CPA threshold)
    critical_domain_radius: float = 100.0  # m (emergency)

    def max_turning_rate(self) -> float:
        if self.turning_circle > 0:
            return self.max_speed / self.turning_circle
        return np.pi / 30  # Default: 6 degrees/s


# =============================================================================
# NAVIGATION DOMAIN
# =============================================================================


@dataclass(frozen=True)
class Waypoint:
    """Navigation waypoint."""

    waypoint_id: int
    x: float
    y: float
    speed: float  # Desired speed at waypoint
    radius: float = 50.0  # Acceptance radius (m)

    def distance_to(self, state: VesselState) -> float:
        return np.sqrt((self.x - state.x) ** 2 + (self.y - state.y) ** 2)


@dataclass
class Route:
    """Planned route for a vessel."""

    vessel_id: int
    waypoints: List[Waypoint]
    current_waypoint_idx: int = 0

    def current_waypoint(self) -> Optional[Waypoint]:
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None

    def advance(self) -> bool:
        self.current_waypoint_idx += 1
        return self.current_waypoint_idx < len(self.waypoints)

    def remaining_distance(self, state: VesselState) -> float:
        dist = 0.0
        current = state
        for wp in self.waypoints[self.current_waypoint_idx :]:
            dist += np.sqrt((wp.x - current.x) ** 2 + (wp.y - current.y) ** 2)
            current = VesselState(
                vessel_id=current.vessel_id, x=wp.x, y=wp.y, heading=current.heading, speed=wp.speed
            )
        return dist


@dataclass
class Encounter:
    """Detected encounter between two vessels."""

    vessel_i: int
    vessel_j: int
    encounter_type: EncounterType
    colregs_rule: Optional[COLREGsRule]
    cpa_distance: float  # Closest Point of Approach (m)
    cpa_time: float  # Time to CPA (s)
    tcpa: float  # Time to CPA (alternative)
    dcpa: float  # Distance at CPA
    relative_bearing: float  # Radians
    is_dangerous: bool = False

    def risk_level(self) -> float:
        """Normalized risk level 0-1."""
        if self.cpa_distance < 100:
            return 1.0
        elif self.cpa_distance < 500:
            return 1.0 - (self.cpa_distance - 100) / 400
        return 0.0


# =============================================================================
# AGENT DOMAIN
# =============================================================================


@dataclass(frozen=True)
class VesselObservation:
    """Local observation for a vessel agent."""

    vessel_id: int
    own_state: VesselState
    own_route: Route

    # Perceived neighbors (from AIS / communication)
    neighbor_states: Dict[int, VesselState]
    neighbor_intents: Dict[int, Route]

    # Environmental
    environment: EnvironmentCondition
    visibility_range: float
    wind_speed: float
    wind_direction: float
    current_speed: float
    current_direction: float

    # Communication
    comm_link_quality: Dict[int, float]
    last_message_timestamp: Dict[int, float]

    # Digital twin
    estimated_neighbor_states: Dict[int, VesselState]
    estimation_confidence: Dict[int, float]

    # COLREGs
    active_encounters: List[Encounter]
    colregs_compliance_score: float


@dataclass(frozen=True)
class VesselAction:
    """Action taken by a vessel agent."""

    vessel_id: int
    propeller_rpm: float  # Normalized -1.0 to 1.0
    rudder_angle: float  # Radians

    # Communication action
    message_targets: List[int]
    message_content: Optional[NDArray] = None
    message_priority: MessagePriority = MessagePriority.MEDIUM

    # Emergency
    emergency_stop: bool = False
    sound_signal: Optional[str] = None


@dataclass(frozen=True)
class MaritimeMessage:
    """Inter-vessel communication message."""

    sender_id: int
    receiver_id: int
    content: NDArray
    priority: MessagePriority
    timestamp: float
    size_bits: int

    # Metadata
    latency: float = 0.0
    delivered: bool = True
    delivery_confirmed: bool = False

    def is_critical(self) -> bool:
        return self.priority == MessagePriority.CRITICAL


@dataclass
class MaritimeCommunicationChannel:
    """Bandwidth-limited maritime communication channel."""

    channel_id: str
    bandwidth_bps: float
    base_latency: float
    packet_loss_rate: float

    # Dynamic conditions
    jamming_active: bool = False
    jamming_zone: Optional[Tuple[float, float, float]] = None
    weather_degradation: float = 0.0

    # State
    message_queue: List[MaritimeMessage] = field(default_factory=list)
    active_links: Dict[Tuple[int, int], CommunicationStatus] = field(default_factory=dict)

    def available_bandwidth(self, time_window: float = 1.0) -> float:
        used = sum(m.size_bits for m in self.message_queue if m.delivered)
        base = self.bandwidth_bps * (1 - self.weather_degradation)
        return max(0.0, base * time_window - used)

    def can_transmit(self, message: MaritimeMessage, time_window: float = 1.0) -> bool:
        if self.jamming_active and self._in_jamming_zone(message):
            return False
        return message.size_bits <= self.available_bandwidth(time_window)

    def _in_jamming_zone(self, message: MaritimeMessage) -> bool:
        if self.jamming_zone is None:
            return False
        return False

    def get_link_status(self, vessel_i: int, vessel_j: int) -> CommunicationStatus:
        return self.active_links.get((vessel_i, vessel_j), CommunicationStatus.ACTIVE)


@dataclass
class VesselAgent:
    """Autonomous vessel agent."""

    vessel_id: int
    specification: VesselSpecification
    dynamics: VesselDynamics

    # Policy
    policy_network: Optional[Callable] = None
    value_network: Optional[Callable] = None

    # State
    current_state: Optional[VesselState] = None
    current_route: Optional[Route] = None
    navigation_mode: NavigationMode = NavigationMode.AUTONOMOUS

    # Communication
    communication_buffer: List[MaritimeMessage] = field(default_factory=list)
    pending_messages: List[MaritimeMessage] = field(default_factory=list)

    # Tracking
    last_observation: Optional[VesselObservation] = None
    last_action: Optional[VesselAction] = None
    cumulative_reward: float = 0.0

    # COLREGs tracking
    colregs_violations: int = 0
    near_misses: int = 0

    # Resilience tracking
    graceful_degradation_score: float = 1.0


# =============================================================================
# DIGITAL TWIN DOMAIN
# =============================================================================


@dataclass
class AISReading:
    """Automatic Identification System reading."""

    vessel_id: int
    timestamp: float
    reported_position: Tuple[float, float]
    reported_heading: float
    reported_speed: float
    confidence: float = 1.0
    is_suspect: bool = False

    def discrepancy(self, estimated: VesselState) -> float:
        pos_diff = np.sqrt(
            (self.reported_position[0] - estimated.x) ** 2
            + (self.reported_position[1] - estimated.y) ** 2
        )
        heading_diff = abs(self.reported_heading - estimated.heading)
        speed_diff = abs(self.reported_speed - estimated.speed)
        return pos_diff + heading_diff * 100 + speed_diff * 10


@dataclass
class RadarTrack:
    """Radar/ARPA track."""

    track_id: int
    timestamp: float
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    confidence: float = 1.0
    associated_vessel: Optional[int] = None


@dataclass
class VesselStateEstimate:
    """Digital twin state estimate for a vessel."""

    vessel_id: int
    estimated_state: VesselState
    covariance: NDArray
    estimation_method: str

    # Source fusion
    ais_contribution: float = 0.0
    radar_contribution: float = 0.0
    dead_reckoning_contribution: float = 0.0

    # Quality
    position_confidence: float = 1.0
    velocity_confidence: float = 1.0
    overall_confidence: float = 1.0

    def is_reliable(self, threshold: float = 0.5) -> bool:
        return self.overall_confidence >= threshold


@dataclass
class MaritimeDigitalTwin:
    """Digital twin for maritime traffic scene."""

    scene_id: str
    timestamp: float

    # Estimates for all vessels
    vessel_estimates: Dict[int, VesselStateEstimate]

    # Raw sensor data
    ais_readings: List[AISReading]
    radar_tracks: List[RadarTrack]

    # Scene understanding
    detected_encounters: List[Encounter]
    predicted_trajectories: Dict[int, List[VesselState]]
    collision_risks: Dict[Tuple[int, int], float]

    # Health
    sensor_health: Dict[str, float]
    communication_health: Dict[str, float]

    def get_estimate(self, vessel_id: int) -> Optional[VesselStateEstimate]:
        return self.vessel_estimates.get(vessel_id)

    def get_fallback_estimate(
        self, vessel_id: int, last_known: VesselState, dt: float
    ) -> VesselState:
        """Dead reckoning fallback when sensors fail."""
        return VesselState(
            vessel_id=vessel_id,
            x=last_known.x + last_known.speed * np.sin(last_known.heading) * dt,
            y=last_known.y + last_known.speed * np.cos(last_known.heading) * dt,
            heading=last_known.heading,
            speed=last_known.speed,
        )


@dataclass
class DigitalTwinConfig:
    """Configuration for maritime digital twin."""

    estimator_type: str = "kalman"
    prediction_horizon: float = 300.0
    prediction_step: float = 10.0

    # Sensor fusion weights
    ais_weight: float = 0.5
    radar_weight: float = 0.3
    dead_reckoning_weight: float = 0.2

    # Fallback
    fallback_timeout: float = 30.0
    conservative_mode: bool = True


# =============================================================================
# ENCOUNTER GRAPH DOMAIN
# =============================================================================


@dataclass
class EncounterGraph:
    """Dynamic graph representation of maritime scene."""

    timestamp: float

    # Nodes: vessels
    node_features: NDArray
    vessel_ids: List[int]

    # Edges: encounters + communication links
    edge_index: NDArray
    edge_features: NDArray
    edge_types: List[str]

    def to_pyg_data(self):
        """Convert to PyTorch Geometric Data object."""
        try:
            from torch_geometric.data import Data
            import torch

            return Data(
                x=torch.tensor(self.node_features, dtype=torch.float32),
                edge_index=torch.tensor(self.edge_index, dtype=torch.long),
                edge_attr=torch.tensor(self.edge_features, dtype=torch.float32),
            )
        except ImportError:
            raise ImportError("PyTorch Geometric is required for to_pyg_data()")

    def get_neighbor_mask(self, vessel_id: int) -> NDArray:
        """Get boolean mask of neighbors for a vessel."""
        idx = self.vessel_ids.index(vessel_id)
        mask = np.zeros(len(self.vessel_ids), dtype=bool)
        for i in range(self.edge_index.shape[1]):
            if self.edge_index[0, i] == idx:
                mask[self.edge_index[1, i]] = True
            if self.edge_index[1, i] == idx:
                mask[self.edge_index[0, i]] = True
        return mask


# =============================================================================
# ENVIRONMENT DOMAIN
# =============================================================================


@dataclass
class MaritimeScene:
    """Complete maritime traffic scene."""

    scene_id: str
    timestamp: float

    # Vessels
    vessels: Dict[int, VesselAgent]

    # Infrastructure
    communication_channel: MaritimeCommunicationChannel
    digital_twin: MaritimeDigitalTwin

    # Environment
    boundaries: Tuple[float, float, float, float]
    environment_condition: EnvironmentCondition

    # Static obstacles (land, restricted zones)
    obstacles: List[Tuple[float, float, float]] = field(default_factory=list)
    restricted_zones: List[Tuple[float, float, float]] = field(default_factory=list)

    def check_collision(self, vessel_id: int) -> bool:
        """Check if vessel has collided with any other vessel or obstacle."""
        vessel = self.vessels[vessel_id]
        state = vessel.current_state
        if state is None:
            return False

        # Check other vessels
        for other_id, other in self.vessels.items():
            if other_id == vessel_id:
                continue
            other_state = other.current_state
            if other_state is None:
                continue
            dist = np.sqrt((state.x - other_state.x) ** 2 + (state.y - other_state.y) ** 2)
            if dist < (vessel.specification.length + other.specification.length) / 2:
                return True

        # Check obstacles
        for ox, oy, oradius in self.obstacles:
            dist = np.sqrt((state.x - ox) ** 2 + (state.y - oy) ** 2)
            if dist < oradius:
                return True

        return False


@dataclass
class SceneTransition:
    """Single environment transition."""

    scene: MaritimeScene
    observations: Dict[int, VesselObservation]
    actions: Dict[int, VesselAction]
    messages: List[MaritimeMessage]
    next_scene: MaritimeScene
    rewards: Dict[int, float]
    team_reward: float
    done: bool
    info: Dict


@dataclass
class VoyageEpisode:
    """Complete voyage episode."""

    episode_id: str
    transitions: List[SceneTransition]
    total_reward: float
    length: int

    # Metrics
    min_cpa: float = float("inf")
    colregs_violations: int = 0
    fuel_consumed: float = 0.0
    time_to_destination: float = 0.0
    communication_success_rate: float = 0.0

    def compute_metrics(self) -> Dict[str, float]:
        return {
            "min_cpa": self.min_cpa,
            "colregs_violations": self.colregs_violations,
            "fuel_consumed": self.fuel_consumed,
            "time_to_destination": self.time_to_destination,
            "communication_success_rate": self.communication_success_rate,
            "average_reward": self.total_reward / max(self.length, 1),
        }


# =============================================================================
# RESILIENCE & METRICS DOMAIN
# =============================================================================


@dataclass
class CoordinationResilienceMetrics:
    """Metrics for coordination resilience under communication degradation."""

    # Baseline (perfect communication)
    baseline_safety_score: float = 0.0
    baseline_efficiency_score: float = 0.0

    # Under degradation
    degraded_safety_score: float = 0.0
    degraded_efficiency_score: float = 0.0

    # Resilience curve
    degradation_levels: List[float] = field(default_factory=list)
    safety_scores: List[float] = field(default_factory=list)
    efficiency_scores: List[float] = field(default_factory=list)

    def resilience_index(self) -> float:
        """Area under resilience curve (should be > 0.5 for graceful degradation)."""
        if not self.degradation_levels or len(self.degradation_levels) < 2:
            return 0.0
        area = 0.0
        for i in range(len(self.degradation_levels) - 1):
            dx = abs(self.degradation_levels[i + 1] - self.degradation_levels[i])
            avg_y = (self.safety_scores[i] + self.safety_scores[i + 1]) / 2
            area += dx * avg_y
        span = abs(self.degradation_levels[-1] - self.degradation_levels[0])
        return area / max(span, 1e-6)

    def is_graceful(self, threshold: float = 0.7) -> bool:
        """Check if degradation is graceful (sub-linear)."""
        if len(self.degradation_levels) < 2:
            return True
        mid_idx = len(self.degradation_levels) // 2
        return self.safety_scores[mid_idx] > threshold * self.baseline_safety_score


@dataclass
class GracefulDegradationReport:
    """Report on graceful degradation performance."""

    vessel_id: int
    full_comms_score: float
    half_comms_score: float
    no_comms_score: float

    fallback_strategy: str  # 'kinematic_inference', 'rule_based', 'conservative'
    fallback_effectiveness: float  # 0.0 to 1.0

    colregs_compliance_under_degradation: float
    safety_margin_maintained: bool


# =============================================================================
# CONFIGURATION & RESULTS
# =============================================================================


@dataclass
class MaritimeExperimentConfig:
    """Top-level experiment configuration."""

    # Scene
    n_vessels: int = 10
    scenario_type: str = "channel"
    boundaries: Tuple[float, float, float, float] = (-5000, 5000, -5000, 5000)

    # Vessels
    vessel_types: List[VesselType] = field(
        default_factory=lambda: [VesselType.CARGO, VesselType.USV]
    )
    heterogeneous: bool = True

    # Training
    algorithm: str = "MAPPO"
    n_episodes: int = 1000
    episode_length: int = 500
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Communication
    bandwidth_bps: float = 9600.0
    base_latency: float = 0.5
    packet_loss_base: float = 0.05
    adaptive_bandwidth: bool = True
    priority_queue: bool = True

    # Digital Twin
    dt_enabled: bool = True
    dt_estimator: str = "kalman"
    dt_fallback_mode: str = "kinematic_inference"

    # Resilience testing
    test_degradation_levels: List[float] = field(
        default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    )

    # COLREGs
    colregs_reward_weight: float = 1.0
    safety_reward_weight: float = 2.0
    efficiency_reward_weight: float = 1.0

    # Logging
    log_dir: str = "./logs"
    checkpoint_frequency: int = 50
    eval_frequency: int = 25


@dataclass
class MaritimeExperimentResult:
    """Complete experiment results."""

    config: MaritimeExperimentConfig
    episodes: List[VoyageEpisode] = field(default_factory=list)

    # Policies
    final_policies: Dict[int, Callable] = field(default_factory=dict)

    # Metrics
    resilience_metrics: CoordinationResilienceMetrics = field(
        default_factory=CoordinationResilienceMetrics
    )
    graceful_degradation_reports: List[GracefulDegradationReport] = field(default_factory=list)

    # Training curves
    training_rewards: List[float] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    colregs_violation_rates: List[float] = field(default_factory=list)
    communication_utilization: List[float] = field(default_factory=list)

    # Comparison
    baseline_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def save(self, path: str) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "MaritimeExperimentResult":
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    def get_summary(self) -> Dict[str, float]:
        return {
            "n_episodes": len(self.episodes),
            "final_avg_reward": (
                float(np.mean(self.eval_rewards[-10:])) if self.eval_rewards else 0.0
            ),
            "resilience_index": self.resilience_metrics.resilience_index(),
            "is_graceful": self.resilience_metrics.is_graceful(),
            "avg_colregs_violations": (
                float(np.mean([ep.colregs_violations for ep in self.episodes]))
                if self.episodes
                else 0.0
            ),
            "avg_min_cpa": (
                float(np.mean([ep.min_cpa for ep in self.episodes if ep.min_cpa < float("inf")]))
                if self.episodes
                else 0.0
            ),
        }
