# Full Concept Document: MARLIN-Twin for Autonomous Maritime Traffic Coordination

> **Version**: 2.0 (Refined with Peer-Review Enhancements)
> **Date**: July 2026
> **Framework Name**: MARLIN-Twin (Maritime Adaptive Resilience Learning with Integrated Networked Twins)
> **Target**: Q1 Journal Publication (Ocean Engineering / IEEE TITS / RESS)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Data Classes & Domain Model](#3-data-classes--domain-model)
4. [API Design](#4-api-design)
5. [Development Plan](#5-development-plan)
6. [UI/UX Design](#6-uiux-design)
7. [Mathematical Specification](#7-mathematical-specification)
8. [Testing & Validation Strategy](#8-testing--validation-strategy)
9. [Appendices](#9-appendices)

---

<a name="1-executive-summary"></a>
## 1. Executive Summary

### Research Question

> *How can a digital twin framework with bandwidth-adaptive communication and graceful degradation enable safe, COLREGs-compliant multi-vessel coordination under realistic maritime communication disruptions?*

### Core Innovation

This project develops **MARLIN-Twin** (**M**aritime **A**daptive **R**esilience **L**earning with **I**ntegrated **N**etworked **Twin**s), a framework that:
1. **Digital Twin Fusion & JPDA**: Integrates real-time digital twin state estimation (Kalman filter with ITU-R M.1371 sensor noise calibration and Joint Probabilistic Data Association for track correlation during AIS denial).
2. **2-Stage Curriculum Learning**: Uses a 2-stage training paradigm (Stage 1: Base COLREGs navigation under full comms; Stage 2: Learned bandwidth adaptation & policy resilience under comms degradation) to mitigate MARL non-stationarity.
3. **Refined COLREGs Rule 17 Engine**: Embeds COLREGs rules (Rules 13-18) directly into reward shaping, explicitly preventing premature stand-on vessel deviations while rewarding emergency collision avoidance when give-way vessels fail to act.
4. **Bandwidth-Adaptive Communication**: Learns adaptive communication protocols (when, what, and to whom to communicate) over bandwidth-constrained and jammed channels.
5. **Heterogeneous MMG Dynamics**: Models heterogeneous vessel dynamics (cargo ships, tankers, USVs) with 3-DOF MMG maneuvering equations.
6. **Formalized Resilience Index**: Evaluates multi-agent coordination using the sub-linear **Coordination Resilience Index** $R_{\text{resilience}} = \int_0^1 \frac{J(\lambda)}{J(1.0)} d\lambda$.

### System Scope

| Component | Description |
|---|---|
| **Maritime Environment** | Open water, channel navigation, port approach scenarios |
| **Vessel Agents** | 2-25 heterogeneous vessels (cargo, USV, tanker, passenger) |
| **Digital Twin** | Kalman-filtered state estimator with AIS/radar fusion, ITU-R noise, and JPDA association |
| **Communication** | Bandwidth-limited channel with priority-based message routing & weather/jamming attenuation |
| **GNN Encoder** | Graph Attention Network (GAT) for encounter graph encoding (with GAT vs MLP ablation) |
| **MAPPO Training** | Multi-agent PPO with 2-stage curriculum and COLREGs-embedded rewards |
| **Resilience Testing** | Progressive communication degradation scenarios & Graceful Degradation metrics |

---

<a name="2-system-architecture"></a>
## 2. System Architecture

### 2.1 High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------+
|                         MARLIN-TWIN FRAMEWORK                                |
+-----------------------------------------------------------------------------+
|                                                                              |
|  +------------------+     +------------------+     +----------------------+ |
|  |   MARITIME       |     |   MAPPO          |     |   DIGITAL TWIN       | |
|  |   ENVIRONMENT    |<--->|   TRAINING       |<--->|   STATE ESTIMATOR    | |
|  |                  |     |   LOOP           |     |                      | |
|  +------------------+     +------------------+     +----------------------+ |
|           ^                        ^                        ^                |
|           |                        |                        |                |
|           v                        v                        v                |
|  +------------------+     +------------------+     +----------------------+ |
|  |   VESSEL         |     |   GNN +          |     |   BANDWIDTH-         | |
|  |   DYNAMICS       |     |   POLICY         |     |   ADAPTIVE COMMS     | |
|  |   (MMG Model)    |     |   NETWORKS       |     |   CHANNEL            | |
|  +------------------+     +------------------+     +----------------------+ |
|                                                                              |
|  +------------------+     +------------------+     +----------------------+ |
|  |   COLREGs        |     |   RESILIENCE     |     |   VISUALIZATION      | |
|  |   RULE ENGINE    |     |   TESTING        |     |   & EXPORT           | |
|  +------------------+     +------------------+     +----------------------+ |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### 2.2 Component Architecture

#### 2.2.1 Maritime Environment (MaritimeCoordEnv)

```
MaritimeCoordEnv
|
|-- VesselFleet (2-25 vessels)
|   |-- VesselAgent: {specification, dynamics, policy, state, route}
|   |-- VesselSpecification: {type, length, beam, draft, max_speed, safety_domain}
|   |-- VesselDynamics: {mass, inertia, damping, propeller, rudder}
|   |-- VesselState: {x, y, heading, speed, yaw_rate, surge, sway}
|   |-- Route: {waypoints, current_idx, remaining_distance}
|
|-- MaritimeScene
|   |-- Boundaries: {min_x, max_x, min_y, max_y}
|   |-- Obstacles: {land masses, restricted zones}
|   |-- EnvironmentCondition: {clear, fog, rain, wind, sea state}
|   |-- EncounterGraph: {nodes, edges, edge_features}
|
|-- CommunicationChannel
|   |-- Bandwidth: {total_bps, available_bps}
|   |-- Latency: {base, actual}
|   |-- PacketLoss: {base_rate, weather_factor}
|   |-- Jamming: {active, zone, intensity}
|   |-- MessageQueue: {priority_queue, scheduled_messages}
|   |-- MaritimeMessage: {sender, receiver, content, priority, size, latency}
|
|-- DigitalTwin
|   |-- StateEstimator: {kalman_filter, particle_filter}
|   |-- SensorFusion: {AIS_weight, radar_weight, dead_reckoning_weight}
|   |-- AISReadings: {vessel_id, position, heading, speed, confidence}
|   |-- RadarTracks: {track_id, position, velocity, associated_vessel}
|   |-- VesselStateEstimate: {state, covariance, confidence, method}
|   |-- FallbackMode: {kinematic_inference, rule_based, conservative}
|   |-- TrajectoryPredictions: {vessel_id, predicted_states, uncertainty}
|   |-- AnomalyDetector: {bad_data_threshold, compromised_sensors}
|
|-- COLREGsEngine
|   |-- EncounterClassifier: {head_on, crossing, overtaking, none}
|   |-- RuleChecker: {Rule 13, 14, 15, 16, 17, 18}
|   |-- ComplianceScore: {0.0 to 1.0}
|   |-- ViolationLog: {timestamp, vessels, rule, severity}
```

#### 2.2.2 Training Loop Architecture

```
MAPPOTrainingLoop
|
|-- Phase 1: Environment Reset
|   |-- Generate scenario (open water / channel / port / dense)
|   |-- Initialize vessel positions, routes, states
|   |-- Reset digital twin, communication channel
|   |-- Reset COLREGs engine
|
|-- Phase 2: Episode Execution
|   |-- For each timestep:
|       |-- Digital Twin Update:
|           |-- Receive AIS readings (if available)
|           |-- Receive radar tracks (if available)
|           |-- Run Kalman filter / particle filter
|           |-- Detect anomalies, identify compromised sensors
|           |-- Switch to fallback if confidence < threshold
|           |-- Predict trajectories for encounter detection
|       |
|       |-- Encounter Detection:
|           |-- Compute CPA, TCPA, DCPA for all vessel pairs
|           |-- Classify encounters (COLREGs Rules 13-18)
|           |-- Flag dangerous encounters (CPA < safety domain)
|       |
|       |-- Communication Phase:
|           |-- Each vessel decides whether to communicate
|           |-- Encode state/intent into message
|           |-- Allocate bandwidth (priority-based)
|           |-- Transmit messages (with latency, packet loss)
|           |-- Vessels receive and decode messages
|           |-- Update neighbor knowledge (or use DT fallback)
|       |
|       |-- Observation Construction:
|           |-- Own state (from DT estimate or direct sensing)
|           |-- Neighbor states (from comms or DT fallback)
|           |-- Encounter information (from COLREGs engine)
|           |-- Communication quality (link status)
|           |-- Environmental conditions
|           |-- Build encounter graph (for GNN encoding)
|       |
|       |-- Action Selection:
|           |-- GNN encodes encounter graph -> embedding
|           |-- Policy network (actor) -> action distribution
|           |-- Sample action: propeller RPM, rudder angle, communication targets
|           |-- Value network (critic) -> state value estimate
|       |
|       |-- Dynamics Integration:
|           |-- Apply actions to MMG model
|           |-- Integrate forward by dt (RK4)
|           |-- Update vessel states
|           |-- Check collisions (vessel-vessel, vessel-obstacle)
|       |
|       |-- Reward Computation:
|           |-- Safety reward: CPA-based, collision penalty
|           |-- COLREGs reward: compliance bonus, violation penalty
|           |-- Efficiency reward: progress toward waypoint, fuel consumption
|           |-- Communication reward: bandwidth utilization (not waste)
|           |-- Team reward: aggregated across vessels
|       |
|       |-- Logging:
|           |-- Store transition (obs, action, reward, next_obs)
|           |-- Log metrics (CPA, COLREGs compliance, comms utilization)
|           |-- Update episode statistics
|
|-- Phase 3: Policy Update (MAPPO)
|   |-- Compute advantages (GAE)
|   |-- Update actor (clipped surrogate objective)
|   |-- Update critic (value function MSE)
|   |-- Update communication policy (if learned)
|   |-- Log training metrics
|
|-- Phase 4: Evaluation
|   |-- Run evaluation episodes with deterministic policies
|   |-- Compute metrics: safety, COLREGs, efficiency, communication
|   |-- Save checkpoints
```

#### 2.2.3 Resilience Testing Architecture

```
ResilienceTestingPipeline
|
|-- Input: Trained MAPPO policies
|
|-- Step 1: Baseline Evaluation (comms_quality = 1.0)
|   |-- Run 100 episodes with perfect communication
|   |-- Record: safety_score, efficiency_score, COLREGs_score
|   |-- Store as baseline
|
|-- Step 2: Progressive Degradation
|   |-- For each degradation_level in [0.8, 0.6, 0.4, 0.2, 0.0]:
|       |-- Set communication channel quality
|       |-- Run 50 episodes
|       |-- Record all metrics
|       |-- Log fallback strategy activations
|
|-- Step 3: Jamming Scenarios
|   |-- Static jamming zone (fixed position)
   |-- Moving jamming source
|   |-- Swarm jamming (multiple sources)
|   |-- Record performance in and around jamming zones
|
|-- Step 4: Sensor Failure Scenarios
|   |-- AIS failure (single vessel)
|   |-- GPS denial zone
|   |-- Radar degradation (weather)
|   |-- Record digital twin confidence and fallback effectiveness
|
|-- Step 5: Graceful Degradation Analysis
|   |-- Plot performance vs. comms quality
|   |-- Check for sub-linear degradation (smooth curve)
|   |-- Identify cliff points (sudden performance drops)
|   |-- Compute resilience index (area under curve)
|   |-- Generate per-vessel graceful degradation reports
|
|-- Step 6: Baseline Comparison
|   |-- Run same tests with:
|       |-- Independent PPO (no communication)
|       |-- MADDPG (fixed communication)
|       |-- MAPPO with full communication (upper bound)
|       |-- Rule-based COLREGs controller
|   |-- Compare degradation curves
|   |-- Statistical significance testing
```

---

<a name="3-data-classes--domain-model"></a>
## 3. Data Classes & Domain Model

### 3.1 Core Data Classes

See separate file: `idea2_data_classes.py`

Key classes include:
- `VesselState`, `VesselDynamics`, `VesselSpecification`
- `Waypoint`, `Route`, `Encounter`
- `VesselObservation`, `VesselAction`, `MaritimeMessage`, `MaritimeCommunicationChannel`
- `VesselAgent`
- `AISReading`, `RadarTrack`, `VesselStateEstimate`, `MaritimeDigitalTwin`, `DigitalTwinConfig`
- `EncounterGraph`
- `MaritimeScene`, `SceneTransition`, `VoyageEpisode`
- `CoordinationResilienceMetrics`, `GracefulDegradationReport`
- `MaritimeExperimentConfig`, `MaritimeExperimentResult`

---

<a name="4-api-design"></a>
## 4. API Design

### 4.1 Protocols (Interfaces)

| Protocol | Purpose |
|---|---|
| `VesselDynamicsSolver` | MMG model integration, CPA computation, COLREGs compliance |
| `StateEstimator` | Digital twin state estimation, trajectory prediction, anomaly detection |
| `CommunicationProtocol` | Bandwidth allocation, message encoding/decoding, communication decisions |
| `Policy` | RL policy interface (act, evaluate, serialize) |
| `GraphEncoder` | GNN-based scene encoding |

### 4.2 Abstract Base Classes

| ABC | Responsibility |
|---|---|
| `BaseMaritimeEnvironment` | Maritime coordination environment (reset, step, render) |
| `BaseTrainer` | MARL training with resilience sweep evaluation |
| `BaseDigitalTwin` | Maritime digital twin with sensor fusion and fallback |

### 4.3 Main API Facade: `MarlinTwinAPI`

```python
# Core workflow
api = MarlinTwinAPI(config)
result = api.train_and_evaluate(n_episodes=1000)

# Test resilience
resilience = api.evaluate_resilience(degradation_levels=[1.0, 0.5, 0.0])

# Compare baselines
comparison = api.compare_baselines(["independent_ppo", "maddpg", "rule_based"])

# Visualize
api.plot_trajectories(result, output_path="trajectories.png")
api.plot_resilience_curve(resilience, output_path="resilience.png")
api.plot_communication_heatmap(result, output_path="communication.png")
api.plot_encounter_graph(scene, output_path="encounter_graph.png")
api.plot_colregs_compliance(result, output_path="colregs.png")

# Export
api.export_results("experiment_results/")
api.generate_paper_figures("./figures")

# Replay
api.replay_episode(episode=0, speed=1.0, save_video=True)
```

See separate file: `idea2_api_design.py` for full implementation.

---

<a name="5-development-plan"></a>
## 5. Development Plan

### 5.1 Project Structure

```
marlin_twin/
|-- marlin_twin/                     # Main package
|   |-- api.py                       # MarlinTwinAPI facade
|   |-- data_classes.py              # All dataclasses
|   |-- envs/                        # Environment implementations
|   |   |-- vessel_dynamics.py, maritime_scene.py, maritime_coord_env.py
|   |   |-- scenarios.py, colregs.py, encounters.py
|   |   |-- communication.py, digital_twin.py, sensors.py
|   |-- agents/                      # Agent implementations
|   |   |-- vessel_agent.py, policies.py, networks.py
|   |   |-- communication_layer.py, observation_builder.py, reward_shaping.py
|   |-- training/                    # Training infrastructure
|   |   |-- mappo.py, rollout_buffer.py, reward_normalizer.py, eval.py, curriculum.py
|   |-- baselines/                   # Baseline algorithms
|   |   |-- independent_ppo.py, maddpg.py, rule_based.py, factory.py
|   |-- visualization/             # UI/visualization
|   |   |-- dashboard.py, chart_display.py, plots.py, episode_replayer.py, export.py
|   |-- utils/                       # Utilities
|   |   |-- config.py, logging.py, seeding.py, metrics.py, geometry.py
|-- tests/                           # Test suite
|-- configs/                         # Experiment configurations
|-- scripts/                         # Utility scripts
|-- notebooks/                       # Jupyter notebooks
|-- docs/                            # Documentation
```

### 5.2 Phase-by-Phase Timeline (5-6 months)

| Phase | Weeks | Goal | Key Deliverable |
|---|---|---|---|
| **1. Vessel Dynamics & Environment** | 1-4 | Working MMG dynamics and basic scene | 2 vessels navigate with CPA detection |
| **2. COLREGs & Multi-Agent** | 5-8 | COLREGs-compliant multi-agent coordination | 5 vessels, >90% COLREGs compliance |
| **3. Communication & Digital Twin** | 9-12 | Bandwidth-adaptive comms and state estimation | DT < 50m error, graceful fallback |
| **4. GNN & Adaptive Communication** | 13-16 | GNN scene encoding and learned bandwidth | GNN outperforms MLP baseline |
| **5. Resilience & Graceful Degradation** | 17-20 | Demonstrate graceful degradation | Sub-linear degradation curve |
| **6. Experiments & Baselines** | 21-24 | Complete experimental comparison | All paper figures generated |
| **7. Paper Writing** | 25-30 | Submission-ready manuscript | Draft complete |

### 5.3 Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| MMG model too complex | Medium | High | Simplify to 3-DOF; validate against known solutions |
| COLREGs compliance hard to learn | High | High | Heavy reward shaping; curriculum from simple to complex |
| GNN training unstable | Medium | Medium | Start with MLP; careful initialization |
| Communication learning collapses | Medium | High | Curriculum: full comms -> gradual degradation |
| DT error accumulates | Medium | Medium | Conservative fallback; frequent AIS updates |

See separate file: `idea2_development_plan.md` for full details.

---

<a name="6-uiux-design"></a>
## 6. UI/UX Design

### 6.1 Design Principles

| Principle | Description |
|---|---|
| **Situational Awareness First** | Critical alerts are immediate and unmissable |
| **Spatial Cognition** | All vessel data is geo-referenced on nautical chart |
| **Temporal Context** | Trajectories and predictions are central |
| **Resilience Transparency** | Communication degradation effects immediately visible |
| **COLREGs Compliance** | Rule violations flagged with explicit rule references |

### 6.2 Key Screens

1. **Chart Display (Primary)**: Nautical chart with vessel positions, trajectories, CPA zones, communication links
2. **Dashboard**: Scene overview, active alerts, experiment status
3. **New Experiment (Wizard)**: 7-step guided setup with maritime-specific config
4. **Training Monitor**: Real-time metrics, coordination quality, COLREGs compliance
5. **Resilience Analysis**: Degradation curves, graceful degradation reports, per-vessel resilience
6. **Digital Twin Monitor**: Sensor fusion view, trajectory predictions, anomaly detection
7. **Communication Analysis**: Link quality map, bandwidth utilization, message priority analysis

### 6.3 Alert System

| Level | Trigger | Visual | Audio |
|---|---|---|---|
| **Emergency** | Collision imminent (< 100m CPA in < 30s) | Full-screen red overlay, pulsing icon | Continuous alarm |
| **Critical** | CPA < 200m within 60s, COLREGs violation | Red banner, flashing border | 3 beeps |
| **Warning** | Communication lost > 30s, DT confidence < 0.5 | Amber banner | 1 beep |
| **Caution** | Bandwidth saturation, minor COLREGs deviation | Yellow indicator | None |
| **Info** | Waypoint reached, scenario transition | Green toast | None |

See separate file: `idea2_ui_ux_design.md` for full specification.

---

<a name="7-mathematical-specification"></a>
## 7. Mathematical Specification

### 7.1 Vessel Dynamics (Simplified MMG Model)

**State vector**: `x = [x, y, psi, u, v, r]^T`

- `(x, y)`: Position (m)
- `psi`: Heading (rad)
- `u`: Surge velocity (m/s)
- `v`: Sway velocity (m/s)
- `r`: Yaw rate (rad/s)

**Equations of motion**:

```
(m - X_u_dot) * du/dt = X_u * u * |u| + X_prop * n * |n| + X_rudder * delta * u^2
(m - Y_v_dot) * dv/dt = Y_v * v * |v| + Y_rudder * delta * u^2 + Y_r * r * |r|
(I_z - N_r_dot) * dr/dt = N_r * r * |r| + N_rudder * delta * u^2 + N_v * v * |v|

dx/dt = u * cos(psi) - v * sin(psi)
dy/dt = u * sin(psi) + v * cos(psi)
dpsi/dt = r
```

Where:
- `n`: Propeller RPM
- `delta`: Rudder angle
- `X_u_dot, Y_v_dot, N_r_dot`: Added mass coefficients
- `X_u, Y_v, N_r`: Damping coefficients
- `X_prop, X_rudder, Y_rudder, N_rudder`: Control coefficients

### 7.2 Closest Point of Approach (CPA)

For vessels `i` and `j` with positions `p_i, p_j` and velocities `v_i, v_j`:

```
Relative position: r = p_j - p_i
Relative velocity: v = v_j - v_i

Time to CPA: TCPA = - (r . v) / (v . v)  [if > 0, otherwise CPA has passed]

CPA position: p_cpa_i = p_i + v_i * TCPA
              p_cpa_j = p_j + v_j * TCPA

CPA distance: DCPA = |p_cpa_j - p_cpa_i|
```

### 7.3 COLREGs Reward Shaping & Rule 17 Logic

```
R_total = w_safety * R_safety + w_colregs * R_colregs + w_efficiency * R_efficiency + w_comm * R_comm

R_safety = -exp(-CPA / lambda)  [penalty for close approaches]
         - M_collision * I_collision  [heavy penalty for collision]

R_colregs = +1.0 if compliant
          - M_violation * I_violation  [penalty for give-way vessel failing to alter course]
          - R_rule17  [refined Rule 17 penalty/reward logic]

Rule 17 Stand-on Logic:
- If vessel is Stand-on and TCPA > TCPA_emergency:
    R_rule17 = - M_premature_stand_on * |heading_change|  [penalize premature deviation]
- If vessel is Stand-on and TCPA <= TCPA_emergency and Give-Way failed to act:
    R_rule17 = + R_emergency_evasion if maneuver_increases_DCPA else - M_collision_risk

R_efficiency = -|v - v_desired| / v_max  [speed tracking]
             - |heading_error| / pi  [heading tracking]
             - c_fuel * |propeller_rpm|^3  [fuel consumption]

R_comm = -c_bandwidth * (bandwidth_used / bandwidth_total)  [discourage waste]
       + c_critical * I_critical_message_delivered  [reward critical comms]
```

### 7.4 Communication Bandwidth Model

```
Available bandwidth: B_avail = B_total * (1 - weather_degradation) * (1 - jamming_factor)

Message size: S = S_header + S_state * I_include_state + S_intent * I_include_intent

Priority queue: messages sorted by priority (CRITICAL > HIGH > MEDIUM > LOW)

Transmission: messages transmitted in priority order until B_avail exhausted

Latency: L = L_base + L_queue + L_weather + L_jamming

Packet loss: P = P_base + P_weather + P_jamming + P_distance
```

### 7.5 Digital Twin Kalman Filter & JPDA Data Association

**State**: `[x, y, psi, u, v, r]^T`

**Sensor Noise Calibration (ITU-R M.1371 / Marine Radar)**:
- $R_{\text{AIS}} = \text{diag}(\sigma_{x,\text{AIS}}^2, \sigma_{y,\text{AIS}}^2, \sigma_{\psi,\text{AIS}}^2, \sigma_{U,\text{AIS}}^2)$ where $\sigma_{p,\text{AIS}} \approx 5.0\,\text{m}$, $\sigma_{\psi,\text{AIS}} \approx 0.5^\circ$.
- $R_{\text{Radar}} = \text{diag}(\sigma_{r,\text{Radar}}^2, \sigma_{\theta,\text{Radar}}^2)$ where $\sigma_{r} \approx 15.0\,\text{m}$, $\sigma_{\theta} \approx 1.0^\circ$.

**Prediction**:
```
x_pred = F * x_prev + B * u_control
P_pred = F * P_prev * F^T + Q
```

**JPDA Track Association (During AIS Loss/Denial)**:
- Compute Mahalanobis distance validation matrix $\Omega$ between unassigned radar tracks and predicted vessel states:
  $d_M^2(z_k, \hat{x}_{i|k-1}) = (z_k - H_i \hat{x}_{i|k-1})^T S_{i,k}^{-1} (z_k - H_i \hat{x}_{i|k-1}) \le \gamma_{\text{gate}}$
- Calculate marginal association probabilities $\beta_{ij} = P(\text{track } j \to \text{vessel } i)$ via Joint Probabilistic Data Association (JPDA).

**Update**:
```
y_i = sum_j beta_ij * (z_j - H * x_pred_i)  [weighted innovation]
S_i = H * P_pred_i * H^T + R  [innovation covariance]
K_i = P_pred_i * H^T * S_i^{-1}  [Kalman gain]
x_est_i = x_pred_i + K_i * y_i
P_est_i = (I - K_i * H) * P_pred_i
```

**Anomaly Detection**: $y^T S^{-1} y > \chi^2_{\text{threshold}} \implies$ flag bad sensor data / spoofing.

**Fallback**: When AIS detection rate $< \tau_{\text{AIS}}$ or confidence $< \tau_{\text{conf}}$:
- Switch to Dead Reckoning: $x = x_{\text{prev}} + v \cdot \Delta t$
- Or Conservative Mode: reduce speed by 30%, maintain safe heading, broadcast warning intent.

### 7.6 Graph Neural Network Encoder & Ablation Architecture

**Encounter Graph**: $G = (V, E)$

- **Nodes**: $v_i = [x_i, y_i, \psi_i, u_i, v_i, r_i, \text{type}_i, \text{intent}_i]$
- **Edges**: $e_{ij} = [d_{ij}, v_{\text{rel},ij}, \text{CPA}_{ij}, \text{TCPA}_{ij}, \text{comm\_quality}_{ij}, \text{encounter\_type}_{ij}]$

**GAT (Graph Attention Network) Forward Pass**:
```
h_i^(0) = MLP_node(v_i)  [initial node embedding]

For each layer l in 1..L:
    alpha_ij^(l) = Softmax_j( LeakyReLU( a^T [ W h_i^(l) || W h_j^(l) || W_e e_ij ] ) )
    h_i^(l+1) = ELU( sum_{j in N(i)} alpha_ij^(l) W h_j^(l) )

h_i^final = MLP_readout(h_i^(L))  [vessel-centric scene embedding]
```

**Ablation Study Matrix**:
1. **GAT Policy (Primary)**: Graph Attention Network with dynamic multi-head attention.
2. **Mean-Pooling GNN**: Isotropic graph aggregation without attention weights.
3. **Flat MLP Baseline**: Fixed-size concatenation of nearest $K$ vessel states.

### 7.7 Formalized Coordination Resilience Index

The normalized performance under communication quality $\lambda \in [0.0, 1.0]$ is defined as:

$$R(\lambda) = \frac{J(\lambda)}{J(1.0)}$$

where $J(\lambda)$ represents the composite evaluation metric (safety score + COLREGs compliance + efficiency score) under communication quality $\lambda$.

**Coordination Resilience Index ($R_{\text{resilience}}$)**:

$$R_{\text{resilience}} = \int_{0}^{1} R(\lambda) \, d\lambda \approx \sum_{k=1}^{K} \frac{R(\lambda_k) + R(\lambda_{k-1})}{2} \Delta\lambda$$

**Graceful Degradation Criterion**:
- **Sub-linear degradation constraint**: $R(0.5) \ge 0.70$ (maintaining $\ge 70\%$ performance under $50\%$ bandwidth/channel quality).
- **Resilience Index Target**: $R_{\text{resilience}} \ge 0.75$.

---

<a name="8-testing--validation-strategy"></a>
## 8. Testing & Validation Strategy

### 8.1 Unit Tests (Target: 80-90% coverage)

| Module | Key Tests |
|---|---|
| `vessel_dynamics.py` | MMG integration accuracy, stability limits, turning circle |
| `colregs.py` | All encounter types, rule compliance, edge cases |
| `encounters.py` | CPA computation accuracy, TCPA, DCPA |
| `communication.py` | Bandwidth enforcement, priority queue, packet loss |
| `digital_twin.py` | Kalman filter convergence, bad data detection, fallback |
| `mappo.py` | Policy gradient correctness, value function learning |

### 8.2 Integration Tests

| Scenario | Pass Criteria |
|---|---|
| End-to-end voyage (2 vessels) | No collisions, all waypoints reached |
| Channel navigation (5 vessels) | COLREGs compliance > 90%, no collisions |
| Communication stress (50% bandwidth) | Graceful degradation, no collisions |
| Sensor failure (AIS loss) | DT fallback maintains < 100m error |
| Jamming scenario | All vessels navigate safely through |

### 8.3 Reproducibility Tests

- Seed consistency: Same seed -> identical trajectories
- Config serialization: Save config -> reload -> identical behavior
- Checkpoint resume: Pause at episode 100 -> resume -> identical episode 101

---

<a name="9-appendices"></a>
## 9. Appendices

### A. Target Journals

| Journal | Fit | Why |
|---|---|---|
| **Ocean Engineering** (Elsevier, Q1) | Excellent | Published Wang & Zhao 2025 MARL paper; maritime-focused |
| **Reliability Engineering & System Safety** (Elsevier, Q1) | Very Good | Resilience + safety angle |
| **IEEE Transactions on Intelligent Transportation Systems** | Good | Broader readership |
| **Applied Ocean Research** (Elsevier, Q1) | Good | Maritime dynamics focus |

### B. Python Stack

| Component | Purpose |
|---|---|
| PyTorch + PyTorch Geometric | GNN policy networks |
| Stable-Baselines3 / RLlib | MAPPO baseline |
| NetworkX | Maritime graph topology |
| NumPy/SciPy | MMG dynamics, Kalman filtering |
| Hydra | Configuration management (extends docl_config.py) |

### C. One-Line Pitch

> *"We introduce **MARLIN-Twin**, a digital twin multi-agent reinforcement learning framework with bandwidth-adaptive communication protocols and JPDA state estimation that maintains COLREGs-compliant coordination even when AIS and GPS sensors degrade — addressing an open challenge in maritime autonomous navigation literature."*

---

*Document generated July 2026. Designed for Q1 journal publication with rigorous methodology, defensible novelty claims, and reproducible Python-only simulation.*
