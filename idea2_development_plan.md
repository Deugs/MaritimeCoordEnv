# ============================================================================
# FILE: marlin_twin/development_plan.md
# ============================================================================

# Development Plan: MARLIN-Twin Maritime Framework

> **Version**: 2.0 (Refined with Peer-Review Enhancements)
> **Date**: July 2026
> **Duration**: 5-6 months
> **Team**: 1 researcher (with potential collaborator for UI)

---

## 1. Project Structure

```
marlin_twin/
|
|-- marlin_twin/                     # Main package
|   |-- __init__.py
|   |-- api.py                       # MarlinTwinAPI facade
|   |-- data_classes.py              # All dataclasses (see separate file)
|   |
|   |-- envs/                        # Environment implementations
|   |   |-- __init__.py
|   |   |-- base_env.py              # BaseMaritimeEnvironment ABC
|   |   |-- vessel_dynamics.py       # MMG model implementation
|   |   |-- maritime_scene.py        # Scene construction and management
|   |   |-- maritime_coord_env.py    # Main environment (MaritimeCoordEnv)
|   |   |-- scenarios.py             # Scenario generators (open water, channel, port)
|   |   |-- colregs.py               # COLREGs rule checking and compliance
|   |   |-- encounters.py            # CPA computation and encounter classification
|   |   |-- communication.py         # Bandwidth-adaptive comms channel
|   |   |-- digital_twin.py          # State estimator (Kalman, particle, JPDA)
|   |   |-- sensors.py               # AIS, radar, dead reckoning simulation
|   |
|   |-- agents/                      # Agent implementations
|   |   |-- __init__.py
|   |   |-- vessel_agent.py          # VesselAgent with policy and state
|   |   |-- policies.py              # Policy network architectures (GAT, Mean-Pooling, MLP)
|   |   |-- networks.py              # Actor-critic, GNN encoders
|   |   |-- communication_layer.py   # Message encoding/decoding, bandwidth allocation
|   |   |-- observation_builder.py   # Build observations from scene + DT + comms
|   |   |-- reward_shaping.py       # COLREGs-embedded reward function (with Rule 17 logic)
|   |
|   |-- training/                    # Training infrastructure
|   |   |-- __init__.py
|   |   |-- base_trainer.py          # BaseTrainer ABC
|   |   |-- mappo.py                 # MAPPO implementation
|   |   |-- curriculum.py            # 2-Stage Curriculum Trainer
|   |   |-- rollout_buffer.py        # Episode/transition storage
|   |   |-- reward_normalizer.py     # Reward scaling and normalization
|   |   |-- eval.py                  # Evaluation utilities
|   |
|   |-- baselines/                   # Baseline algorithms
|   |   |-- __init__.py
|   |   |-- independent_ppo.py       # Independent PPO (no communication)
|   |   |-- maddpg.py                # MADDPG baseline
|   |   |-- rule_based.py            # COLREGs rule-based controller
|   |   |-- factory.py               # BaselineFactory
|   |
|   |-- visualization/             # UI/visualization
|   |   |-- __init__.py
|   |   |-- dashboard.py             # Streamlit web dashboard
|   |   |-- chart_display.py         # Nautical chart with vessel overlays
|   |   |-- plots.py                 # Matplotlib/Plotly figure generation
|   |   |-- episode_replayer.py      # Time-scrubbable replay
|   |   |-- export.py                # PDF, GPX, MP4 export
|   |
|   |-- utils/                       # Utilities
|   |   |-- __init__.py
|   |   |-- config.py                # Configuration management (extends docl_config)
|   |   |-- logging.py               # Structured logging
|   |   |-- seeding.py               # Reproducibility
|   |   |-- metrics.py               # Metric computation (CPA, COLREGs, etc.)
|   |   |-- geometry.py              # Maritime geometry utilities
|   |
|-- tests/                           # Test suite
|   |-- test_dynamics.py
|   |-- test_env.py
|   |-- test_colregs.py
|   |-- test_communication.py
|   |-- test_digital_twin.py
|   |-- test_training.py
|   |-- test_api.py
|
|-- configs/                         # Experiment configurations
|   |-- open_water_2vessels.yaml
|   |-- channel_5vessels.yaml
|   |-- port_approach_15vessels.yaml
|   |-- dense_traffic_25vessels.yaml
|   |-- minimal_test.yaml
|
|-- scripts/                         # Utility scripts
|   |-- run_experiment.py
|   |-- evaluate_policy.py
|   |-- generate_figures.py
|   |-- reproduce_paper.py
|   |-- demo_dashboard.py
|
|-- notebooks/                       # Jupyter notebooks
|   |-- 01_environment_demo.ipynb
|   |-- 02_vessel_dynamics.ipynb
|   |-- 03_training_walkthrough.ipynb
|   |-- 04_resilience_analysis.ipynb
|   |-- 05_paper_figures.ipynb
|
|-- docs/                            # Documentation
|   |-- architecture.md
|   |-- api_reference.md
|   |-- colregs_reference.md
|   |-- tutorials/
|
|-- README.md
|-- setup.py
|-- requirements.txt
|-- pyproject.toml
```

---

## 2. Phase-by-Phase Development Plan

### Phase 1: Vessel Dynamics & Environment Core (Weeks 1-4)
**Goal**: Working vessel dynamics and basic maritime environment

| Week | Task | Deliverable | Risk |
|---|---|---|---|
| 1.1 | Set up project structure, dependencies, CI/CD | Repo with passing tests | Low |
| 1.2 | Implement MMG vessel dynamics model | `vessel_dynamics.py` with RK4 integration | Medium |
| 1.3 | Implement basic maritime scene (2 vessels, open water) | `maritime_scene.py` + `maritime_coord_env.py` v0.1 | Low |
| 1.4 | Implement CPA computation and encounter classification | `encounters.py` with CPA, TCPA, DCPA | Low |

**Milestone**: Two vessels can navigate in open water with realistic dynamics and CPA detection.

**Dependencies**: PyTorch, NumPy, SciPy, pytest

---

### Phase 2: COLREGs & Multi-Agent (Weeks 5-8)
**Goal**: COLREGs-compliant multi-agent coordination with Rule 17 stand-on logic

| Week | Task | Deliverable | Risk |
|---|---|---|---|
| 2.1 | Implement COLREGs rule checking (Rules 13-18) | `colregs.py` with all encounter types | Medium |
| 2.2 | Extend to multi-agent (5-10 vessels) | Multi-agent env v0.2 | Medium |
| 2.3 | Implement COLREGs-embedded reward shaping with Rule 17 logic | `reward_shaping.py` (Rule 17 stand-on vs give-way) | Medium |
| 2.4 | Implement MAPPO Stage-1 training (Full comms baseline navigation) | `mappo.py` & `curriculum.py` v0.1 | High |

**Milestone**: 5 vessels navigate channel with COLREGs compliance > 90% under Stage-1 MAPPO.

**Key Decisions**:
- MMG model fidelity? **Decision: Simplified 3-DOF** (surge, sway, yaw) for tractability
- Encounter detection range? **Decision: 3 nautical miles** (standard maritime practice)
- COLREGs reward weight? **Decision: 2x safety weight** (violations heavily penalized)
- Rule 17 Stand-on behavior? **Decision: Explicit penalty for premature deviation; emergency reward if Give-Way fails to act**

---

### Phase 3: Communication & Digital Twin (Weeks 9-12)
**Goal**: Bandwidth-adaptive communication, ITU-R noise calibration, and JPDA state estimation

| Week | Task | Deliverable | Risk |
|---|---|---|---|
| 3.1 | Implement bandwidth-limited communication channel | `communication.py` with priority queue | Medium |
| 3.2 | Implement message encoding/decoding and bandwidth allocation | `communication_layer.py` | Medium |
| 3.3 | Implement Kalman filter state estimator with ITU-R M.1371 noise calibration | `digital_twin.py` (Kalman) | Medium |
| 3.4 | Implement sensor fusion (AIS + radar + dead reckoning) & JPDA track correlation | `sensors.py` + `digital_twin.py` (JPDA) | Medium |

**Milestone**: Digital twin maintains < 50m position error under normal conditions, JPDA maintains track correlation during AIS dropouts.

**Key Decisions**:
- Message content? **Decision: Position, velocity, heading, intent (waypoint)**
- Bandwidth allocation strategy? **Decision: Priority-based with critical messages guaranteed**
- Sensor Data Association? **Decision: JPDA algorithm for radar-to-vessel track association during AIS dropouts**

---

### Phase 4: GNN & Adaptive Communication (Weeks 13-16)
**Goal**: GNN scene encoding (GAT vs MLP ablation) and Stage-2 Curriculum training

| Week | Task | Deliverable | Risk |
|---|---|---|---|
| 4.1 | Implement encounter graph construction | `encounters.py` (graph generation) | Low |
| 4.2 | Implement GNN encoder (GAT) and ablation variants (Mean-Pooling, MLP) | `networks.py` & `policies.py` | High |
| 4.3 | Integrate GNN into MAPPO policy | Policy with GNN encoder | High |
| 4.4 | Implement Stage-2 Curriculum training (Learned bandwidth & policy fine-tuning under comms loss) | `curriculum.py` (Stage 2 training) | High |

**Milestone**: Stage-2 MAPPO with GAT outperforms Mean-Pooling and MLP baselines on coordination tasks.

**Key Decisions**:
- GNN architecture? **Decision: Graph Attention Network (GAT)** for interpretable attention weights
- Edge features? **Decision: Distance, relative velocity, CPA, comm link quality**
- Training paradigm? **Decision: 2-Stage Curriculum** (Stage 1: Base movement; Stage 2: Adaptive comms & policy fine-tuning)

---

### Phase 5: Resilience & Graceful Degradation (Weeks 17-20)
**Goal**: Demonstrate sub-linear graceful degradation and compute Resilience Index

| Week | Task | Deliverable | Risk |
|---|---|---|---|
| 5.1 | Implement communication degradation scenarios | `scenarios.py` (jamming, weather, AIS spoofing) | Low |
| 5.2 | Implement formalized Resilience Index $R_{\text{resilience}} = \int_0^1 \frac{J(\lambda)}{J(1.0)} d\lambda$ | `metrics.py` (resilience index) | Medium |
| 5.3 | Implement fallback strategies (kinematic, rule-based, conservative) | `digital_twin.py` (fallback modes) | Medium |
| 5.4 | Run resilience sweep and generate degradation curves | Resilience analysis notebook | Medium |

**Milestone**: System shows sub-linear graceful degradation ($R(0.5) \ge 0.70$) under progressive comms failure.

---

### Phase 6: Experiments & Baselines (Weeks 21-24)
**Goal**: Complete experimental comparison with all baselines and ablation models

| Week | Task | Deliverable | Risk |
|---|---|---|---|
| 6.1 | Implement all baselines (Independent PPO, MADDPG, Rule-based, GNN Ablation variants) | `baselines/` complete | Low |
| 6.2 | Run full experimental suite (all scenarios, all algorithms, ablation matrix) | Result datasets | Medium |
| 6.3 | Statistical validation (t-tests, confidence intervals) | `metrics.py` expanded | Low |
| 6.4 | Generate paper figures | `generate_figures.py` | Low |

**Milestone**: All paper figures generated, GAT superiority and sub-linear degradation confirmed.

**Milestone**: All paper figures generated, statistical significance confirmed for all claims.

---

### Phase 7: Paper Writing (Weeks 25-30)
**Goal**: Submission-ready manuscript

| Week | Task | Deliverable | Risk |
|---|---|---|---|
| 7.1 | Draft introduction + related work | Sections 1-2 | Low |
| 7.2 | Draft methodology (dynamics, COLREGs, comms, DT, GNN, MARL) | Sections 3-4 | Medium |
| 7.3 | Draft experiments + results | Sections 5-6 | Low |
| 7.4 | Draft conclusions + reproducibility statement | Section 7 | Low |
| 7.5 | Internal review + revision | Revised draft | Medium |
| 7.6 | External feedback + final polish | Submission draft | Medium |

**Target**: Ocean Engineering, IEEE TITS, or Reliability Engineering & System Safety

---

## 3. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| MMG model too complex / unstable | Medium | High | Simplify to 3-DOF; validate against known solutions |
| COLREGs compliance hard to learn | High | High | Heavy reward shaping; curriculum from simple to complex encounters |
| GNN training unstable | Medium | Medium | Start with MLP; add GNN as ablation; careful initialization |
| Communication learning collapses | Medium | High | Curriculum: start with full comms, gradually degrade |
| Digital twin error accumulates | Medium | Medium | Conservative fallback; frequent AIS updates; error bounds |
| Compute limitations for 25+ vessels | Medium | Medium | Use 5-10 vessels for training; 25 for final validation only |
| Baseline performance too strong | Low | High | Ensure baselines are properly tuned; use strong rule-based baseline |

---

## 4. Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.10+ | Core implementation |
| **ML Framework** | PyTorch | 2.0+ | Neural networks, GNNs |
| **GNN** | PyTorch Geometric | 2.3+ | Graph neural networks |
| **RL** | Stable-Baselines3 / RLlib | 2.0+ | MAPPO baseline |
| **Vessel Dynamics** | NumPy/SciPy | - | Custom MMG solver |
| **Visualization** | Matplotlib, Plotly | - | Static + interactive plots |
| **Dashboard** | Streamlit | 1.25+ | Web UI (optional) |
| **Config** | Hydra | 1.3+ | Configuration management |
| **Testing** | pytest | 7.0+ | Unit + integration tests |
| **Logging** | loguru | 0.7+ | Structured logging |
| **Serialization** | pickle, YAML, JSON | - | Model + result persistence |

---

## 5. Testing Strategy

### 5.1 Unit Tests

| Module | Test Coverage Target | Key Tests |
|---|---|---|
| `vessel_dynamics.py` | 90% | MMG integration accuracy, stability limits, turning circle |
| `colregs.py` | 90% | All encounter types, rule compliance, edge cases |
| `encounters.py` | 85% | CPA computation accuracy, TCPA, DCPA, crossing detection |
| `communication.py` | 85% | Bandwidth enforcement, priority queue, packet loss |
| `digital_twin.py` | 85% | Kalman filter convergence, bad data detection, fallback |
| `mappo.py` | 80% | Policy gradient correctness, value function learning |

### 5.2 Integration Tests

| Scenario | Description | Pass Criteria |
|---|---|---|
| End-to-end voyage | 2 vessels, open water, 500 steps | No collisions, all waypoints reached |
| Channel navigation | 5 vessels, channel, 1000 steps | COLREGs compliance > 90%, no collisions |
| Communication stress | 5 vessels, 50% bandwidth, 500 steps | Graceful degradation, no collisions |
| Sensor failure | 1 vessel loses AIS, 300 steps | DT fallback maintains < 100m error |
| Jamming scenario | GPS denial zone, 5 vessels, 500 steps | All vessels navigate safely through |

### 5.3 Reproducibility Tests

| Test | Description |
|---|---|
| Seed consistency | Same seed -> identical trajectories (deterministic mode) |
| Config serialization | Save config -> reload -> identical behavior |
| Checkpoint resume | Pause at episode 100 -> resume -> identical episode 101 |
| Cross-platform | Runs on Linux (primary) and macOS (development) |

---

## 6. Milestone Checkpoints

| Checkpoint | Date | Criteria | Go/No-Go |
|---|---|---|---|
| **M1: Dynamics** | Week 4 | 2 vessels navigate with realistic dynamics | Must pass |
| **M2: COLREGs** | Week 8 | 5 vessels, >90% COLREGs compliance | Must pass |
| **M3: Digital Twin** | Week 12 | < 50m DT error, graceful fallback | Must pass |
| **M4: GNN + Comms** | Week 16 | GNN outperforms MLP, learned bandwidth allocation | Should pass |
| **M5: Resilience** | Week 20 | Graceful degradation demonstrated | Must pass |
| **M6: Experiments** | Week 24 | All figures generated, stats confirmed | Must pass |
| **M7: Submission** | Week 30 | Paper draft complete | Target |

---

## 7. Resource Requirements

### Compute

| Resource | Specification | Usage |
|---|---|---|
| **Development machine** | 16-core CPU, 64GB RAM, RTX 4080 | Daily development, small experiments |
| **Training cluster** | 4x A100 GPUs (cloud) | Large-scale experiments, hyperparameter search |
| **Storage** | 2TB SSD | Model checkpoints, episode data, trajectories |

### Estimated Compute Budget

| Phase | GPU Hours | CPU Hours | Storage (GB) |
|---|---|---|---|
| 1-2 (Foundation) | 50 | 200 | 10 |
| 3 (Digital Twin) | 100 | 300 | 20 |
| 4 (GNN + Comms) | 200 | 500 | 50 |
| 5 (Resilience) | 150 | 400 | 40 |
| 6 (Experiments) | 300 | 800 | 100 |
| **Total** | **800** | **2200** | **220** |

---

## 8. Collaboration Plan

| Role | Responsibility | Time Commitment |
|---|---|---|
| **Primary Researcher** (You) | Architecture, dynamics, MARL, digital twin, paper | Full-time |
| **Advisor/Supervisor** | Direction, paper feedback, viva preparation | 2 hrs/week |
| **Optional: Co-author** | UI/dashboard, additional baselines, COLREGs validation | 10 hrs/week |
| **Optional: Maritime Expert** | COLREGs correctness, scenario realism, safety assessment | Ad-hoc |

---

## 9. Deliverables

### Code
1. **GitHub repository** with full source code
2. **Reproducibility package** (Docker image + config files)
3. **Benchmark environment** (standalone, pip-installable)
4. **Streamlit dashboard** (optional, for demonstration)

### Data
5. **Training logs** (all episodes, all metrics)
6. **Trajectory datasets** (GPX format for external validation)
7. **Resilience curves** (all degradation levels)

### Paper
8. **Main manuscript** (8,000-10,000 words)
9. **Supplementary material** (additional experiments, COLREGs details)
10. **Response to reviewers** (prepared during revision)

### Presentation
11. **Conference presentation** (15-min talk)
12. **Poster** (for workshops/conferences)
13. **Demo video** (2-min system walkthrough with chart display)
