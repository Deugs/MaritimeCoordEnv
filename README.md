# MARLIN-Twin: Maritime Adaptive Resilience Learning with Integrated Networked Twins

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Q1 Research Framework**: Digital twin multi-agent reinforcement learning (MARL) for autonomous vessel traffic coordination under communication degradation and sensor failure.

---

## 🌟 Key Features

1. **Digital Twin Fusion & JPDA**: Real-time state estimation integrating AIS, marine radar, and kinematic dead reckoning with Joint Probabilistic Data Association (JPDA) during AIS denial.
2. **Bandwidth-Adaptive Communication**: Priority-queued message passing (`CRITICAL > HIGH > MEDIUM > LOW`) under dynamic weather degradation and localized jamming.
3. **2-Stage Curriculum Learning**: Mitigation of MARL non-stationarity by separating spatial movement learning (Stage 1) from bandwidth allocation & policy fine-tuning (Stage 2).
4. **COLREGs Rule Engine**: Full compliance evaluation for Rules 13 (Overtaking), 14 (Head-on), 15 (Crossing), and 17 (Action by Stand-on vessel).
5. **3-DOF MMG Vessel Dynamics**: Hydrodynamic equations of motion (Surge, Sway, Yaw rate) for heterogeneous vessel fleets.
6. **Coordination Resilience Index**: Sub-linear degradation metric $R_{\text{resilience}} = \int_0^1 \frac{J(\lambda)}{J(1.0)} d\lambda$.

---

## 📁 Repository Structure

```
MaritimeCoordEnv/
├── marlin_twin/                         # Core Python package
│   ├── envs/                            # Environment, MMG dynamics, COLREGs, Digital Twin
│   ├── agents/                          # GAT policy networks & observation builder
│   ├── training/                        # MAPPO trainer & 2-Stage Curriculum
│   ├── baselines/                       # IPPO, MADDPG & Rule-based controllers
│   ├── visualization/                   # Streamlit dashboard & nautical charts
│   ├── utils/                           # Resilience metrics & geometry helpers
│   ├── data_classes.py                  # Domain dataclasses
│   └── api.py                           # Unified MarlinTwinAPI facade
├── configs/                             # Scenario configuration files
├── scripts/                             # CLI execution & evaluation scripts
├── tests/                               # Pytest suite
└── docs/                                # Documentation & reference guides
```

---

## 🚀 Quickstart

### Installation

```bash
pip install -e .
```

### Python API Example

```python
import marlin_twin

# Create API facade with minimal scenario
api = marlin_twin.create_minimal_api(scenario_type="channel", n_vessels=5)

# Train and evaluate
result = api.train_and_evaluate(n_episodes=100)

# Evaluate resilience across communication degradation levels
resilience = api.evaluate_resilience(degradation_levels=[1.0, 0.5, 0.0])

# Print summary
api.print_summary()
```

### Launch Interactive Dashboard

```bash
streamlit run scripts/demo_dashboard.py
```

---

## 🧪 Running Tests

```bash
pytest tests/
```
