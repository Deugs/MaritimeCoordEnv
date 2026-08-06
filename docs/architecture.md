# MARLIN-Twin System Architecture

## Overview

MARLIN-Twin (**M**aritime **A**daptive **R**esilience **L**earning with **I**ntegrated **N**etworked **Twin**s) is a digital twin multi-agent reinforcement learning framework designed for autonomous vessel coordination under communication degradation.

```
+-----------------------------------------------------------------------------+
|                         MARLIN-TWIN FRAMEWORK                                |
+-----------------------------------------------------------------------------+
|                                                                              |
|  +------------------+     +------------------+     +----------------------+ |
|  |   MARITIME       |     |   MAPPO          |     |   DIGITAL TWIN       | |
|  |   ENVIRONMENT    |<--->|   TRAINING       |<--->|   STATE ESTIMATOR    | |
|  |                  |     |   LOOP           |     |   (Kalman + JPDA)    | |
|  +------------------+     +------------------+     +----------------------+ |
|           ^                        ^                        ^                |
|           |                        |                        |                |
|           v                        v                        v                |
|  +------------------+     +------------------+     +----------------------+ |
|  |   VESSEL         |     |   GNN +          |     |   BANDWIDTH-         | |
|  |   DYNAMICS       |     |   POLICY         |     |   ADAPTIVE COMMS     | |
|  |   (3-DOF MMG)    |     |   NETWORKS       |     |   CHANNEL            | |
|  +------------------+     +------------------+     +----------------------+ |
|                                                                              |
|  +------------------+     +------------------+     +----------------------+ |
|  |   COLREGs        |     |   RESILIENCE     |     |   VISUALIZATION      | |
|  |   RULE ENGINE    |     |   METRICS        |     |   & DASHBOARD        | |
|  +------------------+     +------------------+     +----------------------+ |
+-----------------------------------------------------------------------------+
```

## Key Subsystems

1. **Hydrodynamic Dynamics (`marlin_twin.envs.vessel_dynamics`)**: 3-DOF MMG differential equations solved with 4th-order Runge-Kutta.
2. **COLREGs Engine (`marlin_twin.envs.colregs`)**: Rule checking for Rules 13 (Overtaking), 14 (Head-on), 15 (Crossing), and 17 (Stand-on action).
3. **Digital Twin (`marlin_twin.envs.digital_twin`)**: Extended Kalman Filtering with Joint Probabilistic Data Association (JPDA).
4. **Communication Channel (`marlin_twin.envs.communication`)**: Priority-queued bandwidth allocation.
5. **GNN Policy Networks (`marlin_twin.agents.networks`)**: Graph Attention Networks (GAT), run end-to-end over a shared per-timestep encounter graph — the encoder trains jointly with the actor-critic, not as a frozen feature extractor.
6. **2-Stage Curriculum (`marlin_twin.training.curriculum`)**: Two-phase training for multi-agent policy stability, built on the on-policy `MAPPOTrainer`.
7. **Off-Policy MADDPG (`marlin_twin.training.maddpg`)**: Decentralized deterministic actors with per-agent centralized critics and target networks, trained off-policy from a joint-transition replay buffer — an independent baseline for comparison against the on-policy MAPPO/curriculum trainers.
