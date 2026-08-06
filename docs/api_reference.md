# MARLIN-Twin API Reference

## Primary Facade: `MarlinTwinAPI`

```python
import marlin_twin

# Create API instance
api = marlin_twin.create_minimal_api(scenario_type="channel", n_vessels=5)

# Configure experiment parameters
api.configure(bandwidth_bps=4800.0, dt_enabled=True)

# Train and evaluate
result = api.train_and_evaluate(n_episodes=100)

# Evaluate resilience
metrics = api.evaluate_resilience(degradation_levels=[1.0, 0.5, 0.0])

# Export results
api.export_results("experiment_results/")
```

## Key Classes & Protocols

* `MarlinTwinAPI`: Main interface facade.
* `MaritimeCoordEnv`: Multi-vessel Gym environment.
* `MMGDynamicsSolver`: 3-DOF hydrodynamics RK4 solver.
* `DigitalTwinEstimator`: Kalman Filter & JPDA data association engine.
* `COLREGsEngine`: Rules 13-18 classification and compliance validator.
* `GATPolicy`: Graph Attention Network multi-agent policy — `own_feats ++ GATEncoder(scene_graph)[own_node]` fed into a shared actor-critic, trained on-policy via `MAPPOTrainer`.
* `MeanPoolingPolicy`: Ablation of `GATPolicy` — uniform 1/degree neighbor aggregation instead of learned attention.
* `MLPPolicy`: Ablation with no graph encoder — the original fixed-neighbor-cap flattened observation vector.
* `IndependentPPOPolicy`: Independent-learner baseline — own vessel state only, no neighbor or communication information.
* `MADDPGPolicy` / `MADDPGTrainer`: Multi-Agent DDPG baseline — decentralized deterministic actors, per-agent centralized critics, target networks with soft updates, trained off-policy via `ReplayBuffer`.
* `TwoStageCurriculumTrainer`: 2-Stage Curriculum training manager (builds on `MAPPOTrainer`).
