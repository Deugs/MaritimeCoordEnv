#!/usr/bin/env python3
"""
Full Retraining Driver -- Fixed CARGO/USV Turning Dynamics:
Every checkpoint in checkpoints/ was trained under the pre-fix CARGO/USV N_r/
yaw_coefficient values (marlin_twin/data_classes.py, marlin_twin/envs/vessel_profiles.py),
which gave both vessel types a much weaker yaw response than the values now used (needed
to satisfy the IMO Res. MSC.137(76) turning-circle criterion -- see those two files'
updated docstrings). Since every scenario in scenarios.py builds its vessels from those
same profiles, this changes how sharply every vessel in every RL experiment can turn --
every existing checkpoint is stale under the corrected physics and must be retrained, not
just the sea-trial figure (fig5).

Retrains, unconditionally overwriting existing checkpoints:
  - 6 baseline/ablation variants x 4 seeds (marlin_twin, ablation_mean_pooling,
    ablation_flat_mlp, ablation_no_digital_twin, independent_ppo, maddpg) x
    (42, 100, 200, 300) -- backs fig8/fig9 (benchmark/resilience) and fig12
    (ablation study).
  - 3 multi-vessel-generalization variants x 4 seeds (marlin_twin, independent_ppo,
    ablation_flat_mlp) x (42, 100, 200, 300) -- backs the paper's Multi-Vessel
    Scenario Generalization section / Table tab:multivessel_table.

Uses a 4-worker multiprocessing.Pool with torch.set_num_threads(1) per worker (naive
Pool(4) without this causes catastrophic CPU oversubscription -- each worker's own
multi-threaded torch BLAS pool competes with the other 3 workers' -- confirmed the hard
way earlier in this project's history).

Usage:
    python scripts/retrain_all_fixed_dynamics.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import multiprocessing as mp  # noqa: E402

from loguru import logger  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

SEEDS = [42, 100, 200, 300]
BASELINE_VARIANTS = [
    "marlin_twin",
    "ablation_mean_pooling",
    "ablation_flat_mlp",
    "ablation_no_digital_twin",
    "independent_ppo",
    "maddpg",
]
MV4_VARIANTS = ["marlin_twin", "independent_ppo", "ablation_flat_mlp"]
N_EPISODES = 150


def _init_worker():
    import torch

    torch.set_num_threads(1)


def _train_baseline_job(args):
    variant, seed = args
    import torch  # noqa: F401  (ensures torch is imported inside the worker before use)
    from run_retrain_all_baselines import retrain_variant

    logger.info(f"[baseline] {variant} seed {seed}: starting ({N_EPISODES} episodes)")
    retrain_variant(variant, seeds=[seed], n_episodes=N_EPISODES)
    logger.info(f"[baseline] {variant} seed {seed}: done")
    return ("baseline", variant, seed)


def _train_mv4_job(args):
    variant, seed = args
    import torch  # noqa: F401
    from run_multivessel_generalization import train_variant

    logger.info(f"[mv4] {variant} seed {seed}: starting ({N_EPISODES} episodes)")
    train_variant(variant, seed)
    logger.info(f"[mv4] {variant} seed {seed}: done")
    return ("mv4", variant, seed)


def main():
    os.makedirs(os.path.join(REPO_ROOT, "checkpoints"), exist_ok=True)

    baseline_jobs = [(v, s) for v in BASELINE_VARIANTS for s in SEEDS]
    mv4_jobs = [(v, s) for v in MV4_VARIANTS for s in SEEDS]

    print(f"=== Retraining {len(baseline_jobs)} baseline + {len(mv4_jobs)} mv4 checkpoints ===")

    with mp.Pool(4, initializer=_init_worker) as pool:
        print("--- Baseline/ablation variants ---")
        for result in pool.imap_unordered(_train_baseline_job, baseline_jobs):
            print(f"  completed: {result}")

        print("--- Multi-vessel generalization variants ---")
        for result in pool.imap_unordered(_train_mv4_job, mv4_jobs):
            print(f"  completed: {result}")

    print("=== All checkpoints retrained under corrected dynamics ===")


if __name__ == "__main__":
    main()
