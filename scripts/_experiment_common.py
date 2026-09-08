"""Shared experiment-orchestration driver used by every Phase 1-3 study
script (run_training_budget_study.py, run_reward_reweighting_study.py,
run_scenario_battery.py): the Pool(4) parallelism pattern, a checkpoint-
naming convention that never overwrites a differently-configured checkpoint,
JSON result persistence (git SHA + full per-seed raw values, not just
aggregates), and a Welch's t-test + Cohen's d significance helper.

Callers MUST set `os.environ.setdefault("OMP_NUM_THREADS", "1")` before
importing `multiprocessing` themselves (mirroring
scripts/retrain_all_fixed_dynamics.py's existing pattern) -- this module
defers its own `import multiprocessing` to inside `run_pool` so it doesn't
matter which of the two modules a caller imports first.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"


def init_worker() -> None:
    """`multiprocessing.Pool` initializer. Naive `Pool(4)` without pinning
    each worker's own torch BLAS thread pool to 1 causes catastrophic CPU
    oversubscription -- each of the 4 workers' multi-threaded BLAS pool
    competes with the other 3 workers' -- confirmed the hard way earlier in
    this project's history (see retrain_all_fixed_dynamics.py)."""
    import torch

    torch.set_num_threads(1)


def run_pool(job_fn, jobs: list, n_workers: int = 4) -> list:
    """Run `job_fn` over `jobs` under a `Pool(n_workers)` with
    `init_worker` pinning each worker to a single BLAS thread. Deferred
    `import multiprocessing` so callers can set `OMP_NUM_THREADS` first."""
    import multiprocessing as mp

    with mp.Pool(n_workers, initializer=init_worker) as pool:
        return list(pool.imap_unordered(job_fn, jobs))


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def checkpoint_path(tag: str, variant: str, seed: int) -> str:
    """`tag` distinguishes budget/experiment arms sharing the same variant
    name without colliding on disk or silently overwriting a differently-
    configured checkpoint -- e.g. `tag="e150"`/`"e500"` for Phase 1's
    training-budget study, `tag="rwA1"` for a Phase 2 reward-reweighting arm.
    Never reuses the bare `{variant}_seed_{seed}.pt` convention the original
    150-episode retraining scripts use -- those checkpoints back the
    currently-published figures and must not be touched by new studies."""
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    return str(CHECKPOINTS_DIR / f"{tag}_{variant}_seed_{seed}.pt")


def checkpoint_exists(tag: str, variant: str, seed: int) -> bool:
    return os.path.exists(checkpoint_path(tag, variant, seed))


@dataclass
class TrainingJob:
    """One (variant, seed) unit of training work handed to a Pool worker."""

    tag: str
    variant: str
    seed: int
    n_episodes: int
    extra: dict = field(default_factory=dict)  # arbitrary per-job config overrides


def save_results_json(name: str, payload: dict) -> str:
    """Persist a study's full results -- config, seeds, episode counts, and
    per-seed RAW values (not just aggregated mean/std) -- under a git SHA,
    so tables/figures can be regenerated later without re-running anything
    and so a result can always be traced back to the exact code that
    produced it."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    full_payload = {"git_sha": get_git_sha(), **payload}
    out_path = RESULTS_DIR / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(full_payload, f, indent=2, default=str)
    return str(out_path)


def load_results_json(name: str) -> dict:
    with open(RESULTS_DIR / f"{name}.json") as f:
        return json.load(f)


def welch_ttest_and_cohens_d(sample_a: list[float], sample_b: list[float]) -> dict:
    """Welch's t-test (unequal variances, doesn't assume equal sample size)
    plus Cohen's d effect size, for comparing two algorithms' per-seed J(1.0)
    (or any other per-seed metric) values. At the n=4-5 seed counts used
    throughout this repo, "overlapping +/-1 std error bars" is not a
    substitute for a stated test -- this is what "statistically tied" or
    "statistically distinguishable" should mean in any table that uses
    those words."""
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * a.std(ddof=1) ** 2 + (n_b - 1) * b.std(ddof=1) ** 2) / (n_a + n_b - 2)
    )
    cohens_d = float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0

    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "significant_at_0.05": bool(p_value < 0.05),
        "n_a": n_a,
        "n_b": n_b,
    }
