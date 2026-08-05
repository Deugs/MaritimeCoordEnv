#!/usr/bin/env python3
"""
CLI Script to execute MARLIN-Twin experiments.
Usage:
    python scripts/run_experiment.py --config configs/channel_5vessels.yaml --episodes 100
"""

import argparse
from pathlib import Path

import marlin_twin

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run MARLIN-Twin Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "minimal_test.yaml"),
        help="Path to config file",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    api = marlin_twin.MarlinTwinAPI()
    api.load_config(args.config)
    api.create_environment()

    print("Running training and evaluation...")
    api.train_and_evaluate(n_episodes=args.episodes)

    api.print_summary()


if __name__ == "__main__":
    main()
