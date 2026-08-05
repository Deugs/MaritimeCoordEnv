"""2D Euclidean geometry helpers (distance, angle normalization)."""

import numpy as np


def distance(
    pos1: tuple[float, float] | np.ndarray, pos2: tuple[float, float] | np.ndarray
) -> float:
    """Euclidean distance between two 2D coordinates."""
    p1 = np.array(pos1)
    p2 = np.array(pos2)
    return float(np.linalg.norm(p1 - p2))


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
