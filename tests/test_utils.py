import random

import numpy as np
import pytest
from marlin_twin.utils.geometry import distance, normalize_angle
from marlin_twin.utils.seeding import seed_everything


def test_distance_euclidean():
    assert distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert distance(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.0)


def test_normalize_angle_wraps_to_pi_range():
    assert normalize_angle(0.0) == pytest.approx(0.0)
    assert normalize_angle(3 * np.pi) == pytest.approx(-np.pi, abs=1e-6)
    assert normalize_angle(-3 * np.pi) == pytest.approx(-np.pi, abs=1e-6)
    assert -np.pi <= normalize_angle(10.0) <= np.pi


def test_seed_everything_reproducible_across_random_and_numpy():
    seed_everything(123)
    random_draw_1 = random.random()
    numpy_draw_1 = np.random.rand()

    seed_everything(123)
    random_draw_2 = random.random()
    numpy_draw_2 = np.random.rand()

    assert random_draw_1 == random_draw_2
    assert numpy_draw_1 == numpy_draw_2
