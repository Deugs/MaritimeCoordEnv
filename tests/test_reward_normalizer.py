import numpy as np
import pytest
from marlin_twin.training.reward_normalizer import RewardNormalizer, RunningMeanStd


def test_running_mean_std_converges_to_batch_statistics():
    rms = RunningMeanStd(shape=())
    rms.update(np.array([1.0, 2.0, 3.0]))

    assert rms.mean == pytest.approx(2.0, abs=0.1)
    assert rms.var > 0.0


def test_running_mean_std_accumulates_across_updates():
    rms = RunningMeanStd(shape=())
    rms.update(np.array([1.0, 1.0, 1.0]))
    rms.update(np.array([5.0, 5.0, 5.0]))

    assert rms.count > 6.0
    assert 1.0 < rms.mean < 5.0


def test_reward_normalizer_returns_finite_scaled_reward():
    normalizer = RewardNormalizer(shape=())

    values = [normalizer.normalize(1.0) for _ in range(20)]

    assert all(np.isfinite(v) for v in values)
    assert all(isinstance(v, float) for v in values)
