# ============================================================================
# FILE: tests/test_phase5_training.py
# ============================================================================

import numpy as np
from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.training.rollout_buffer import RolloutBuffer
from marlin_twin.training.mappo import MAPPOTrainer
from marlin_twin.training.curriculum import TwoStageCurriculumTrainer
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv


def test_rollout_buffer_gae_computation():
    buffer = RolloutBuffer(buffer_size=10, n_vessels=2, feat_dim=32, act_dim=2)
    for t in range(10):
        obs = np.random.randn(2, 32).astype(np.float32)
        act = np.random.randn(2, 2).astype(np.float32)
        rew = np.array([1.0, -0.5], dtype=np.float32)
        val = np.array([0.5, 0.2], dtype=np.float32)
        logp = np.array([0.0, 0.0], dtype=np.float32)
        buffer.add(obs, act, rew, val, logp)

    buffer.compute_returns_and_advantages(last_values=np.zeros(2, dtype=np.float32))
    assert buffer.adv_buf.shape == (10, 2)
    assert buffer.ret_buf.shape == (10, 2)
    assert abs(np.mean(buffer.adv_buf)) < 1e-4  # Normalized advantages mean ~ 0


def test_mappo_trainer_short_loop():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2, episode_length=10)
    env = MaritimeCoordEnv(config)
    trainer = MAPPOTrainer(config)

    policies = trainer.train(env, n_episodes=2)
    assert len(policies) == 2
    assert 0 in policies


def test_two_stage_curriculum_trainer():
    config = MaritimeExperimentConfig(scenario_type="open_water", n_vessels=2, episode_length=10)
    env = MaritimeCoordEnv(config)
    curriculum = TwoStageCurriculumTrainer(config)

    policies = curriculum.train_curriculum(env, total_episodes=4)
    assert len(policies) == 2


def test_two_stage_curriculum_uses_distinct_seeds_across_all_episodes():
    """Regression guard: Stage 2 runs `self.train(env, 1)` once per episode,
    and without a seed_offset every one of those calls resets with the
    identical seed=0 (train()'s internal `ep` is always 0 for a
    single-episode call) -- silently replaying one episode for the entire
    40% of curriculum training Stage 2 covers, regardless of how many total
    episodes were requested."""
    config = MaritimeExperimentConfig(scenario_type="open_water", n_vessels=2, episode_length=5)
    env = MaritimeCoordEnv(config)
    curriculum = TwoStageCurriculumTrainer(config)

    seen_seeds = []
    original_reset = env.reset

    def recording_reset(seed=None, **kwargs):
        seen_seeds.append(seed)
        return original_reset(seed=seed, **kwargs)

    env.reset = recording_reset
    curriculum.train_curriculum(env, total_episodes=10)

    assert len(seen_seeds) == 10
    assert len(set(seen_seeds)) == 10
