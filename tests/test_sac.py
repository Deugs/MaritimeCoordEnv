"""Verification tests for multi-agent SAC (MASAC): actor bounds/log-probs,
the PPO-update trap, checkpoint round-trip (incl. log_alpha rebind), twin-Q
minimum, timeout bootstrapping, and end-to-end gradient flow."""

import numpy as np
import pytest
import torch

from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.agents.networks import SquashedGaussianActor
from marlin_twin.baselines.sac import SACPolicy
from marlin_twin.training.sac import MASACTrainer
from marlin_twin.training.mappo import _build_scene_graph


def test_squashed_gaussian_actor_clamps_log_std_and_produces_bounded_actions():
    actor = SquashedGaussianActor(obs_dim=70, action_dim=2, hidden_dim=64)
    obs = torch.randn(8, 70) * 1e4
    mean, log_std = actor(obs)

    assert mean.shape == (8, 2)
    assert log_std.shape == (8, 2)
    assert torch.all(log_std >= SquashedGaussianActor.LOG_STD_MIN)
    assert torch.all(log_std <= SquashedGaussianActor.LOG_STD_MAX)
    assert torch.isfinite(log_std).all()

    pol = SACPolicy(n_vessels=2)
    pol.actor = actor
    action, logp = pol.sample_action(obs, deterministic=False)
    assert action.shape == (8, 2)
    # tanh is bounded on [-1, 1] -- with a deliberately extreme obs scale
    # (1e4) the pre-activation saturates float32 to exactly +/-1.0, which is
    # the correct saturated value, not an overflow bug.
    assert torch.all(action.abs() <= 1.0)
    assert logp.shape == (8, 1)
    assert torch.isfinite(logp).all()


def test_sac_policy_act_shape_deterministic_and_stochastic():
    config = MaritimeExperimentConfig(scenario_type="head_on", n_vessels=2)
    env = MaritimeCoordEnv(config)
    obs, _ = env.reset(seed=1)
    graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))

    pol = SACPolicy(n_vessels=2)
    a1 = pol.act(obs[0], graph, node_idx_map[0], deterministic=True)
    a2 = pol.act(obs[0], graph, node_idx_map[0], deterministic=True)
    assert a1.shape == (2,)
    assert np.all(np.isfinite(a1))
    assert np.all(np.abs(a1) < 1.0)
    assert np.allclose(a1, a2)  # deterministic is repeatable

    s1 = pol.act(obs[0], graph, node_idx_map[0], deterministic=False)
    s2 = pol.act(obs[0], graph, node_idx_map[0], deterministic=False)
    assert not np.allclose(s1, s2)  # stochastic sampling differs run to run


def test_sac_policy_does_not_trip_mappo_ppo_update_gate():
    pol = SACPolicy(n_vessels=2)
    assert not (hasattr(pol, "optimizer") and hasattr(pol, "evaluate_tensors"))


def test_sac_trainer_smoke_run_and_checkpoint_roundtrip(tmp_path):
    config = MaritimeExperimentConfig(n_vessels=2, episode_length=40, eval_frequency=5)
    env = MaritimeCoordEnv(config)
    trainer = MASACTrainer(config, batch_size=4, min_replay_size=8, random_warmup_steps=0)

    policies = trainer.train(env, n_episodes=8)
    assert len(policies) == 2
    assert len(trainer.reward_history) == 8
    assert len(trainer.alpha_history) == 8

    for pol in policies.values():
        assert isinstance(pol, SACPolicy)
        for net in (
            pol.encoder,
            pol.actor,
            pol.critic1,
            pol.critic2,
            pol.target_critic1,
            pol.target_critic2,
        ):
            for p in net.parameters():
                assert torch.isfinite(p).all()
        assert torch.isfinite(pol.log_alpha)

    ckpt = tmp_path / "sac_ckpt.pt"
    trainer.save_checkpoint(str(ckpt))
    trainer.load_checkpoint(str(ckpt))

    results = trainer.evaluate(env, policies, n_episodes=2)
    assert isinstance(results["average_reward"], float)
    assert np.isfinite(results["average_reward"])


def test_sac_checkpoint_roundtrip_preserves_and_rebinds_log_alpha(tmp_path):
    config = MaritimeExperimentConfig(n_vessels=2, episode_length=20, eval_frequency=5)
    env = MaritimeCoordEnv(config)
    trainer = MASACTrainer(config, batch_size=4, min_replay_size=8, random_warmup_steps=0)
    trainer.train(env, n_episodes=4)

    pol = trainer.policies[0]
    saved_alpha = float(pol.log_alpha.detach())

    ckpt = tmp_path / "sac_alpha_ckpt.pt"
    trainer.save_checkpoint(str(ckpt))

    with torch.no_grad():
        pol.log_alpha.fill_(999.0)
    trainer.load_checkpoint(str(ckpt))

    assert float(trainer.policies[0].log_alpha.detach()) == pytest.approx(saved_alpha)
    # copy_, not rebind -- the optimizer must still be tracking this exact tensor.
    assert trainer.policies[0].alpha_optimizer.param_groups[0]["params"][0] is (
        trainer.policies[0].log_alpha
    )


def test_sac_trainer_updates_encoder_actor_and_critics():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2, episode_length=20)
    env = MaritimeCoordEnv(config)
    trainer = MASACTrainer(config, batch_size=4, min_replay_size=8, random_warmup_steps=0)

    trainer.policies = {i: SACPolicy(n_vessels=2) for i in range(2)}
    before_encoder = {k: v.clone() for k, v in trainer.policies[0].encoder.state_dict().items()}
    before_actor = {k: v.clone() for k, v in trainer.policies[0].actor.state_dict().items()}
    before_critic1 = {k: v.clone() for k, v in trainer.policies[0].critic1.state_dict().items()}

    trainer.train(env, n_episodes=1)

    encoder = trainer.policies[0].encoder
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in encoder.parameters())

    after_encoder = encoder.state_dict()
    after_actor = trainer.policies[0].actor.state_dict()
    after_critic1 = trainer.policies[0].critic1.state_dict()
    assert any(not torch.equal(before_encoder[k], after_encoder[k]) for k in before_encoder)
    assert any(not torch.equal(before_actor[k], after_actor[k]) for k in before_actor)
    assert any(not torch.equal(before_critic1[k], after_critic1[k]) for k in before_critic1)


def test_sac_alpha_is_auto_tuned():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2, episode_length=20)
    env = MaritimeCoordEnv(config)
    trainer = MASACTrainer(config, batch_size=4, min_replay_size=8, random_warmup_steps=0)
    trainer.policies = {i: SACPolicy(n_vessels=2) for i in range(2)}

    before = float(trainer.policies[0].log_alpha.detach())
    trainer.train(env, n_episodes=3)
    after = float(trainer.policies[0].log_alpha.detach())

    assert np.isfinite(after)
    assert after != before


def test_min_target_q_uses_elementwise_minimum():
    pol = SACPolicy(n_vessels=2)
    joint_obs = torch.randn(4, 2, 70)
    joint_act = torch.randn(4, 2, 2)

    # Force target_critic2 to output a constant -1000.0 regardless of input.
    with torch.no_grad():
        for p in pol.target_critic2.net.parameters():
            p.zero_()
        pol.target_critic2.net[-1].bias.fill_(-1000.0)

    result = MASACTrainer._min_target_q(pol, joint_obs, joint_act)
    assert torch.allclose(result, torch.full((4, 1), -1000.0), atol=1e-3)

    # Now do the same to target_critic1 and confirm IT dominates too --
    # fails if the code always reads only one particular critic.
    pol2 = SACPolicy(n_vessels=2)
    with torch.no_grad():
        for p in pol2.target_critic1.net.parameters():
            p.zero_()
        pol2.target_critic1.net[-1].bias.fill_(-1000.0)
    result2 = MASACTrainer._min_target_q(pol2, joint_obs, joint_act)
    assert torch.allclose(result2, torch.full((4, 1), -1000.0), atol=1e-3)


def test_td_target_bootstraps_through_timeout():
    reward = torch.tensor([[1.0]])
    target_q = torch.tensor([[10.0]])
    gamma = 0.99
    dones_true = torch.ones(1, 1)
    dones_false = torch.zeros(1, 1)

    result_done = MASACTrainer._td_target(reward, target_q, gamma, dones_true)
    result_not_done = MASACTrainer._td_target(reward, target_q, gamma, dones_false)

    expected = reward + gamma * target_q
    assert torch.allclose(result_done, expected)
    assert torch.allclose(result_done, result_not_done)


def test_sac_trainer_fills_replay_buffer_with_joint_transitions():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2, episode_length=20)
    env = MaritimeCoordEnv(config)
    trainer = MASACTrainer(config, batch_size=4, min_replay_size=8, random_warmup_steps=0)
    trainer.train(env, n_episodes=1)

    assert len(trainer.replay_buffer) == 20
    batch = trainer.replay_buffer.sample(4)
    assert batch["own_feats"].shape == (4, 2, 6)
