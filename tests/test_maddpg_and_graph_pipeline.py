"""Verification tests for the graph-native policy redesign and real MADDPG:
self-loops, MeanPoolingEncoder uniformity, per-policy pipeline distinctness,
RolloutBuffer/ReplayBuffer graph-batching round-trips, MADDPGTrainer
end-to-end training, the tanh log-prob correction, and encoder gradient flow.
"""

import numpy as np
import torch
from torch.distributions import Normal

from marlin_twin.data_classes import MaritimeExperimentConfig, VesselAction, VesselState
from marlin_twin.envs.encounters import EncounterManager
from marlin_twin.envs.maritime_coord_env import MaritimeCoordEnv
from marlin_twin.agents.networks import MeanPoolingEncoder
from marlin_twin.agents.policies import (
    GATPolicy,
    MeanPoolingPolicy,
    MLPPolicy,
    own_feats,
    tanh_corrected_log_prob,
)
from marlin_twin.baselines.independent_ppo import IndependentPPOPolicy
from marlin_twin.baselines.maddpg import MADDPGPolicy
from marlin_twin.training.mappo import MAPPOTrainer, _build_scene_graph
from marlin_twin.training.maddpg import MADDPGTrainer
from marlin_twin.training.replay_buffer import ReplayBuffer
from marlin_twin.training.rollout_buffer import RolloutBuffer


def _make_states(n: int) -> dict[int, VesselState]:
    return {
        i: VesselState(vessel_id=i, x=i * 200.0, y=0.0, heading=0.0, speed=5.0) for i in range(n)
    }


def test_build_encounter_graph_adds_self_loops_by_default():
    states = _make_states(3)
    graph = EncounterManager.build_encounter_graph(states, timestamp=0.0)
    edge_index = graph.edge_index

    pairs = {(int(edge_index[0, k]), int(edge_index[1, k])) for k in range(edge_index.shape[1])}
    for i in range(3):
        assert (i, i) in pairs

    for k in range(edge_index.shape[1]):
        if edge_index[0, k] == edge_index[1, k]:
            assert np.allclose(graph.edge_features[k], 0.0)


def test_build_encounter_graph_can_omit_self_loops():
    states = _make_states(3)
    graph = EncounterManager.build_encounter_graph(states, timestamp=0.0, include_self_loops=False)
    edge_index = graph.edge_index
    for k in range(edge_index.shape[1]):
        assert edge_index[0, k] != edge_index[1, k]


def test_mean_pooling_encoder_uses_uniform_not_learned_weights():
    encoder = MeanPoolingEncoder(in_features=6, edge_features=4, hidden_dim=8, heads=1)
    x = torch.randn(4, 6)
    # Node 0 has 3 incoming edges (from 1, 2, 3); node 1 has 1 incoming edge (from 0).
    edge_index = torch.tensor([[1, 2, 3, 0], [0, 0, 0, 1]], dtype=torch.long)
    edge_attr = torch.zeros(4, 4)

    out, alpha = encoder(x, edge_index, edge_attr, return_attention=True)

    assert torch.allclose(alpha[:3], torch.full((3,), 1.0 / 3.0))
    assert torch.allclose(alpha[3:4], torch.ones(1))

    h_node = encoder.node_proj(x)
    expected_node0 = torch.relu(encoder.out_proj((h_node[1] + h_node[2] + h_node[3]) / 3.0))
    assert torch.allclose(out[0], expected_node0, atol=1e-5)


def test_four_policies_have_distinct_obs_dims_and_actions():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=3)
    env = MaritimeCoordEnv(config)
    obs, _ = env.reset(seed=7)
    graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))

    policies = {
        "gat": GATPolicy(),
        "mean_pooling": MeanPoolingPolicy(),
        "mlp": MLPPolicy(),
        "independent_ppo": IndependentPPOPolicy(),
    }
    expected_dims = {"gat": 70, "mean_pooling": 70, "mlp": 32, "independent_ppo": 6}
    for name, pol in policies.items():
        assert pol.net.actor[0].in_features == expected_dims[name]

    actions = {
        name: pol.act(obs[0], graph, node_idx_map[0], deterministic=True)
        for name, pol in policies.items()
    }
    names = list(actions.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.allclose(actions[names[i]], actions[names[j]])


def test_rollout_buffer_batched_graph_matches_stored_node_features():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=3)
    env = MaritimeCoordEnv(config)
    buffer = RolloutBuffer(buffer_size=4, n_vessels=3, feat_dim=6, act_dim=2)

    obs, _ = env.reset(seed=3)
    for _ in range(4):
        graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))
        feat = np.stack([own_feats(obs[v]) for v in range(3)])
        node_idx_vec = np.array([node_idx_map[v] for v in range(3)], dtype=np.int64)
        act = np.zeros((3, 2), dtype=np.float32)
        rew = np.zeros(3, dtype=np.float32)
        val = np.zeros(3, dtype=np.float32)
        logp = np.zeros(3, dtype=np.float32)
        buffer.add(feat, act, rew, val, logp, graph=graph, node_idx=node_idx_vec)

        actions = {
            v: VesselAction(vessel_id=v, propeller_rpm=0.6, rudder_angle=0.0, message_targets=[])
            for v in range(3)
        }
        obs, _, _, done, _ = env.step(actions)
        if done:
            break

    batch, global_idx = buffer.batched_graph(vid=0)
    for t in range(buffer.ptr):
        node_row = batch.x[global_idx[t]]
        expected_row = buffer.graphs[t].x[buffer.node_idx_buf[t, 0]].clone()
        assert torch.allclose(node_row, expected_row)


def test_replay_buffer_sample_shapes():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2)
    env = MaritimeCoordEnv(config)
    rb = ReplayBuffer(capacity=20, n_vessels=2, feat_dim=6, act_dim=2)

    obs, _ = env.reset(seed=5)
    for _ in range(10):
        graph, node_idx_map = _build_scene_graph(env, obs.keys(), float(env.time_step))
        own_feats_vec = np.stack([own_feats(obs[v]) for v in range(2)])
        node_idx_vec = np.array([node_idx_map[v] for v in range(2)], dtype=np.int64)
        act_vec = np.zeros((2, 2), dtype=np.float32)

        actions = {
            v: VesselAction(vessel_id=v, propeller_rpm=0.6, rudder_angle=0.0, message_targets=[])
            for v in range(2)
        }
        next_obs, rewards, _, done, _ = env.step(actions)

        next_graph, next_node_idx_map = _build_scene_graph(
            env, next_obs.keys(), float(env.time_step)
        )
        next_own_feats_vec = np.stack([own_feats(next_obs[v]) for v in range(2)])
        next_node_idx_vec = np.array([next_node_idx_map[v] for v in range(2)], dtype=np.int64)
        rew_arr = np.array([rewards.get(v, 0.0) for v in range(2)], dtype=np.float32)

        rb.add(
            own_feats_vec,
            act_vec,
            rew_arr,
            next_own_feats_vec,
            done,
            graph,
            node_idx_vec,
            next_graph,
            next_node_idx_vec,
        )
        obs = next_obs
        if done:
            break

    batch = rb.sample(batch_size=4)
    assert batch["own_feats"].shape == (4, 2, 6)
    assert batch["actions"].shape == (4, 2, 2)
    assert batch["rewards"].shape == (4, 2)
    assert batch["dones"].shape == (4,)
    assert batch["node_idx"].shape == (4, 2)
    assert batch["next_node_idx"].shape == (4, 2)


def test_maddpg_trainer_smoke_run_and_checkpoint_roundtrip(tmp_path):
    config = MaritimeExperimentConfig(n_vessels=2, episode_length=40, eval_frequency=5)
    env = MaritimeCoordEnv(config)
    trainer = MADDPGTrainer(config)

    policies = trainer.train(env, n_episodes=8)
    assert len(policies) == 2
    assert len(trainer.reward_history) == 8

    for pol in policies.values():
        assert isinstance(pol, MADDPGPolicy)
        for net in (
            pol.encoder,
            pol.actor,
            pol.critic,
            pol.target_encoder,
            pol.target_actor,
            pol.target_critic,
        ):
            for p in net.parameters():
                assert torch.isfinite(p).all()

    ckpt = tmp_path / "maddpg_ckpt.pt"
    trainer.save_checkpoint(str(ckpt))
    trainer.load_checkpoint(str(ckpt))

    results = trainer.evaluate(env, policies, n_episodes=2)
    assert isinstance(results["average_reward"], float)
    assert np.isfinite(results["average_reward"])


def test_tanh_corrected_log_prob_matches_hand_computation():
    mean = torch.tensor([[0.3]])
    std = torch.tensor([[0.5]])
    raw_action = torch.tensor([[0.2]])
    dist = Normal(mean, std)

    result = tanh_corrected_log_prob(dist, raw_action)

    expected = dist.log_prob(raw_action) - torch.log(1 - torch.tanh(raw_action) ** 2 + 1e-6)
    expected = expected.sum(dim=-1, keepdim=True)
    assert torch.allclose(result, expected, atol=1e-6)

    uncorrected = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
    assert not torch.allclose(result, uncorrected)


def test_mappo_trainer_actually_updates_encoder_gradients_and_weights():
    config = MaritimeExperimentConfig(scenario_type="channel", n_vessels=2, episode_length=20)
    env = MaritimeCoordEnv(config)
    trainer = MAPPOTrainer(config)
    trainer.policies = {i: GATPolicy() for i in range(2)}

    before = {k: v.clone() for k, v in trainer.policies[0].encoder.state_dict().items()}

    trainer.train(env, n_episodes=1)

    encoder = trainer.policies[0].encoder
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in encoder.parameters())

    after = encoder.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
