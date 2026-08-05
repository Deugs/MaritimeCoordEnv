# ============================================================================
# FILE: tests/test_phase4_communication.py
# ============================================================================

import pytest
import torch
import numpy as np
from marlin_twin.data_classes import VesselState
from marlin_twin.agents.networks import GATEncoder
from marlin_twin.agents.communication_layer import CommunicationLayer
from marlin_twin.agents.policies import GATPolicy


def test_gat_encoder_attention_computation():
    encoder = GATEncoder(in_features=6, edge_features=4, hidden_dim=32, heads=4)
    x = torch.randn(3, 6)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)

    out, alpha = encoder(x, edge_index, edge_attr, return_attention=True)
    assert out.shape == (3, 32)
    assert alpha.shape[0] == 4  # 4 edges


def test_binary_message_serialization_roundtrip():
    state = VesselState(vessel_id=1, x=1234.5, y=-567.8, heading=1.23, speed=12.5)
    payload = CommunicationLayer.encode_binary(sender_id=1, receiver_id=2, state=state)

    assert len(payload) == 16  # Exactly 128 bits
    x, y, heading, speed = CommunicationLayer.decode_binary(payload)

    assert x == pytest.approx(1234.5, abs=1e-3)
    assert y == pytest.approx(-567.8, abs=1e-3)
    assert heading == pytest.approx(1.23, abs=1e-3)
    assert speed == pytest.approx(12.5, abs=1e-3)


def test_gat_policy_action_generation():
    policy = GATPolicy(obs_dim=32, action_dim=2)
    obs = np.random.randn(32).astype(np.float32)
    act = policy.act(obs, deterministic=True)

    assert act.shape == (2,)
    assert -1.0 <= act[0] <= 1.0
    assert -1.0 <= act[1] <= 1.0
