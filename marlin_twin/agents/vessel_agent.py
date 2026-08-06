"""Wrapper connecting VesselAgent state tracking with RL policy decisions."""

import numpy as np
from marlin_twin.data_classes import VesselAgent, VesselObservation, VesselAction, MessagePriority
from marlin_twin.agents.policies import GATPolicy


class VesselAgentWrapper:
    """Wrapper connecting VesselAgent state tracking with RL policy decisions."""

    def __init__(self, agent: VesselAgent, policy: GATPolicy | None = None):
        self.agent = agent
        self.policy = policy or GATPolicy()

    def select_action(
        self,
        observation: VesselObservation,
        graph=None,
        node_idx: int = None,
        deterministic: bool = False,
    ) -> VesselAction:
        """`graph`/`node_idx` are the shared per-scene encounter graph and
        this vessel's node index within it (built once per step by the
        caller, which has access to every vessel's state) — only used by
        graph-based policies (`GATPolicy`/`MeanPoolingPolicy`); other policy
        types ignore them."""
        act_arr = self.policy.act(observation, graph, node_idx, deterministic=deterministic)
        return self.build_action(observation, act_arr)

    def build_action(self, observation: VesselObservation, act_arr: np.ndarray) -> VesselAction:
        """Build a VesselAction from an already-computed tanh-squashed [-1,1]
        action vector (e.g. from `policy.get_action_and_val`) without drawing a
        new sample — used during training so the action sent to the
        environment matches the sample whose log-prob/value were recorded."""
        rpm = float(np.clip(act_arr[0] * 0.5 + 0.6, 0.2, 1.0))
        rudder = float(np.clip(act_arr[1] * (np.pi / 6), -np.pi / 6, np.pi / 6))

        # Communication action: transmit to neighbors within 3km if risk > 0.3
        targets = [
            nid
            for nid, st in observation.neighbor_states.items()
            if np.linalg.norm(st.position() - observation.own_state.position()) < 3000.0
        ]

        return VesselAction(
            vessel_id=self.agent.vessel_id,
            propeller_rpm=rpm,
            rudder_angle=rudder,
            message_targets=targets[:3],  # Max 3 targets
            message_priority=MessagePriority.MEDIUM,
        )
