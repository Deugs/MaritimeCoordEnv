# ============================================================================
# FILE: marlin_twin/agents/vessel_agent.py
# ============================================================================

import numpy as np
from marlin_twin.data_classes import VesselAgent, VesselObservation, VesselAction, MessagePriority
from marlin_twin.agents.policies import GATPolicy

class VesselAgentWrapper:
    """Wrapper connecting VesselAgent state tracking with RL policy decisions."""

    def __init__(self, agent: VesselAgent, policy: GATPolicy | None = None):
        self.agent = agent
        self.policy = policy or GATPolicy()

    def select_action(self, observation: VesselObservation, deterministic: bool = False) -> VesselAction:
        obs_vec = np.zeros(32, dtype=np.float32)
        obs_vec[0] = observation.own_state.x / 5000.0
        obs_vec[1] = observation.own_state.y / 5000.0
        obs_vec[2] = observation.own_state.heading / np.pi
        obs_vec[3] = observation.own_state.speed / 15.0

        act_arr = self.policy.act(obs_vec, deterministic=deterministic)

        rpm = float(np.clip(act_arr[0], -1.0, 1.0))
        rudder = float(np.clip(act_arr[1], -np.pi / 6, np.pi / 6))

        # Communication action: transmit to neighbors within 3km if risk > 0.3
        targets = [nid for nid, st in observation.neighbor_states.items() if np.linalg.norm(st.position() - observation.own_state.position()) < 3000.0]

        return VesselAction(
            vessel_id=self.agent.vessel_id,
            propeller_rpm=rpm,
            rudder_angle=rudder,
            message_targets=targets[:3],  # Max 3 targets
            message_priority=MessagePriority.MEDIUM
        )
