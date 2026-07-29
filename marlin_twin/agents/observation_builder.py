# ============================================================================
# FILE: marlin_twin/agents/observation_builder.py
# ============================================================================

import numpy as np
from marlin_twin.data_classes import VesselObservation

class ObservationBuilder:
    """Flattens structured VesselObservation dataclasses into fixed-size tensors for neural network consumption."""

    @staticmethod
    def to_vector(obs: VesselObservation, vector_dim: int = 32) -> np.ndarray:
        vec = np.zeros(vector_dim, dtype=np.float32)
        vec[0] = obs.own_state.x / 5000.0
        vec[1] = obs.own_state.y / 5000.0
        vec[2] = obs.own_state.heading / np.pi
        vec[3] = obs.own_state.speed / 15.0
        vec[4] = obs.own_state.surge_velocity / 15.0
        vec[5] = obs.own_state.yaw_rate

        idx = 6
        for nid, nstate in list(obs.neighbor_states.items())[:4]:
            if idx + 4 <= vector_dim:
                vec[idx] = (nstate.x - obs.own_state.x) / 5000.0
                vec[idx+1] = (nstate.y - obs.own_state.y) / 5000.0
                vec[idx+2] = nstate.heading / np.pi
                vec[idx+3] = nstate.speed / 15.0
                idx += 4

        return vec
