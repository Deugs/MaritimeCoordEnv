"""Flattens structured VesselObservation dataclasses into fixed-size tensors."""

import numpy as np
from marlin_twin.data_classes import VesselObservation


class ObservationBuilder:
    """Flattens structured VesselObservation dataclasses into fixed-size NN input tensors."""

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
                vec[idx + 1] = (nstate.y - obs.own_state.y) / 5000.0
                vec[idx + 2] = nstate.heading / np.pi
                vec[idx + 3] = nstate.speed / 15.0
                idx += 4

        return vec

    @staticmethod
    def build_observation(
        vessel_id: int, all_states: dict, twin_estimates: dict | None = None
    ) -> VesselObservation:
        """Construct structured VesselObservation dataclass."""
        own_state = all_states[vessel_id]
        neighbor_states = {vid: st for vid, st in all_states.items() if vid != vessel_id}
        twin_est = twin_estimates.get(vessel_id) if twin_estimates else None

        return VesselObservation(
            vessel_id=vessel_id,
            own_state=own_state,
            neighbor_states=neighbor_states,
            digital_twin_estimate=twin_est,
            received_messages=[],
        )

    @staticmethod
    def to_pyg_graph(scene_states: dict, encounters: list = []):
        """Construct PyTorch Geometric Data graph representation."""
        try:
            import torch
            from torch_geometric.data import Data

            v_ids = list(scene_states.keys())
            num_nodes = len(v_ids)

            # Node features (N x 6)
            node_feats = []
            for vid in v_ids:
                st = scene_states[vid]
                node_feats.append(
                    [
                        st.x / 5000.0,
                        st.y / 5000.0,
                        st.heading / np.pi,
                        st.speed / 15.0,
                        st.surge_velocity / 15.0,
                        st.yaw_rate,
                    ]
                )

            x = torch.tensor(node_feats, dtype=torch.float)

            # Fully connected or encounter edges (2 x E)
            edge_src, edge_dst, edge_attrs = [], [], []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_src.append(i)
                        edge_dst.append(j)
                        st_i, st_j = scene_states[v_ids[i]], scene_states[v_ids[j]]
                        dist = np.linalg.norm(st_j.position() - st_i.position()) / 5000.0
                        bearing = (
                            np.arctan2(st_j.x - st_i.x, st_j.y - st_i.y) - st_i.heading + np.pi
                        ) % (2 * np.pi) - np.pi
                        edge_attrs.append([dist, bearing / np.pi, st_j.speed / 15.0, 0.0])

            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        except ImportError:
            return None
