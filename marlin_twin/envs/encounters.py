"""CPA/TCPA/DCPA computation and encounter graph construction for GNN processing."""

import numpy as np
from marlin_twin.data_classes import VesselState, Encounter, EncounterGraph, EncounterType
from marlin_twin.envs.colregs import COLREGsEngine


class EncounterManager:
    """
    Computes CPA (Closest Point of Approach), TCPA, DCPA between vessel pairs,
    and constructs dynamic encounter graphs for GNN processing.
    """

    @staticmethod
    def compute_cpa(state_i: VesselState, state_j: VesselState) -> tuple[float, float, float]:
        """
        Compute TCPA (seconds) and DCPA (meters).
        Returns: (tcpa, dcpa, cpa_distance)
        """
        r = state_j.position() - state_i.position()
        v = state_j.velocity_vector() - state_i.velocity_vector()

        v_sq = np.dot(v, v)
        if v_sq < 1e-6:
            dist = float(np.linalg.norm(r))
            return 0.0, dist, dist

        tcpa = float(-np.dot(r, v) / v_sq)
        if tcpa < 0.0:
            dist = float(np.linalg.norm(r))
            return dist, 0.0, dist

        cpa_pos_i = state_i.position() + state_i.velocity_vector() * tcpa
        cpa_pos_j = state_j.position() + state_j.velocity_vector() * tcpa
        dcpa = float(np.linalg.norm(cpa_pos_j - cpa_pos_i))

        return dcpa, tcpa, dcpa

    @classmethod
    def detect_encounters(cls, states: dict[int, VesselState]) -> list[Encounter]:
        """Detect and classify encounters across all vessel pairs."""
        encounters = []
        v_ids = list(states.keys())
        for i in range(len(v_ids)):
            for j in range(i + 1, len(v_ids)):
                st_i, st_j = states[v_ids[i]], states[v_ids[j]]
                cpa_dist, tcpa, dcpa = cls.compute_cpa(st_i, st_j)

                enc_type, rule = COLREGsEngine.classify_encounter(st_i, st_j, cpa_dist)
                if enc_type != EncounterType.NO_ENCOUNTER:
                    rel_pos = st_j.position() - st_i.position()
                    rel_bearing = float(
                        (np.arctan2(rel_pos[0], rel_pos[1]) - st_i.heading + np.pi) % (2 * np.pi)
                        - np.pi
                    )
                    encounters.append(
                        Encounter(
                            vessel_i=st_i.vessel_id,
                            vessel_j=st_j.vessel_id,
                            encounter_type=enc_type,
                            colregs_rule=rule,
                            cpa_distance=cpa_dist,
                            cpa_time=tcpa,
                            tcpa=tcpa,
                            dcpa=dcpa,
                            relative_bearing=rel_bearing,
                            is_dangerous=cpa_dist < 500.0,
                        )
                    )
        return encounters

    @classmethod
    def build_encounter_graph(
        cls, states: dict[int, VesselState], timestamp: float
    ) -> EncounterGraph:
        """Build dynamic encounter graph for GNN policy encoding."""
        v_ids = sorted(list(states.keys()))
        n_nodes = len(v_ids)

        # Node features: [x, y, heading, speed, surge, yaw_rate]
        node_feats = np.zeros((n_nodes, 6), dtype=np.float32)
        for idx, vid in enumerate(v_ids):
            s = states[vid]
            node_feats[idx] = [
                s.x / 5000.0,
                s.y / 5000.0,
                s.heading / np.pi,
                s.speed / 15.0,
                s.surge_velocity / 15.0,
                s.yaw_rate,
            ]

        edges = []
        edge_feats = []
        edge_types = []

        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                st_i, st_j = states[v_ids[i]], states[v_ids[j]]
                dist = np.linalg.norm(st_j.position() - st_i.position())
                if dist < 5000.0:  # Within sensing/comm range
                    tcpa, dcpa, cpa_dist = cls.compute_cpa(st_i, st_j)
                    edges.append([i, j])
                    edge_feats.append(
                        [dist / 5000.0, tcpa / 600.0, dcpa / 1000.0, cpa_dist / 1000.0]
                    )
                    edge_types.append("proximity")

        edge_index = (
            np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        )
        edge_features = (
            np.array(edge_feats, dtype=np.float32)
            if edge_feats
            else np.zeros((0, 4), dtype=np.float32)
        )

        return EncounterGraph(
            timestamp=timestamp,
            node_features=node_feats,
            vessel_ids=v_ids,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_types=edge_types,
        )
