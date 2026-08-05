"""Deterministic rule-based COLREGs collision-avoidance controller."""

import numpy as np
from marlin_twin.data_classes import VesselObservation, EncounterType
from marlin_twin.envs.colregs import COLREGsEngine


class RuleBasedCOLREGsController:
    """Deterministic Rule-Based COLREGs Collision Avoidance Controller."""

    def __init__(self, vessel_id: int):
        self.vessel_id = vessel_id

    def act(self, observation: VesselObservation, deterministic: bool = True) -> np.ndarray:
        rudder = 0.0
        rpm = 0.8

        own_state = observation.own_state
        min_cpa = 5000.0
        most_dangerous_neighbor = None

        for nid, nstate in observation.neighbor_states.items():
            dist = np.linalg.norm(nstate.position() - own_state.position())
            if dist < min_cpa:
                min_cpa = dist
                most_dangerous_neighbor = nstate

        if most_dangerous_neighbor and min_cpa < 2000.0:
            enc_type, rule = COLREGsEngine.classify_encounter(
                own_state, most_dangerous_neighbor, min_cpa
            )

            if enc_type in [EncounterType.HEAD_ON, EncounterType.CROSSING_GIVE_WAY]:
                rudder = np.pi / 12  # Alter course 15 deg to starboard
            elif enc_type == EncounterType.CROSSING_STAND_ON:
                rudder = 0.0  # Hold course
                if min_cpa < 300.0:  # Emergency evasion
                    rudder = np.pi / 6

        return np.array([rpm, rudder], dtype=np.float32)

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict) -> None:
        pass
