"""Deterministic rule-based COLREGs collision-avoidance controller."""

import numpy as np
from marlin_twin.data_classes import VesselObservation, EncounterType
from marlin_twin.envs.colregs import COLREGsEngine


class RuleBasedCOLREGsController:
    """Deterministic Rule-Based COLREGs Collision Avoidance Controller.

    `act()` returns a `[-1,1]` tanh-space action vector -- the same convention
    every learned policy emits and the only convention `VesselAgentWrapper
    .build_action` (`agents/vessel_agent.py`) knows how to interpret. This
    controller used to return physical values (rpm as a literal fraction,
    rudder in literal radians) directly; every evaluation path fed those
    straight into `build_action`, which reinterpreted them as tanh-space and
    silently reran a different maneuver than the one specified below (rpm=0.8
    became "always full throttle", the intended 15 deg rudder became ~7.85
    deg). The mapping below is the exact inverse of `build_action`'s
    `rpm=clip(a*0.5+0.6, 0.2, 1.0)` / `rudder=clip(a*(pi/6), -pi/6, pi/6)`, so
    the physical maneuver this class specifies is the one that is actually
    executed.
    """

    def __init__(self, vessel_id: int):
        self.vessel_id = vessel_id

    def act(
        self, observation: VesselObservation, graph=None, node_idx=None, deterministic: bool = True
    ) -> np.ndarray:
        rudder_rad = 0.0
        rpm_frac = 0.8

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

            if enc_type in [
                EncounterType.HEAD_ON,
                EncounterType.CROSSING_GIVE_WAY,
                EncounterType.OVERTAKING,
            ]:
                rudder_rad = np.pi / 12  # Alter course 15 deg to starboard (Rule 14/15/13)
            elif enc_type == EncounterType.CROSSING_STAND_ON:
                rudder_rad = 0.0  # Hold course
                if min_cpa < 300.0:  # Emergency evasion
                    rudder_rad = np.pi / 6
            # OVERTAKEN (being overtaken): stand-on duty, hold course.

        # Invert build_action's rpm=clip(a*0.5+0.6, 0.2, 1.0) and
        # rudder=clip(a*(pi/6), -pi/6, pi/6) so the physical values above are
        # what actually reaches the environment.
        rpm_action = (rpm_frac - 0.6) / 0.5
        rudder_action = rudder_rad / (np.pi / 6)

        return np.array([rpm_action, rudder_action], dtype=np.float32)

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict) -> None:
        pass
