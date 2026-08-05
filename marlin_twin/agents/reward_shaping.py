"""Fine-grained COLREGs reward shaping."""

from marlin_twin.data_classes import (
    VesselState,
    VesselAction,
    Encounter,
    EncounterType,
)


class COLREGsRewardShaper:
    """
    Computes fine-grained COLREGs rewards including Rule 17 stand-on behavior.
    """

    @staticmethod
    def compute_reward(
        state: VesselState,
        action: VesselAction,
        encounters: list[Encounter],
        w_safety: float = 2.0,
        w_colregs: float = 1.0,
        w_efficiency: float = 1.0,
    ) -> float:
        r_safety = 0.0
        r_colregs = 0.0

        for e in encounters:
            if e.vessel_i != state.vessel_id and e.vessel_j != state.vessel_id:
                continue

            # Safety penalty for small CPA / Collision
            if e.cpa_distance < 50.0:
                r_safety -= 100.0  # Collision penalty
            elif e.cpa_distance < 500.0:
                r_safety -= (500.0 - e.cpa_distance) / 50.0

            # Rule 17 Stand-on Logic
            if e.encounter_type == EncounterType.CROSSING_STAND_ON:
                if e.tcpa > 120.0 and abs(action.rudder_angle) > 0.05:
                    r_colregs -= 5.0  # Premature deviation penalty
                elif e.tcpa <= 60.0 and abs(action.rudder_angle) >= 0.1:
                    r_colregs += 2.0  # Emergency evasive maneuver reward

        # Efficiency & Waypoint Progress
        r_efficiency = state.speed / 15.0

        return float(w_safety * r_safety + w_colregs * r_colregs + w_efficiency * r_efficiency)
