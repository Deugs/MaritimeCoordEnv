"""COLREGs encounter classification and compliance checking (Rules 13-18)."""

import numpy as np
from marlin_twin.data_classes import (
    VesselState,
    VesselAction,
    EncounterType,
    COLREGsRule,
)


class COLREGsEngine:
    """
    Maritime International Regulations for Preventing Collisions at Sea (COLREGs)
    Encounter Classifier and Compliance Checking Engine (Rules 13-18).
    """

    @staticmethod
    def classify_encounter(
        state_i: VesselState, state_j: VesselState, cpa_dist: float
    ) -> tuple[EncounterType, COLREGsRule | None]:
        """
        Classify encounter geometry between vessel_i and vessel_j based on relative bearing.
        Bearing is relative to vessel_i's heading.
        """
        rel_pos = state_j.position() - state_i.position()
        dist = np.linalg.norm(rel_pos)
        if dist > 5556.0 or cpa_dist > 1852.0:  # 3 nautical miles / 1 nautical mile CPA
            return EncounterType.NO_ENCOUNTER, None

        # Relative bearing from i to j (0 = dead ahead, pi/2 = starboard, -pi/2 = port)
        angle_to_j = np.arctan2(rel_pos[0], rel_pos[1])
        rel_bearing = (angle_to_j - state_i.heading + np.pi) % (2 * np.pi) - np.pi

        # Target j's heading relative to i
        rel_heading = (state_j.heading - state_i.heading + np.pi) % (2 * np.pi) - np.pi

        # Rule 13: Overtaking (approaching > 22.5 deg abaft the beam -> |rel_bearing| > 112.5 deg)
        if abs(rel_bearing) > np.radians(112.5):
            if state_i.speed > state_j.speed:
                return EncounterType.OVERTAKING, COLREGsRule.RULE_13_OVERTAKING
            else:
                return EncounterType.OVERTAKEN, COLREGsRule.RULE_13_OVERTAKING

        # Rule 14: Head-on (reciprocal courses within 15 deg, meeting nearly dead ahead)
        if abs(rel_bearing) < np.radians(15.0) and abs(abs(rel_heading) - np.pi) < np.radians(15.0):
            return EncounterType.HEAD_ON, COLREGsRule.RULE_14_HEAD_ON

        # Rule 15: Crossing
        if rel_bearing > 0:  # Other vessel is on starboard side -> give way
            return EncounterType.CROSSING_GIVE_WAY, COLREGsRule.RULE_15_CROSSING
        else:  # Other vessel is on port side -> stand on
            return EncounterType.CROSSING_STAND_ON, COLREGsRule.RULE_17_STAND_ON

    @staticmethod
    def evaluate_compliance(
        state_i: VesselState,
        action_i: VesselAction,
        state_j: VesselState,
        encounter_type: EncounterType,
        tcpa: float,
    ) -> float:
        """
        Evaluate COLREGs compliance score (0.0 to 1.0) and penalty.
        - Give-way vessel must alter course to starboard early.
        - Stand-on vessel must maintain course/speed unless emergency (Rule 17).
        """
        if encounter_type == EncounterType.NO_ENCOUNTER:
            return 1.0

        rudder = action_i.rudder_angle

        if encounter_type in [
            EncounterType.HEAD_ON,
            EncounterType.CROSSING_GIVE_WAY,
            EncounterType.OVERTAKING,
        ]:
            # Give-way responsibility: alter course to starboard (rudder > 0)
            if tcpa < 300.0:
                if rudder >= 0.05:  # Starboard alteration
                    return 1.0
                elif abs(rudder) < 0.05:
                    return 0.5  # Inaction
                else:
                    return 0.0  # Violation (turning to port)

        elif encounter_type == EncounterType.CROSSING_STAND_ON:
            # Rule 17: Stand-on vessel holds course unless tcpa < emergency threshold
            if tcpa > 60.0:
                if abs(rudder) < 0.1:
                    return 1.0  # Holding course
                else:
                    return 0.3  # Premature deviation penalty
            else:  # Emergency evasion
                if abs(rudder) >= 0.1:
                    return 1.0  # Proper emergency action

        return 1.0

    @staticmethod
    def flip_role(encounter_type: EncounterType) -> EncounterType:
        """
        Map an encounter type classified from vessel_i's perspective to the
        corresponding type for vessel_j, whose give-way/stand-on role is the
        opposite of vessel_i's in a crossing/overtaking encounter (head-on
        responsibility is symmetric, so it is unchanged).
        """
        flipped = {
            EncounterType.CROSSING_GIVE_WAY: EncounterType.CROSSING_STAND_ON,
            EncounterType.CROSSING_STAND_ON: EncounterType.CROSSING_GIVE_WAY,
            EncounterType.OVERTAKING: EncounterType.OVERTAKEN,
            EncounterType.OVERTAKEN: EncounterType.OVERTAKING,
        }
        return flipped.get(encounter_type, encounter_type)
