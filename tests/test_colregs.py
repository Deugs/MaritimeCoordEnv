import numpy as np
from marlin_twin.data_classes import VesselState, EncounterType, COLREGsRule
from marlin_twin.envs.colregs import COLREGsEngine


def test_head_on_classification():
    v1 = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=10.0)
    v2 = VesselState(vessel_id=1, x=0.0, y=1000.0, heading=np.pi, speed=10.0)

    enc_type, rule = COLREGsEngine.classify_encounter(v1, v2, cpa_dist=0.0)
    assert enc_type == EncounterType.HEAD_ON
    assert rule == COLREGsRule.RULE_14_HEAD_ON


def test_crossing_give_way_classification():
    v1 = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=10.0)
    v2 = VesselState(vessel_id=1, x=500.0, y=500.0, heading=-np.pi / 2, speed=10.0)

    enc_type, rule = COLREGsEngine.classify_encounter(v1, v2, cpa_dist=100.0)
    assert enc_type == EncounterType.CROSSING_GIVE_WAY
    assert rule == COLREGsRule.RULE_15_CROSSING
