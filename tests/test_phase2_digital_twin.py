# ============================================================================
# FILE: tests/test_phase2_digital_twin.py
# ============================================================================

import pytest
import numpy as np
from marlin_twin.data_classes import (
    VesselState, AISReading, RadarTrack, MaritimeMessage, MessagePriority
)
from marlin_twin.envs.sensors import SensorSimulator
from marlin_twin.envs.digital_twin import DigitalTwinEstimator
from marlin_twin.envs.communication import CommunicationChannelManager

def test_sensor_simulator_noise_and_sweeps():
    state = VesselState(vessel_id=0, x=100.0, y=200.0, heading=0.0, speed=10.0)
    ais = SensorSimulator.generate_ais(state, timestamp=0.0, drop_prob=0.0)
    assert ais is not None
    assert abs(ais.reported_position[0] - 100.0) < 30.0
    assert abs(ais.reported_position[1] - 200.0) < 30.0

    radar = SensorSimulator.generate_radar(state, timestamp=0.0, track_id=101)
    assert radar.track_id == 101

    scene_states = {0: state, 1: VesselState(vessel_id=1, x=500.0, y=500.0, heading=np.pi, speed=8.0)}
    ais_dict, radar_tracks = SensorSimulator.generate_scene_sensors(scene_states, timestamp=1.0)
    assert len(radar_tracks) == 2

def test_digital_twin_ekf_and_jpda_fallback():
    estimator = DigitalTwinEstimator()
    vessels = {
        0: VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=10.0),
        1: VesselState(vessel_id=1, x=500.0, y=500.0, heading=np.pi, speed=10.0)
    }

    # Step 1: Full AIS Available
    ais_dict, radar_tracks = SensorSimulator.generate_scene_sensors(vessels, timestamp=0.0, ais_drop_prob=0.0)
    twin = estimator.update("scene_0", 0.0, vessels, ais_dict, radar_tracks)
    assert len(twin.vessel_estimates) == 2
    assert twin.vessel_estimates[0].estimation_method == "kalman_ais"

    # Step 2: Total AIS Loss (JPDA + Dead Reckoning)
    twin2 = estimator.update("scene_0", 1.0, vessels, {}, radar_tracks)
    assert twin2.vessel_estimates[0].estimation_method in ["jpda_radar", "jpda_dead_reckoning"]

def test_communication_priority_and_degradation():
    comm = CommunicationChannelManager(bandwidth_bps=9600.0)
    
    msg_critical = MaritimeMessage(
        sender_id=0, receiver_id=1, content="COLLISION_ALERT",
        priority=MessagePriority.CRITICAL, timestamp=0.0, size_bits=400.0
    )
    msg_low = MaritimeMessage(
        sender_id=0, receiver_id=1, content="ROUTINE_TELEMETRY",
        priority=MessagePriority.LOW, timestamp=0.0, size_bits=8000.0
    )

    delivered = comm.process_step([msg_low, msg_critical])
    assert len(delivered) >= 1
    assert delivered[0].priority == MessagePriority.CRITICAL

    # Test bandwidth degradation limit
    comm.set_degradation(0.1)  # 10% capacity
    assert comm.channel.bandwidth_bps == pytest.approx(960.0)
