from marlin_twin.data_classes import (
    VesselState,
    VesselDynamics,
    VesselType,
    VesselAction,
    EnvironmentCondition,
)
from marlin_twin.envs.vessel_dynamics import MMGDynamicsSolver


def test_mmg_dynamics_step():
    dynamics = VesselDynamics(
        vessel_id=0,
        vessel_type=VesselType.CARGO,
        mass=1000000.0,
        moment_of_inertia=1e7,
        max_rpm=100.0,
        propeller_diameter=3.0,
    )
    solver = MMGDynamicsSolver(dynamics)
    state = VesselState(vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=5.0, surge_velocity=5.0)
    action = VesselAction(vessel_id=0, propeller_rpm=0.8, rudder_angle=0.1, message_targets=[])

    next_state = solver.step(state, action, dt=1.0, environment=EnvironmentCondition.CLEAR)

    assert next_state.speed >= 0.0
    assert isinstance(next_state.x, float)
    assert isinstance(next_state.y, float)
