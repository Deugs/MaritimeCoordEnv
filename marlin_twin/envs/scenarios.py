# ============================================================================
# FILE: marlin_twin/envs/scenarios.py
# ============================================================================

import numpy as np
from marlin_twin.data_classes import (
    VesselState, VesselDynamics, VesselSpecification, VesselType,
    Waypoint, Route, VesselAgent, MaritimeScene, EnvironmentCondition,
    MaritimeCommunicationChannel, MaritimeDigitalTwin
)

class ScenarioGenerator:
    """Generates open water, channel navigation, and port approach scenario layouts."""

    @staticmethod
    def create_scenario(scenario_type: str = "channel", n_vessels: int = 5, seed: int | None = None) -> dict[int, VesselAgent]:
        if seed is not None:
            np.random.seed(seed)

        agents = {}
        for i in range(n_vessels):
            vtype = VesselType.CARGO if i % 2 == 0 else VesselType.USV
            spec = VesselSpecification(
                vessel_id=i, name=f"Vessel_{i}", vessel_type=vtype,
                length=150.0 if vtype == VesselType.CARGO else 30.0,
                beam=25.0 if vtype == VesselType.CARGO else 8.0,
                draft=8.0 if vtype == VesselType.CARGO else 2.0,
                max_speed=12.0, turning_circle=400.0
            )

            dynamics = VesselDynamics(
                vessel_id=i, vessel_type=vtype,
                mass=15000000.0 if vtype == VesselType.CARGO else 500000.0,
                moment_of_inertia=2e9 if vtype == VesselType.CARGO else 5e7,
                X_u_dot=-50000.0, Y_v_dot=-100000.0, N_r_dot=-500000.0,
                X_u=-1000.0, Y_v=-5000.0, N_r=-20000.0,
                propeller_diameter=4.0, max_rpm=150.0, rudder_area=20.0
            )

            if scenario_type == "open_water":
                angle = 2 * np.pi * i / n_vessels
                radius = 2000.0
                start_x = float(radius * np.cos(angle))
                start_y = float(radius * np.sin(angle))
                heading = float((angle + np.pi) % (2 * np.pi) - np.pi)
                target_x, target_y = -start_x, -start_y
            else:  # Channel navigation (traffic lanes)
                lane = i % 2
                start_x = -2000.0 if lane == 0 else 2000.0
                start_y = float(i * 300.0 - (n_vessels * 150.0))
                heading = float(0.5 * np.pi if lane == 0 else -0.5 * np.pi)
                target_x = 2000.0 if lane == 0 else -2000.0
                target_y = start_y

            state = VesselState(
                vessel_id=i, x=start_x, y=start_y, heading=heading, speed=8.0, surge_velocity=8.0
            )

            waypoints = [
                Waypoint(waypoint_id=0, x=start_x, y=start_y, speed=8.0),
                Waypoint(waypoint_id=1, x=target_x, y=target_y, speed=8.0)
            ]
            route = Route(vessel_id=i, waypoints=waypoints)

            agent = VesselAgent(
                vessel_id=i, specification=spec, dynamics=dynamics,
                current_state=state, current_route=route
            )
            agents[i] = agent

        return agents
