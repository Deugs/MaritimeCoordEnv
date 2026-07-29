# ============================================================================
# FILE: marlin_twin/envs/vessel_dynamics.py
# ============================================================================

import numpy as np
from marlin_twin.data_classes import (
    VesselState, VesselAction, VesselDynamics, EnvironmentCondition
)

class MMGDynamicsSolver:
    """
    3-DOF MMG (Mathematical Maneuvering Group) Hydrodynamic Dynamics Solver.
    Integrates surge u, sway v, and yaw rate r using 4th-order Runge-Kutta (RK4).
    """

    def __init__(self, dynamics: VesselDynamics):
        self.dynamics = dynamics

    def step(
        self,
        state: VesselState,
        action: VesselAction,
        dt: float = 1.0,
        environment: EnvironmentCondition = EnvironmentCondition.CLEAR
    ) -> VesselState:
        """Integrate state forward by dt using RK4 integration."""
        target_rpm = action.propeller_rpm * self.dynamics.max_rpm
        target_rudder = np.clip(
            action.rudder_angle,
            -self.dynamics.max_rudder_angle,
            self.dynamics.max_rudder_angle
        )

        def f(s: VesselState) -> np.ndarray:
            dx, dy, dheading, du, dr = self.dynamics.compute_derivatives(
                s, target_rpm, target_rudder
            )
            return np.array([dx, dy, dheading, du, dr])

        s_vec = np.array([state.x, state.y, state.heading, state.surge_velocity, state.yaw_rate])

        # RK4 Steps
        k1 = f(state)

        st_k2 = VesselState(
            vessel_id=state.vessel_id,
            x=state.x + 0.5 * dt * k1[0],
            y=state.y + 0.5 * dt * k1[1],
            heading=state.heading + 0.5 * dt * k1[2],
            speed=state.speed + 0.5 * dt * k1[3],
            surge_velocity=state.surge_velocity + 0.5 * dt * k1[3],
            yaw_rate=state.yaw_rate + 0.5 * dt * k1[4]
        )
        k2 = f(st_k2)

        st_k3 = VesselState(
            vessel_id=state.vessel_id,
            x=state.x + 0.5 * dt * k2[0],
            y=state.y + 0.5 * dt * k2[1],
            heading=state.heading + 0.5 * dt * k2[2],
            speed=state.speed + 0.5 * dt * k2[3],
            surge_velocity=state.surge_velocity + 0.5 * dt * k2[3],
            yaw_rate=state.yaw_rate + 0.5 * dt * k2[4]
        )
        k3 = f(st_k3)

        st_k4 = VesselState(
            vessel_id=state.vessel_id,
            x=state.x + dt * k3[0],
            y=state.y + dt * k3[1],
            heading=state.heading + dt * k3[2],
            speed=state.speed + dt * k3[3],
            surge_velocity=state.surge_velocity + dt * k3[3],
            yaw_rate=state.yaw_rate + dt * k3[4]
        )
        k4 = f(st_k4)

        new_vec = s_vec + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        new_heading = (new_vec[2] + np.pi) % (2 * np.pi) - np.pi
        new_speed = max(0.0, new_vec[3])

        return VesselState(
            vessel_id=state.vessel_id,
            x=float(new_vec[0]),
            y=float(new_vec[1]),
            heading=float(new_heading),
            speed=float(new_speed),
            surge_velocity=float(new_speed),
            sway_velocity=state.sway_velocity,
            yaw_rate=float(new_vec[4])
        )
