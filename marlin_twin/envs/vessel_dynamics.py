"""3-DOF MMG hydrodynamic vessel dynamics solver (RK4 integration)."""

import numpy as np
from marlin_twin.data_classes import VesselState, VesselAction, VesselDynamics, EnvironmentCondition


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
        environment: EnvironmentCondition = EnvironmentCondition.CLEAR,
    ) -> VesselState:
        """Integrate state forward by dt using RK4 integration."""
        # `compute_derivatives` multiplies its `propeller_rpm` argument by
        # `self.max_rpm` itself to get real RPM, so passing the fraction
        # straight through here (not pre-multiplying) avoids scaling by
        # max_rpm twice -- a bug that was inflating thrust by ~150-180x.
        target_rpm = action.propeller_rpm
        target_rudder = np.clip(
            action.rudder_angle, -self.dynamics.max_rudder_angle, self.dynamics.max_rudder_angle
        )

        def f(s: VesselState) -> np.ndarray:
            dx, dy, dheading, du, dv, dr = self.dynamics.compute_derivatives(
                s, target_rpm, target_rudder
            )
            return np.array([dx, dy, dheading, du, dv, dr])

        s_vec = np.array(
            [
                state.x,
                state.y,
                state.heading,
                state.surge_velocity,
                state.sway_velocity,
                state.yaw_rate,
            ]
        )

        # RK4 Steps
        k1 = f(state)

        st_k2 = VesselState(
            vessel_id=state.vessel_id,
            x=state.x + 0.5 * dt * k1[0],
            y=state.y + 0.5 * dt * k1[1],
            heading=state.heading + 0.5 * dt * k1[2],
            speed=state.speed + 0.5 * dt * k1[3],
            surge_velocity=state.surge_velocity + 0.5 * dt * k1[3],
            sway_velocity=state.sway_velocity + 0.5 * dt * k1[4],
            yaw_rate=state.yaw_rate + 0.5 * dt * k1[5],
        )
        k2 = f(st_k2)

        st_k3 = VesselState(
            vessel_id=state.vessel_id,
            x=state.x + 0.5 * dt * k2[0],
            y=state.y + 0.5 * dt * k2[1],
            heading=state.heading + 0.5 * dt * k2[2],
            speed=state.speed + 0.5 * dt * k2[3],
            surge_velocity=state.surge_velocity + 0.5 * dt * k2[3],
            sway_velocity=state.sway_velocity + 0.5 * dt * k2[4],
            yaw_rate=state.yaw_rate + 0.5 * dt * k2[5],
        )
        k3 = f(st_k3)

        st_k4 = VesselState(
            vessel_id=state.vessel_id,
            x=state.x + dt * k3[0],
            y=state.y + dt * k3[1],
            heading=state.heading + dt * k3[2],
            speed=state.speed + dt * k3[3],
            surge_velocity=state.surge_velocity + dt * k3[3],
            sway_velocity=state.sway_velocity + dt * k3[4],
            yaw_rate=state.yaw_rate + dt * k3[5],
        )
        k4 = f(st_k4)

        new_vec = s_vec + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        new_heading = (new_vec[2] + np.pi) % (2 * np.pi) - np.pi
        new_surge = max(0.0, new_vec[3])
        new_sway = float(new_vec[4])
        new_speed = float(np.sqrt(new_surge**2 + new_sway**2))

        return VesselState(
            vessel_id=state.vessel_id,
            x=float(new_vec[0]),
            y=float(new_vec[1]),
            heading=float(new_heading),
            speed=new_speed,
            surge_velocity=float(new_surge),
            sway_velocity=new_sway,
            yaw_rate=float(new_vec[5]),
        )

    def run_turning_circle_test(
        self,
        initial_state: VesselState | None = None,
        rudder_angle_deg: float = 35.0,
        duration: float = 600.0,
        dt: float = 1.0,
    ) -> dict[str, float | bool | list[VesselState]]:
        """
        Executes standard IMO Turning Circle Sea Trial.
        Measures Tactical Diameter, Advance, and Transfer.

        `tactical_diameter` is the max lateral deviation over the vessel's
        *first complete 360-degree loop* (cumulative heading change reaching
        2*pi), not over the whole `duration` window -- measuring over the
        whole window would keep growing for as long as the run continues
        past one closed loop, independent of the actual circle's size. If
        the vessel hasn't completed a full loop by `duration` (a sluggish
        vessel/weak rudder relative to `duration`), falls back to the
        max-lateral-deviation-so-far with `loop_completed=False` so callers
        can tell a real measurement from a truncated one, rather than
        silently reporting a duration-dependent number as if it were the
        circle's true diameter.
        """
        state = initial_state or VesselState(
            vessel_id=0,
            x=0.0,
            y=0.0,
            heading=0.0,
            speed=self.dynamics.max_rpm * 0.1,
            surge_velocity=self.dynamics.max_rpm * 0.1,
        )
        rudder_rad = np.radians(rudder_angle_deg)
        action = VesselAction(
            vessel_id=state.vessel_id,
            propeller_rpm=0.8,
            rudder_angle=rudder_rad,
            message_targets=[],
        )

        trajectory = [state]
        max_y = 0.0
        advance = 0.0
        transfer = 0.0
        found_90deg = False
        loop_completed = False
        loop_completed_step = None
        cumulative_turn = 0.0
        prev_heading = state.heading

        n_steps = int(duration / dt)
        for step_idx in range(n_steps):
            state = self.step(state, action, dt)
            trajectory.append(state)

            step_turn = (state.heading - prev_heading + np.pi) % (2 * np.pi) - np.pi
            cumulative_turn += step_turn
            prev_heading = state.heading

            if not loop_completed:
                max_y = max(max_y, abs(state.y))

            # Check 90 deg heading change
            heading_diff = abs(
                (state.heading - trajectory[0].heading + np.pi) % (2 * np.pi) - np.pi
            )
            if not found_90deg and heading_diff >= np.pi / 2:
                advance = state.x
                transfer = abs(state.y)
                found_90deg = True

            if not loop_completed and abs(cumulative_turn) >= 2 * np.pi:
                loop_completed = True
                loop_completed_step = step_idx + 1

        tactical_diameter = max_y * 2.0
        if not found_90deg:
            advance = max([s.x for s in trajectory])
            transfer = max_y

        return {
            "tactical_diameter": float(tactical_diameter),
            "advance": float(advance),
            "transfer": float(transfer),
            "loop_completed": loop_completed,
            "loop_completed_step": loop_completed_step,
            "trajectory": trajectory,
        }

    def run_zigzag_test(
        self,
        initial_state: VesselState | None = None,
        angle_deg: float = 10.0,
        duration: float = 400.0,
        dt: float = 1.0,
    ) -> dict[str, float | bool | list[VesselState] | None]:
        """
        Executes standard IMO 10/10 or 20/20 Zig-zag Sea Trial.
        Measures First and Second Yaw Overshoot Angles.

        A vessel that is too sluggish (or given too short a `duration`) for
        this rudder angle may never complete the first and/or second
        overshoot within the test window. `first_overshoot_angle`/
        `second_overshoot_angle` are `None` in that case -- check
        `first_overshoot_converged`/`second_overshoot_converged` before
        trusting either value. Do not substitute a placeholder number for
        a maneuver that didn't actually complete; that would misrepresent
        a non-convergent run as a real sea-trial measurement.
        """
        state = initial_state or VesselState(
            vessel_id=0, x=0.0, y=0.0, heading=0.0, speed=8.0, surge_velocity=8.0
        )
        angle_rad = np.radians(angle_deg)
        current_rudder = angle_rad

        trajectory = [state]
        heading_history = [state.heading]
        overshoots = []

        target_heading_change = angle_rad
        phase = 1
        max_heading_phase = 0.0

        n_steps = int(duration / dt)
        for _ in range(n_steps):
            action = VesselAction(
                vessel_id=state.vessel_id,
                propeller_rpm=0.8,
                rudder_angle=current_rudder,
                message_targets=[],
            )
            state = self.step(state, action, dt)
            trajectory.append(state)
            heading_history.append(state.heading)

            # Signed, wrap-corrected relative heading -- `state.heading` is
            # wrapped to [-pi, pi] every step (see `step()`), so a raw
            # `state.heading - trajectory[0].heading` difference (used here
            # in an earlier version of this method for the phase 2/3 exit
            # checks) breaks discontinuously the moment a fast-turning
            # vessel's true relative heading passes +-180 degrees -- exactly
            # the regime a vessel tight enough to meet the IMO turning-circle
            # criterion operates in. Compute it once, signed, and reuse it
            # for every phase's crossing check.
            rel_signed = (state.heading - trajectory[0].heading + np.pi) % (2 * np.pi) - np.pi
            rel_heading = abs(rel_signed)
            max_heading_phase = max(max_heading_phase, rel_heading)

            if phase == 1 and rel_signed >= target_heading_change:
                # First execute completed, reverse rudder
                current_rudder = -angle_rad
                phase = 2
            elif phase == 2:
                if rel_signed <= -target_heading_change:
                    overshoots.append(float(np.degrees(max_heading_phase - target_heading_change)))
                    current_rudder = angle_rad
                    phase = 3
                    max_heading_phase = 0.0
            elif phase == 3:
                if rel_signed >= target_heading_change:
                    overshoots.append(float(np.degrees(max_heading_phase - target_heading_change)))
                    phase = 4

        first_overshoot = overshoots[0] if len(overshoots) > 0 else None
        second_overshoot = overshoots[1] if len(overshoots) > 1 else None

        return {
            "first_overshoot_angle": first_overshoot,
            "second_overshoot_angle": second_overshoot,
            "first_overshoot_converged": len(overshoots) > 0,
            "second_overshoot_converged": len(overshoots) > 1,
            "trajectory": trajectory,
        }
