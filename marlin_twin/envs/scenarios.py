"""Scenario generator for open-water, channel, port-approach, and encounter layouts."""

import numpy as np
from marlin_twin.data_classes import (
    VesselState,
    VesselDynamics,
    VesselSpecification,
    VesselType,
    Waypoint,
    Route,
    VesselAgent,
)
from marlin_twin.envs.vessel_profiles import VESSEL_PROFILES
from marlin_twin.utils.seeding import seed_everything


class ScenarioGenerator:
    """Generates open water, channel navigation, and port approach scenario layouts."""

    @staticmethod
    def create_scenario(
        scenario_type: str = "channel",
        n_vessels: int = 5,
        seed: int | None = None,
        vessel_types: list[VesselType] | None = None,
    ) -> dict[int, VesselAgent]:
        """`vessel_types`, if given, is cycled across the fleet (vessel `i`
        gets `vessel_types[i % len(vessel_types)]`) and drawn from
        `VESSEL_PROFILES` for genuinely distinct per-type hydrodynamics —
        `None` (the default) keeps the original CARGO/USV-by-parity fleet for
        backward compatibility."""
        if seed is not None:
            seed_everything(seed)

        # "crossing" is the natural name used by several eval scripts, but the
        # branch below is keyed on "crossing_give_way" — without this alias it
        # silently falls through to the channel layout instead of erroring.
        if scenario_type == "crossing":
            scenario_type = "crossing_give_way"

        # These three named experimental scenarios differentiate purely at
        # the config level (fleet mix, weather, comms schedule — set by the
        # caller, e.g. the sweep driver) rather than geometry, so they alias
        # onto the existing layout that already fits their purpose:
        # - congested_port_approach: port_approach run with denser n_vessels
        #   and a genuinely mixed fleet (see `vessel_types` above).
        # - restricted_visibility_crossing: crossing_give_way run under
        #   `MaritimeExperimentConfig.environment_condition=FOG`.
        # - comms_blackout_transit: open_water run with a
        #   `MaritimeExperimentConfig.comms_schedule` blackout/jamming window.
        if scenario_type == "congested_port_approach":
            scenario_type = "port_approach"
        elif scenario_type == "restricted_visibility_crossing":
            scenario_type = "crossing_give_way"
        elif scenario_type == "comms_blackout_transit":
            scenario_type = "open_water"

        agents = {}
        for i in range(n_vessels):
            if vessel_types:
                vtype = vessel_types[i % len(vessel_types)]
            else:
                vtype = VesselType.CARGO if i % 2 == 0 else VesselType.USV
            profile = VESSEL_PROFILES[vtype]

            spec = VesselSpecification(
                vessel_id=i,
                name=f"Vessel_{i}",
                vessel_type=vtype,
                length=profile["length"],
                beam=profile["beam"],
                draft=profile["draft"],
                max_speed=profile["max_speed"],
                turning_circle=profile["turning_circle"],
            )

            dynamics = VesselDynamics(
                vessel_id=i,
                vessel_type=vtype,
                mass=profile["mass"],
                moment_of_inertia=profile["moment_of_inertia"],
                X_u_dot=profile["X_u_dot"],
                Y_v_dot=profile["Y_v_dot"],
                N_r_dot=profile["N_r_dot"],
                X_u=profile["X_u"],
                Y_v=profile["Y_v"],
                N_r=profile["N_r"],
                propeller_diameter=profile["propeller_diameter"],
                max_rpm=profile["max_rpm"],
                rudder_area=profile["rudder_area"],
                thrust_coefficient=profile["thrust_coefficient"],
                yaw_coefficient=profile["yaw_coefficient"],
            )

            if scenario_type == "head_on" and n_vessels >= 2:
                # Vessels are grouped into reciprocal-course pairs; each pair
                # is offset onto its own lane so multiple simultaneous
                # head-on encounters don't collide with each other. A leftover
                # unpaired vessel (odd n_vessels) gets role 0 of its own
                # lane — a neutral leg with no forced encounter, not an error.
                #
                # Separation of 150m/side (300m gap) -- not the earlier
                # 800m/side (1600m gap). That 1600m gap was tuned for the
                # CARGO/USV N_r/yaw_coefficient values in place before the
                # IMO turning-circle fix (see VesselDynamics.N_r's
                # docstring), whose yaw response took on the order of
                # 1000s to develop meaningfully. The fix raised both by
                # several orders of magnitude specifically so the vessel
                # can turn tightly enough to meet the 5*L turning-circle
                # ceiling, which also means it can react to an oncoming
                # vessel almost immediately -- at the old 1600m gap, every
                # policy (including Rule-Based COLREGs) now starts
                # avoiding early enough to clear 1000m+ of separation,
                # saturating `compute_safety_score`'s 1000m-normalized
                # ceiling for every method and destroying its ability to
                # differentiate them (empirically confirmed: true minimum
                # pairwise distance exceeded 1250m for both Rule-Based and
                # MARLIN-Twin at the old gap under the corrected dynamics).
                # The tighter 300m gap reliably reproduces a real ~100-150m
                # closest approach with a rule-based avoidance policy
                # within a 400-600 step episode, restoring real dynamic
                # range to the safety score under the corrected dynamics.
                pair_idx, role = i // 2, i % 2
                lane_x = pair_idx * 150.0
                if role == 0:
                    start_x, start_y, heading = lane_x, -150.0, 0.0  # Heading North
                    target_x, target_y = lane_x, 150.0
                else:
                    start_x, start_y, heading = lane_x, 150.0, np.pi  # Heading South
                    target_x, target_y = lane_x, -150.0
            elif scenario_type == "crossing_give_way" and n_vessels >= 2:
                # Same pair-grouping as head_on, offset diagonally per pair so
                # each pair's crossing point is spatially distinct.
                pair_idx, role = i // 2, i % 2
                center_x = center_y = pair_idx * 1000.0
                if role == 0:
                    start_x, start_y, heading = center_x, center_y - 1500.0, 0.0
                    target_x, target_y = center_x, center_y + 1500.0
                else:
                    start_x, start_y, heading = (
                        center_x + 1500.0,
                        center_y,
                        -np.pi / 2,
                    )  # Heading West (On starboard side)
                    target_x, target_y = center_x - 1500.0, center_y
            elif scenario_type == "overtaking" and n_vessels >= 2:
                pair_idx, role = i // 2, i % 2
                lane_x = pair_idx * 800.0
                if role == 0:  # Slower target ship ahead
                    start_x, start_y, heading = lane_x, 500.0, 0.0
                    target_x, target_y = lane_x, 3000.0
                else:  # Faster overtaking ship behind
                    start_x, start_y, heading = lane_x, -1000.0, 0.0
                    target_x, target_y = lane_x, 3000.0
            elif scenario_type == "multi_vessel_channel_convergence":
                # A denser two-way channel than the standard "channel" layout
                # below — tighter lane separation and tighter longitudinal
                # spacing so several vessels converge and pass each other
                # concurrently, exercising EncounterManager under multiple
                # simultaneous pairwise encounters rather than one isolated
                # pair at a time.
                lane = i % 2
                start_x = -1000.0 if lane == 0 else 1000.0
                start_y = float(i * 150.0 - (n_vessels * 75.0))
                heading = float(0.5 * np.pi if lane == 0 else -0.5 * np.pi)
                target_x = 1000.0 if lane == 0 else -1000.0
                target_y = start_y
            elif scenario_type == "port_approach":
                # Converging multi-lane port approach layout
                sector = i % 3
                dist = 3000.0 + (i // 3) * 400.0
                angle = (sector * (2 * np.pi / 3)) + (np.random.rand() - 0.5) * 0.2
                start_x = float(dist * np.cos(angle))
                start_y = float(dist * np.sin(angle))
                # `angle` is a standard math-convention bearing (0 = +X); heading
                # uses the nautical convention (0 = +Y/North, pi/2 = +X/East), so
                # pointing back toward the origin requires -angle - pi/2, not
                # angle + pi (which would be correct only if both used the same
                # convention).
                heading = float((-angle - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi)
                target_x, target_y = 0.0, 0.0  # All heading to port center
            elif scenario_type == "open_water":
                angle = 2 * np.pi * i / n_vessels
                radius = 2000.0
                start_x = float(radius * np.cos(angle))
                start_y = float(radius * np.sin(angle))
                # Same math-convention-vs-nautical-convention correction as port_approach.
                heading = float((-angle - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi)
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
                Waypoint(waypoint_id=1, x=target_x, y=target_y, speed=8.0),
            ]
            route = Route(vessel_id=i, waypoints=waypoints)

            agent = VesselAgent(
                vessel_id=i,
                specification=spec,
                dynamics=dynamics,
                current_state=state,
                current_route=route,
            )
            agents[i] = agent

        return agents
