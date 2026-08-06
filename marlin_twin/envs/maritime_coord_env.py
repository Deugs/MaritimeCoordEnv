"""Concrete multi-vessel maritime coordination Gym-style environment."""

import numpy as np
from marlin_twin.data_classes import (
    MaritimeExperimentConfig,
    MaritimeScene,
    VesselObservation,
    VesselAction,
    EnvironmentCondition,
    MaritimeMessage,
    AISReading,
)
from marlin_twin.envs.base_env import BaseMaritimeEnvironment
from marlin_twin.envs.scenarios import ScenarioGenerator
from marlin_twin.envs.vessel_dynamics import MMGDynamicsSolver
from marlin_twin.envs.colregs import COLREGsEngine
from marlin_twin.envs.encounters import EncounterManager
from marlin_twin.envs.digital_twin import DigitalTwinEstimator
from marlin_twin.envs.communication import CommunicationChannelManager
from marlin_twin.envs.sensors import SensorSimulator

# Baseline sensing/comm range under CLEAR conditions is 5000.0m (see
# `EncounterManager.build_encounter_graph`'s default `max_range`) — every
# other condition's visibility is expressed as a fraction of that baseline,
# which is what actually shrinks detection range and grows sensor noise
# below (not just a label on the observation, as it was before).
VISIBILITY_BY_CONDITION: dict[EnvironmentCondition, float] = {
    EnvironmentCondition.CLEAR: 5000.0,
    EnvironmentCondition.FOG: 800.0,
    EnvironmentCondition.RAIN: 3000.0,
    EnvironmentCondition.HIGH_WIND: 4000.0,
    EnvironmentCondition.HIGH_SEA: 3500.0,
    EnvironmentCondition.ICE: 4500.0,
}


class MaritimeCoordEnv(BaseMaritimeEnvironment):
    """
    Main Gymnasium-compatible multi-vessel maritime coordination environment.
    Integrates 3-DOF MMG dynamics, COLREGs engine, Digital Twin EKF/JPDA state estimation,
    and priority bandwidth-constrained communication.
    """

    def __init__(self, config: MaritimeExperimentConfig):
        super().__init__(config)
        self.solvers: dict[int, MMGDynamicsSolver] = {}
        self.dt_estimator = DigitalTwinEstimator()
        self.comm_manager = CommunicationChannelManager(config.bandwidth_bps, config.base_latency)
        self._visibility_range = VISIBILITY_BY_CONDITION[EnvironmentCondition.CLEAR]
        self.set_communication_schedule(config.comms_schedule)

    def reset(
        self,
        scenario_type: str | None = None,
        n_vessels: int | None = None,
        seed: int | None = None,
    ) -> tuple[dict[int, VesselObservation], dict]:
        st = scenario_type or self.config.scenario_type
        nv = n_vessels or self.config.n_vessels

        self.time_step = 0
        self._visibility_range = VISIBILITY_BY_CONDITION[self.config.environment_condition]
        vessel_types = self.config.vessel_types if self.config.heterogeneous else None
        agents = ScenarioGenerator.create_scenario(st, nv, seed, vessel_types=vessel_types)
        self.solvers = {vid: MMGDynamicsSolver(ag.dynamics) for vid, ag in agents.items()}

        states = {vid: ag.current_state for vid, ag in agents.items() if ag.current_state}
        visibility_factor = (
            self._visibility_range / VISIBILITY_BY_CONDITION[EnvironmentCondition.CLEAR]
        )
        encounters = EncounterManager.detect_encounters(
            states, max_range=5556.0 * visibility_factor, max_cpa=1852.0 * visibility_factor
        )

        dt_scene = self.dt_estimator.update("init_scene", 0.0, states, [], [])

        self.scene = MaritimeScene(
            scene_id="maritime_scene_0",
            timestamp=0.0,
            vessels=agents,
            communication_channel=self.comm_manager.channel,
            digital_twin=dt_scene,
            boundaries=self.config.boundaries,
            environment_condition=self.config.environment_condition,
        )

        if self._comms_schedule:
            # Snapshot whatever degradation/jamming was configured before
            # reset (e.g. a sweep script's `env.set_communication_degradation(lam)`)
            # as the fallback the schedule reverts to outside its windows.
            self._comms_baseline_level = self.comms_degradation_level
            self._comms_baseline_jamming_zone = self.scene.communication_channel.jamming_zone
            self._apply_comms_schedule(0.0)

        obs = self._build_observations()
        return obs, {"encounters": len(encounters)}

    def step(
        self, actions: dict[int, VesselAction]
    ) -> tuple[dict[int, VesselObservation], dict[int, float], float, bool, dict]:
        self.time_step += 1
        dt = 1.0
        self._apply_comms_schedule(float(self.time_step))

        # Step 1: Update Dynamics
        new_states = {}
        for vid, ag in self.scene.vessels.items():
            action = actions.get(
                vid,
                VesselAction(
                    vessel_id=vid, propeller_rpm=0.5, rudder_angle=0.0, message_targets=[]
                ),
            )
            new_state = self.solvers[vid].step(ag.current_state, action, dt)
            ag.current_state = new_state
            ag.last_action = action
            new_states[vid] = new_state

        # Step 2: Process Communication Messages -- moved ahead of the
        # sensor/Digital Twin update below so a successfully-delivered V2V
        # self-report can supplement (or substitute for) a dropped AIS
        # packet there. Previously this ran last and its `delivered` return
        # value was discarded entirely, so this channel's bandwidth/
        # packet-loss degradation had zero effect on anything the policy
        # could actually see.
        outgoing = []
        for vid, act in actions.items():
            for target in act.message_targets:
                outgoing.append(
                    MaritimeMessage(
                        sender_id=vid,
                        receiver_id=target,
                        content=np.array(
                            [
                                new_states[vid].x,
                                new_states[vid].y,
                                new_states[vid].heading,
                                new_states[vid].speed,
                            ]
                        ),
                        priority=act.message_priority,
                        timestamp=float(self.time_step),
                        size_bits=256,
                    )
                )

        weather = 1.0 - self.comms_degradation_level
        delivered = self.comm_manager.process_step(outgoing, weather_degradation=weather)

        # Step 3: Sensors & Digital Twin Update
        drop_prob = float(np.clip(1.0 - self.comms_degradation_level, 0.0, 0.98))
        visibility_factor = (
            self._visibility_range / VISIBILITY_BY_CONDITION[EnvironmentCondition.CLEAR]
        )
        noise_scale = 1.0 / max(visibility_factor, 1e-6)
        ais_readings = [
            SensorSimulator.generate_ais(
                s, float(self.time_step), drop_prob=drop_prob, noise_scale=noise_scale
            )
            for s in new_states.values()
        ]
        ais_by_vid = {r.vessel_id: r for r in ais_readings if r is not None}

        # A delivered V2V message is the sender reporting its own state --
        # an independent fix, on top of (or in place of) whatever AIS
        # already provided, so degraded bandwidth/packet-loss actually
        # reaches the Digital Twin's estimate quality instead of being
        # computed and discarded.
        for msg in delivered:
            if (
                msg.sender_id not in ais_by_vid
                and msg.content is not None
                and len(msg.content) >= 4
            ):
                ais_by_vid[msg.sender_id] = AISReading(
                    vessel_id=msg.sender_id,
                    timestamp=float(self.time_step),
                    reported_position=(float(msg.content[0]), float(msg.content[1])),
                    reported_heading=float(msg.content[2]),
                    reported_speed=float(msg.content[3]),
                    confidence=0.9,
                )
        ais_readings = list(ais_by_vid.values())

        # Radar is a self-contained onboard sensor, so it stays largely
        # independent of ordinary communication-bandwidth degradation
        # (unlike AIS, which needs a working transponder/channel) -- but
        # broadband RF jamming realistically can blind radar too, so a
        # vessel actively inside an active jamming zone gets a near-total
        # miss probability instead of the small ambient one. Without this,
        # radar silently backstops every dropped AIS packet every step
        # regardless of degradation level, masking any observable effect of
        # `comms_degradation_level` on perception quality.
        radar_drop_prob = float(np.clip(0.4 * (1.0 - self.comms_degradation_level), 0.0, 0.4))
        channel = self.scene.communication_channel
        jamming_zone = channel.jamming_zone if channel.jamming_active else None
        radar_tracks = []
        for idx, s in enumerate(new_states.values()):
            in_jamming_zone = False
            if jamming_zone is not None:
                zx, zy, zr = jamming_zone
                in_jamming_zone = float(np.hypot(s.x - zx, s.y - zy)) <= zr
            track = SensorSimulator.generate_radar(
                s,
                float(self.time_step),
                idx,
                noise_scale=noise_scale,
                drop_prob=0.9 if in_jamming_zone else radar_drop_prob,
            )
            if track is not None:
                radar_tracks.append(track)

        dt_scene = self.dt_estimator.update(
            f"scene_{self.time_step}", float(self.time_step), new_states, ais_readings, radar_tracks
        )
        self.scene.digital_twin = dt_scene

        # Step 4: Encounters & COLREGs Compliance — range/CPA gates shrink
        # under degraded visibility so restricted visibility genuinely
        # reduces/delays what counts as an "encounter," not just a label.
        encounters = EncounterManager.detect_encounters(
            new_states, max_range=5556.0 * visibility_factor, max_cpa=1852.0 * visibility_factor
        )

        # Step 5: Compute Rewards
        rewards = {}
        colregs_violations = 0
        for vid, ag in self.scene.vessels.items():
            # Safety reward (CPA penalty)
            min_cpa = min(
                [e.cpa_distance for e in encounters if e.vessel_i == vid or e.vessel_j == vid],
                default=5000.0,
            )
            r_safety = -np.exp(-min_cpa / 200.0)

            # COLREGs compliance reward
            r_colregs = 1.0
            for e in encounters:
                if e.vessel_i == vid:
                    other_state = new_states[e.vessel_j]
                    role_type = e.encounter_type
                elif e.vessel_j == vid:
                    other_state = new_states[e.vessel_i]
                    role_type = COLREGsEngine.flip_role(e.encounter_type)
                else:
                    continue

                score = COLREGsEngine.evaluate_compliance(
                    ag.current_state,
                    actions.get(vid),
                    other_state,
                    role_type,
                    e.tcpa,
                )
                r_colregs *= score
                if score < 0.5:
                    colregs_violations += 1

            # Goal progress reward
            target_wp = ag.current_route.current_waypoint()
            if target_wp:
                dist = target_wp.distance_to(ag.current_state)
                r_efficiency = -dist / 5000.0
                if dist < target_wp.radius:
                    ag.current_route.advance()
            else:
                r_efficiency = 1.0

            rewards[vid] = float(
                self.config.safety_reward_weight * r_safety
                + self.config.colregs_reward_weight * r_colregs
                + self.config.efficiency_reward_weight * r_efficiency
            )

        team_reward = float(np.mean(list(rewards.values())))
        done = self.time_step >= self.config.episode_length

        obs = self._build_observations()
        vessel_ids = list(new_states.keys())
        true_pairwise_distances = [
            float(np.linalg.norm(new_states[vessel_ids[i]].position() - new_states[vessel_ids[j]].position()))
            for i in range(len(vessel_ids))
            for j in range(i + 1, len(vessel_ids))
        ]
        info = {
            "encounters": len(encounters),
            # Projected CPA from EncounterManager.compute_cpa -- a per-step
            # LINEAR extrapolation of current velocity, used for reward
            # shaping above. It reads near-zero in the instant just before a
            # vessel's rudder command actually changes its heading, even if
            # the real (curving) trajectory never gets that close -- don't
            # use it as a safety/resilience metric across an episode; use
            # `true_min_pairwise_distance` below for that.
            "min_cpa": min([e.cpa_distance for e in encounters], default=5000.0),
            # Actual Euclidean separation between every vessel pair's real
            # (not projected, not estimated) position this step -- the
            # quantity a true episode-minimum safety metric should reduce
            # over, since it reflects what actually happened rather than a
            # myopic one-step-ahead extrapolation.
            "true_min_pairwise_distance": min(true_pairwise_distances, default=5000.0),
            "colregs_violations": colregs_violations,
        }

        return obs, rewards, team_reward, done, info

    def _build_observations(self) -> dict[int, VesselObservation]:
        obs = {}
        dt_estimates = (
            self.scene.digital_twin.vessel_estimates
            if self.scene and self.scene.digital_twin
            else {}
        )

        for vid, ag in self.scene.vessels.items():
            neighbors = {}
            for other_id, other in self.scene.vessels.items():
                if other_id == vid:
                    continue
                # Use Digital Twin estimated state if available, else true state
                if other_id in dt_estimates and dt_estimates[other_id].estimated_state:
                    neighbors[other_id] = dt_estimates[other_id].estimated_state
                else:
                    neighbors[other_id] = other.current_state

            intents = {
                other_id: other.current_route
                for other_id, other in self.scene.vessels.items()
                if other_id != vid
            }
            comm_quality = {other_id: self.comms_degradation_level for other_id in neighbors}

            obs[vid] = VesselObservation(
                vessel_id=vid,
                own_state=ag.current_state,
                own_route=ag.current_route,
                neighbor_states=neighbors,
                neighbor_intents=intents,
                environment=self.scene.environment_condition,
                visibility_range=self._visibility_range,
                wind_speed=5.0,
                wind_direction=0.0,
                current_speed=0.5,
                current_direction=0.0,
                comm_link_quality=comm_quality,
                last_message_timestamp={nid: float(self.time_step) for nid in neighbors},
                estimated_neighbor_states=neighbors,
                estimation_confidence={nid: 0.9 for nid in neighbors},
                active_encounters=[],
                colregs_compliance_score=1.0,
            )
        return obs

    def get_scene(self) -> MaritimeScene:
        return self.scene
