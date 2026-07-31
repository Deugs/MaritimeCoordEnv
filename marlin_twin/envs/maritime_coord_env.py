# ============================================================================
# FILE: marlin_twin/envs/maritime_coord_env.py
# ============================================================================

import numpy as np
from marlin_twin.data_classes import (
    MaritimeExperimentConfig, MaritimeScene, VesselObservation, VesselAction,
    EnvironmentCondition, MaritimeCommunicationChannel, MaritimeDigitalTwin,
    MessagePriority, MaritimeMessage, VesselState
)
from marlin_twin.envs.base_env import BaseMaritimeEnvironment
from marlin_twin.envs.scenarios import ScenarioGenerator
from marlin_twin.envs.vessel_dynamics import MMGDynamicsSolver
from marlin_twin.envs.colregs import COLREGsEngine
from marlin_twin.envs.encounters import EncounterManager
from marlin_twin.envs.digital_twin import DigitalTwinEstimator
from marlin_twin.envs.communication import CommunicationChannelManager
from marlin_twin.envs.sensors import SensorSimulator

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

    def reset(
        self,
        scenario_type: str | None = None,
        n_vessels: int | None = None,
        seed: int | None = None
    ) -> tuple[dict[int, VesselObservation], dict]:
        st = scenario_type or self.config.scenario_type
        nv = n_vessels or self.config.n_vessels
        
        self.time_step = 0
        agents = ScenarioGenerator.create_scenario(st, nv, seed)
        self.solvers = {vid: MMGDynamicsSolver(ag.dynamics) for vid, ag in agents.items()}

        states = {vid: ag.current_state for vid, ag in agents.items() if ag.current_state}
        encounters = EncounterManager.detect_encounters(states)

        dt_scene = self.dt_estimator.update("init_scene", 0.0, states, [], [])

        self.scene = MaritimeScene(
            scene_id="maritime_scene_0",
            timestamp=0.0,
            vessels=agents,
            communication_channel=self.comm_manager.channel,
            digital_twin=dt_scene,
            boundaries=self.config.boundaries,
            environment_condition=EnvironmentCondition.CLEAR
        )

        obs = self._build_observations()
        return obs, {"encounters": len(encounters)}

    def step(
        self,
        actions: dict[int, VesselAction]
    ) -> tuple[dict[int, VesselObservation], dict[int, float], float, bool, dict]:
        self.time_step += 1
        dt = 1.0

        # Step 1: Update Dynamics
        new_states = {}
        for vid, ag in self.scene.vessels.items():
            action = actions.get(vid, VesselAction(vessel_id=vid, propeller_rpm=0.5, rudder_angle=0.0, message_targets=[]))
            new_state = self.solvers[vid].step(ag.current_state, action, dt)
            ag.current_state = new_state
            ag.last_action = action
            new_states[vid] = new_state

        # Step 2: Sensors & Digital Twin Update
        drop_prob = float(np.clip(1.0 - self.comms_degradation_level, 0.0, 0.98))
        ais_readings = [SensorSimulator.generate_ais(s, float(self.time_step), drop_prob=drop_prob) for s in new_states.values()]
        ais_readings = [r for r in ais_readings if r is not None]
        radar_tracks = [SensorSimulator.generate_radar(s, float(self.time_step), idx) for idx, s in enumerate(new_states.values())]
        
        dt_scene = self.dt_estimator.update(f"scene_{self.time_step}", float(self.time_step), new_states, ais_readings, radar_tracks)
        self.scene.digital_twin = dt_scene

        # Step 3: Encounters & COLREGs Compliance
        encounters = EncounterManager.detect_encounters(new_states)

        # Step 4: Process Communication Messages
        outgoing = []
        for vid, act in actions.items():
            for target in act.message_targets:
                outgoing.append(MaritimeMessage(
                    sender_id=vid, receiver_id=target,
                    content=np.array([new_states[vid].x, new_states[vid].y, new_states[vid].heading, new_states[vid].speed]),
                    priority=act.message_priority, timestamp=float(self.time_step), size_bits=256
                ))

        weather = 1.0 - self.comms_degradation_level
        self.comm_manager.process_step(outgoing, weather_degradation=weather)

        # Step 5: Compute Rewards
        rewards = {}
        for vid, ag in self.scene.vessels.items():
            # Safety reward (CPA penalty)
            min_cpa = min([e.cpa_distance for e in encounters if e.vessel_i == vid or e.vessel_j == vid], default=5000.0)
            r_safety = -np.exp(-min_cpa / 200.0)

            # COLREGs compliance reward
            r_colregs = 1.0
            for e in encounters:
                if e.vessel_i == vid:
                    score = COLREGsEngine.evaluate_compliance(ag.current_state, actions.get(vid), new_states[e.vessel_j], e.encounter_type, e.tcpa)
                    r_colregs *= score

            # Goal progress reward
            target_wp = ag.current_route.current_waypoint()
            if target_wp:
                dist = target_wp.distance_to(ag.current_state)
                r_efficiency = -dist / 5000.0
                if dist < target_wp.radius:
                    ag.current_route.advance()
            else:
                r_efficiency = 1.0

            rewards[vid] = float(self.config.safety_reward_weight * r_safety + self.config.colregs_reward_weight * r_colregs + self.config.efficiency_reward_weight * r_efficiency)

        team_reward = float(np.mean(list(rewards.values())))
        done = self.time_step >= self.config.episode_length

        obs = self._build_observations()
        info = {"encounters": len(encounters), "min_cpa": min([e.cpa_distance for e in encounters], default=5000.0)}

        return obs, rewards, team_reward, done, info

    def _build_observations(self) -> dict[int, VesselObservation]:
        obs = {}
        dt_estimates = self.scene.digital_twin.vessel_estimates if self.scene and self.scene.digital_twin else {}

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

            intents = {other_id: other.current_route for other_id, other in self.scene.vessels.items() if other_id != vid}
            comm_quality = {other_id: self.comms_degradation_level for other_id in neighbors}

            obs[vid] = VesselObservation(
                vessel_id=vid,
                own_state=ag.current_state,
                own_route=ag.current_route,
                neighbor_states=neighbors,
                neighbor_intents=intents,
                environment=EnvironmentCondition.CLEAR,
                visibility_range=5000.0,
                wind_speed=5.0, wind_direction=0.0,
                current_speed=0.5, current_direction=0.0,
                comm_link_quality=comm_quality,
                last_message_timestamp={nid: float(self.time_step) for nid in neighbors},
                estimated_neighbor_states=neighbors,
                estimation_confidence={nid: 0.9 for nid in neighbors},
                active_encounters=[],
                colregs_compliance_score=1.0
            )
        return obs

    def get_scene(self) -> MaritimeScene:
        return self.scene
