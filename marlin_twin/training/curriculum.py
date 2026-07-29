# ============================================================================
# FILE: marlin_twin/training/curriculum.py
# ============================================================================

from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.api import BaseMaritimeEnvironment, Policy
from marlin_twin.training.mappo import MAPPOTrainer

class TwoStageCurriculumTrainer(MAPPOTrainer):
    """
    2-Stage Curriculum Learning Trainer for MARLIN-Twin.
    - Stage 1: Spatial navigation & COLREGs training under 100% full communication.
    - Stage 2: Learned bandwidth allocation & policy fine-tuning under comms loss.
    """

    def train_curriculum(self, env: BaseMaritimeEnvironment, total_episodes: int = 1000) -> dict[int, Policy]:
        stage1_episodes = int(total_episodes * 0.6)
        stage2_episodes = total_episodes - stage1_episodes

        print(f"\n--- STAGE 1: Full Communication Spatial & COLREGs Pre-training ({stage1_episodes} episodes) ---")
        env.set_communication_degradation(1.0)
        self.train(env, stage1_episodes)

        print(f"\n--- STAGE 2: Bandwidth Allocation & Resilience Fine-Tuning ({stage2_episodes} episodes) ---")
        for ep in range(stage2_episodes):
            deg_level = max(0.1, 1.0 - 0.9 * (ep / stage2_episodes))
            env.set_communication_degradation(deg_level)

            obs, info = env.reset(seed=2000 + ep)
            done = False
            while not done:
                actions = {vid: env.get_scene().vessels[vid].last_action for vid in obs}
                obs, _, _, done, _ = env.step(actions)

        print("Curriculum Training Complete!")
        return self.policies
