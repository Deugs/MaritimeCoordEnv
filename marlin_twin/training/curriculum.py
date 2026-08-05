"""2-stage curriculum trainer: full-comms pre-training then bandwidth-degradation fine-tuning."""

from loguru import logger

from marlin_twin.api import BaseMaritimeEnvironment, Policy
from marlin_twin.training.mappo import MAPPOTrainer


class TwoStageCurriculumTrainer(MAPPOTrainer):
    """
    2-Stage Curriculum Learning Trainer for MARLIN-Twin.
    - Stage 1: Spatial navigation & COLREGs training under 100% full communication.
    - Stage 2: Learned bandwidth allocation & policy fine-tuning under comms loss.
    """

    def train_curriculum(
        self, env: BaseMaritimeEnvironment, total_episodes: int = 1000
    ) -> dict[int, Policy]:
        stage1_episodes = int(total_episodes * 0.6)
        stage2_episodes = total_episodes - stage1_episodes

        logger.info(f"--- STAGE 1: Spatial & COLREGs Pre-training ({stage1_episodes} eps) ---")
        env.set_communication_degradation(1.0)
        self.train(env, stage1_episodes)

        logger.info(f"--- STAGE 2: Bandwidth & Resilience Fine-Tuning ({stage2_episodes} eps) ---")
        for ep in range(stage2_episodes):
            deg_level = max(0.1, 1.0 - 0.9 * (ep / max(1, stage2_episodes)))
            env.set_communication_degradation(deg_level)
            self.train(env, 1)

        logger.info("Curriculum Training Complete!")
        return self.policies
