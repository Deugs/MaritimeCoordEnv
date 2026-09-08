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
        self, env: BaseMaritimeEnvironment, total_episodes: int = 1000, on_episode_end=None
    ) -> dict[int, Policy]:
        stage1_episodes = int(total_episodes * 0.6)
        stage2_episodes = total_episodes - stage1_episodes

        logger.info(f"--- STAGE 1: Spatial & COLREGs Pre-training ({stage1_episodes} eps) ---")
        env.set_communication_degradation(1.0)
        self.train(env, stage1_episodes, on_episode_end=on_episode_end)

        logger.info(f"--- STAGE 2: Bandwidth & Resilience Fine-Tuning ({stage2_episodes} eps) ---")
        for ep in range(stage2_episodes):
            deg_level = max(0.1, 1.0 - 0.9 * (ep / max(1, stage2_episodes)))
            env.set_communication_degradation(deg_level)
            # seed_offset=stage1_episodes+ep, NOT bare `ep` -- each of these
            # calls passes n_episodes=1, so train()'s internal `ep` is
            # always 0; without an offset every Stage 2 episode (40% of all
            # curriculum training) reset with the identical seed regardless
            # of how many total episodes were requested.
            self.train(
                env,
                1,
                seed_offset=stage1_episodes + ep,
                on_episode_end=(
                    (
                        lambda _ep, trainer, real_ep=ep: on_episode_end(
                            stage1_episodes + real_ep, trainer
                        )
                    )
                    if on_episode_end is not None
                    else None
                ),
            )

        logger.info("Curriculum Training Complete!")
        return self.policies
