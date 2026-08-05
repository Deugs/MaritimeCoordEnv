# ============================================================================
# FILE: tests/test_extended_training.py
# ============================================================================

import os
from marlin_twin.data_classes import MaritimeExperimentConfig
from marlin_twin.training.mappo import MAPPOTrainer


def test_mappo_save_load_checkpoint(tmp_path):
    config = MaritimeExperimentConfig(n_vessels=2)
    trainer = MAPPOTrainer(config)
    trainer.policies = {0: trainer.config.scenario_type, 1: trainer.config.scenario_type}

    from marlin_twin.agents.policies import GATPolicy

    trainer.policies = {0: GATPolicy(), 1: GATPolicy()}

    ckpt_file = os.path.join(tmp_path, "test_ckpt.pt")
    trainer.save_checkpoint(ckpt_file)
    assert os.path.exists(ckpt_file)

    trainer.load_checkpoint(ckpt_file)
    assert len(trainer.policies) == 2
