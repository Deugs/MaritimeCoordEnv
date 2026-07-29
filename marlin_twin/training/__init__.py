"""
MARLIN-Twin Training Infrastructure subpackage.
"""

from marlin_twin.training.mappo import MAPPOTrainer
from marlin_twin.training.curriculum import TwoStageCurriculumTrainer

__all__ = [
    "MAPPOTrainer",
    "TwoStageCurriculumTrainer",
]
