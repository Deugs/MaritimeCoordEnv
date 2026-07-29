"""
Baseline Algorithms subpackage for MARLIN-Twin benchmark comparison.
"""

from marlin_twin.baselines.rule_based import RuleBasedCOLREGsController
from marlin_twin.baselines.factory import BaselineFactory

__all__ = [
    "RuleBasedCOLREGsController",
    "BaselineFactory",
]
