# ============================================================================
# FILE: marlin_twin/baselines/independent_ppo.py
# ============================================================================

from marlin_twin.agents.policies import GATPolicy

class IndependentPPOPolicy(GATPolicy):
    """Independent PPO Baseline Policy (No inter-agent communication)."""
    pass
