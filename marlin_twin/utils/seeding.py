# ============================================================================
# FILE: marlin_twin/utils/seeding.py
# ============================================================================

import random
import numpy as np

def seed_everything(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
