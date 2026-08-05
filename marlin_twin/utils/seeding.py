"""Global random seed management for reproducibility."""

import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
