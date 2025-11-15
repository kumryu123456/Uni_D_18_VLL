"""
Random seed utilities for reproducibility.
"""

import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Random seed value (default: 42)

    Note:
        Setting deterministic=True may slow down training by ~10%
        but ensures exact reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
