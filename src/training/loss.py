"""Loss functions for training."""
import torch
import torch.nn.functional as F


def smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Smooth L1 (Huber) loss for bounding boxes.

    Args:
        predictions: (B, 4) predicted [cx, cy, w, h]
        targets: (B, 4) ground truth [cx, cy, w, h]

    Returns:
        Scalar loss
    """
    return F.smooth_l1_loss(predictions, targets, reduction="mean")
