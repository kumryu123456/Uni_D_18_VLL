"""Evaluation metrics for model performance."""
from typing import List, Tuple
import numpy as np


def iou_xywh_pixel(
    pred_xywh: Tuple[float, float, float, float],
    gt_xywh: Tuple[float, float, float, float],
) -> float:
    """
    Compute Intersection over Union for bounding boxes in pixel coordinates.

    Args:
        pred_xywh: (x, y, w, h) predicted bbox in pixels
        gt_xywh: (x, y, w, h) ground truth bbox in pixels

    Returns:
        IoU score in [0, 1]
    """
    px, py, pw, ph = pred_xywh
    gx, gy, gw, gh = gt_xywh

    # Convert to (x1, y1, x2, y2)
    px2, py2 = px + pw, py + ph
    gx2, gy2 = gx + gw, gy + gh

    # Compute intersection
    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)

    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    # Compute union
    pred_area = pw * ph
    gt_area = gw * gh
    union_area = pred_area + gt_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_miou(ious: List[float]) -> float:
    """
    Compute mean IoU.

    Args:
        ious: List of IoU scores

    Returns:
        Mean IoU
    """
    if len(ious) == 0:
        return 0.0
    return float(np.mean(ious))
