"""
Evaluation Interface Contract

Defines the interface for metrics computation and model evaluation.
"""

from typing import List, Tuple, Dict, Optional
import torch
import pandas as pd


# IoU computation

def compute_iou(
    pred_bbox: Tuple[float, float, float, float],
    gt_bbox: Tuple[float, float, float, float],
) -> float:
    """
    Compute Intersection over Union between two bounding boxes.

    Args:
        pred_bbox: (x, y, w, h) predicted bbox in pixels
        gt_bbox: (x, y, w, h) ground truth bbox in pixels

    Returns:
        IoU score in [0, 1]
    """
    ...


def compute_iou_batch(
    pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor
) -> torch.Tensor:
    """
    Compute IoU for batch of bounding boxes.

    Args:
        pred_bboxes: (B, 4) tensor [x, y, w, h]
        gt_bboxes: (B, 4) tensor [x, y, w, h]

    Returns:
        (B,) tensor of IoU scores
    """
    ...


def compute_miou(ious: List[float]) -> float:
    """
    Compute mean IoU.

    Args:
        ious: List of IoU scores

    Returns:
        Mean IoU
    """
    ...


# Generalized IoU

def compute_giou(
    pred_bbox: Tuple[float, float, float, float],
    gt_bbox: Tuple[float, float, float, float],
) -> float:
    """
    Compute Generalized IoU.

    Args:
        pred_bbox: (x, y, w, h) predicted bbox
        gt_bbox: (x, y, w, h) ground truth bbox

    Returns:
        GIoU score in [-1, 1]
    """
    ...


# Evaluation metrics

def compute_precision_at_threshold(
    ious: List[float], threshold: float = 0.5
) -> float:
    """
    Compute precision at IoU threshold.

    Args:
        ious: List of IoU scores
        threshold: IoU threshold

    Returns:
        Fraction of predictions with IoU >= threshold
    """
    ...


def compute_center_error(
    pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor
) -> float:
    """
    Compute mean L2 distance between bbox centers.

    Args:
        pred_bboxes: (B, 4) tensor [cx, cy, w, h] in pixels
        gt_bboxes: (B, 4) tensor [cx, cy, w, h] in pixels

    Returns:
        Mean L2 distance in pixels
    """
    ...


def compute_size_error(
    pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor
) -> float:
    """
    Compute mean relative size error.

    Args:
        pred_bboxes: (B, 4) tensor [cx, cy, w, h]
        gt_bboxes: (B, 4) tensor [cx, cy, w, h]

    Returns:
        Mean relative error in bbox area
    """
    ...


# Comprehensive evaluation

def evaluate_predictions(
    predictions: List[Dict],
    ground_truths: List[Dict],
    group_by: Optional[str] = None,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of predictions.

    Args:
        predictions: List of dicts with keys:
            - query_id: str
            - bbox: (x, y, w, h) tuple
            - class_name: str (optional)
        ground_truths: List of dicts with same structure
        group_by: Optional grouping key ("class_name", "bbox_size")

    Returns:
        Dictionary of metrics:
            - miou: Mean IoU
            - precision_at_50: Precision at IoU > 0.5
            - precision_at_75: Precision at IoU > 0.75
            - mean_center_error: Mean center distance
            - mean_size_error: Mean size error
            - per_class_iou: Dict (if group_by="class_name")
            - per_size_iou: Dict (if group_by="bbox_size")
    """
    ...


def categorize_bbox_size(bbox: Tuple[float, float, float, float]) -> str:
    """
    Categorize bounding box by size.

    Args:
        bbox: (x, y, w, h) in pixels

    Returns:
        "small" (area < 32^2), "medium" (< 96^2), or "large"
    """
    ...


# Inference

def predict_on_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> List[Dict]:
    """
    Run inference on dataset.

    Args:
        model: VisionLanguageModel instance
        dataloader: DataLoader for inference
        device: Device for computation

    Returns:
        List of prediction dicts:
            - query_id: str
            - query_text: str
            - bbox: (x, y, w, h) tuple in pixels
            - confidence: float (optional)
    """
    ...


def save_predictions_csv(
    predictions: List[Dict],
    output_path: str,
    columns: List[str] = None,
) -> None:
    """
    Save predictions to CSV file.

    Args:
        predictions: List of prediction dicts
        output_path: Output CSV file path
        columns: Column order (default: ["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
    """
    ...


def load_predictions_csv(csv_path: str) -> List[Dict]:
    """
    Load predictions from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of prediction dicts
    """
    ...


# Validation utilities

def validate_submission_format(csv_path: str, test_queries: List[str]) -> bool:
    """
    Validate submission CSV format.

    Args:
        csv_path: Path to submission CSV
        test_queries: List of expected query IDs

    Returns:
        True if valid, raises ValueError otherwise

    Checks:
        - Required columns present
        - All test queries included
        - No NaN values
        - Bounding boxes non-negative
        - Correct encoding (UTF-8 with BOM)
    """
    ...


def check_bbox_validity(
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> bool:
    """
    Check if bounding box is valid.

    Args:
        bbox: (x, y, w, h) in pixels
        img_width: Image width
        img_height: Image height

    Returns:
        True if bbox is within image bounds and non-negative
    """
    ...


# Analysis and visualization (optional)

def compute_error_statistics(
    predictions: List[Dict],
    ground_truths: List[Dict],
) -> Dict[str, Dict]:
    """
    Compute detailed error statistics.

    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts

    Returns:
        Dictionary with error analysis:
            - iou_distribution: Histogram of IoU scores
            - error_by_class: Per-class error stats
            - error_by_size: Per-size error stats
            - outliers: List of worst predictions
    """
    ...


def find_worst_predictions(
    predictions: List[Dict],
    ground_truths: List[Dict],
    n: int = 10,
) -> List[Dict]:
    """
    Find N worst predictions by IoU.

    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        n: Number of worst cases to return

    Returns:
        List of dicts with prediction, ground truth, and IoU
    """
    ...
