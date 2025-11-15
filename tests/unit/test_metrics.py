"""Unit tests for metrics."""
import pytest
import torch

try:
    from src.training.metrics import iou_xywh_pixel, compute_miou
except ImportError:
    pytest.skip("Metrics module not yet implemented", allow_module_level=True)


class TestIoU:
    def test_iou_perfect_match(self):
        """Test IoU is 1.0 for perfect match."""
        bbox = (100, 200, 300, 150)
        iou = iou_xywh_pixel(bbox, bbox)
        assert abs(iou - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        """Test IoU is 0.0 for no overlap."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (200, 200, 100, 100)
        iou = iou_xywh_pixel(bbox1, bbox2)
        assert abs(iou) < 1e-6

    def test_iou_partial_overlap(self):
        """Test IoU for partial overlap."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 100, 100)
        iou = iou_xywh_pixel(bbox1, bbox2)
        assert 0 < iou < 1

    def test_compute_miou(self):
        """Test mean IoU computation."""
        ious = [0.8, 0.9, 0.7, 0.6]
        miou = compute_miou(ious)
        assert abs(miou - 0.75) < 1e-6
