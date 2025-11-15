"""Unit tests for loss functions."""
import pytest
import torch

try:
    from src.training.loss import smooth_l1_loss
except ImportError:
    pytest.skip("Loss module not yet implemented", allow_module_level=True)


class TestSmoothL1Loss:
    def test_smooth_l1_basic(self):
        """Test smooth L1 loss computation."""
        pred = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.4, 0.1, 0.1]])
        target = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.5, 0.15, 0.15]])

        loss = smooth_l1_loss(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_perfect_prediction(self):
        """Test loss is zero for perfect predictions."""
        pred = target = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        loss = smooth_l1_loss(pred, target)
        assert loss.item() < 1e-6
