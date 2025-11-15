"""
Training Interface Contract

Defines the interface for training, evaluation, and metrics.
"""

from typing import Protocol, Dict, List, Tuple, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LossFunctionProtocol(Protocol):
    """Loss function interface."""

    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.

        Args:
            predictions: (B, 4) float tensor, predicted bboxes [cx, cy, w, h]
            targets: (B, 4) float tensor, ground truth bboxes [cx, cy, w, h]

        Returns:
            Scalar loss tensor
        """
        ...


# Training function signatures

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: str = "cuda",
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    """
    Train model for one epoch.

    Args:
        model: VisionLanguageModel instance
        dataloader: Training data loader
        optimizer: Optimizer instance
        loss_fn: Loss function
        device: Training device
        scaler: Optional AMP gradient scaler

    Returns:
        Dictionary with metrics:
            - loss: Average training loss
            - lr: Current learning rate
    """
    ...


def validate_epoch(
    model: nn.Module, dataloader: DataLoader, loss_fn: Callable, device: str = "cuda"
) -> Dict[str, float]:
    """
    Validate model on validation set.

    Args:
        model: VisionLanguageModel instance
        dataloader: Validation data loader
        loss_fn: Loss function
        device: Device for validation

    Returns:
        Dictionary with metrics:
            - loss: Average validation loss
            - miou: Mean IoU
            - precision_at_50: % predictions with IoU > 0.5
            - precision_at_75: % predictions with IoU > 0.75
    """
    ...


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn: Callable,
    epochs: int,
    device: str = "cuda",
    checkpoint_dir: str = "./outputs/ckpt",
    early_stopping_patience: int = 5,
) -> Dict[str, List[float]]:
    """
    Complete training loop with validation and checkpointing.

    Args:
        model: VisionLanguageModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        scheduler: Optional learning rate scheduler
        loss_fn: Loss function
        epochs: Number of training epochs
        device: Training device
        checkpoint_dir: Directory to save checkpoints
        early_stopping_patience: Epochs without improvement before stopping

    Returns:
        History dictionary with lists:
            - train_loss: Training losses per epoch
            - val_loss: Validation losses per epoch
            - val_miou: Validation mIoU per epoch
            - learning_rate: Learning rates per epoch
    """
    ...


# Loss functions

def smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Smooth L1 (Huber) loss for bounding boxes.

    Args:
        predictions: (B, 4) predicted [cx, cy, w, h]
        targets: (B, 4) ground truth [cx, cy, w, h]

    Returns:
        Scalar loss
    """
    ...


def giou_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU loss.

    Args:
        predictions: (B, 4) predicted [cx, cy, w, h]
        targets: (B, 4) ground truth [cx, cy, w, h]

    Returns:
        Scalar loss (1 - GIoU)
    """
    ...


def combined_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> torch.Tensor:
    """
    Weighted combination of Smooth L1 and GIoU loss.

    Args:
        predictions: (B, 4) predicted bboxes
        targets: (B, 4) ground truth bboxes
        alpha: Weight for Smooth L1
        beta: Weight for GIoU

    Returns:
        Scalar loss
    """
    ...


# Optimizer and scheduler factories

def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    optimizer_type: str = "adamw",
) -> torch.optim.Optimizer:
    """
    Create optimizer instance.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        optimizer_type: "adamw", "adam", "sgd"

    Returns:
        Optimizer instance
    """
    ...


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 10,
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_type: "cosine", "step", "plateau"
        epochs: Total number of epochs
        **kwargs: Additional scheduler-specific arguments

    Returns:
        Scheduler instance
    """
    ...


# Checkpointing

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict,
    vocab: Dict,
    save_path: str,
) -> None:
    """
    Save training checkpoint.

    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch number
        metrics: Current metrics (val_miou, etc.)
        config: Hyperparameters dict
        vocab: Vocabulary dict
        save_path: Output checkpoint path
    """
    ...


def load_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> Tuple[nn.Module, torch.optim.Optimizer, int, Dict]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device

    Returns:
        model: Loaded model
        optimizer: Loaded optimizer (optional, may be None)
        epoch: Epoch number
        config: Hyperparameters dict
    """
    ...


# Early stopping

class EarlyStopping:
    """Early stopping utility."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Initialize early stopping.

        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum improvement to reset patience
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, val_metric: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_metric: Current validation metric (higher is better)

        Returns:
            True if training should stop
        """
        ...

    def reset(self) -> None:
        """Reset early stopping state."""
        ...
