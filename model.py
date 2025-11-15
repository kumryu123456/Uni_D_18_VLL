"""
Model definition for competition submission.

This file is a required deliverable for the Dacon competition.
It provides the main model interface and factory functions.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

from src.models.vlm import CrossAttnVLM


def create_model(
    vocab_size: int,
    dim: int = 256,
    backbone: str = "resnet18",
    pretrained: bool = True,
    img_size: int = 512,
) -> nn.Module:
    """
    Factory function to create vision-language model.

    Args:
        vocab_size: Size of text vocabulary
        dim: Embedding dimension
        backbone: Image encoder backbone name (currently supports: resnet18)
        pretrained: Whether to use pretrained weights
        img_size: Input image size (square)

    Returns:
        CrossAttnVLM instance
    """
    model = CrossAttnVLM(
        vocab_size=vocab_size,
        dim=dim,
        pretrained_backbone=pretrained,
        img_size=img_size,
    )

    return model


def load_model(checkpoint_path: str, device: str = "cpu") -> Tuple[nn.Module, Dict]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Target device ("cpu", "cuda", "cuda:0", etc.)

    Returns:
        Tuple of (model, vocab_dict)
            model: Loaded CrossAttnVLM
            vocab_dict: Dictionary with 'itos' and 'stoi' keys
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract vocabulary
    from src.data.vocab import Vocab

    vocab = Vocab()
    vocab.itos = checkpoint["vocab_itos"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}

    # Create model
    model = CrossAttnVLM(
        vocab_size=len(vocab.itos),
        dim=checkpoint.get("dim", 256),
        pretrained_backbone=not checkpoint.get("no_pretrain", False),
        img_size=checkpoint.get("img_size", 512),
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    vocab_dict = {"itos": vocab.itos, "stoi": vocab.stoi}

    return model, vocab_dict


def save_model(
    model: nn.Module,
    vocab: Dict,
    save_path: str,
    config: Dict,
    metadata: Dict = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: CrossAttnVLM instance
        vocab: Vocabulary dict with 'itos' and 'stoi', or Vocab object
        save_path: Output .pth file path
        config: Hyperparameters dict
        metadata: Optional training metadata (epoch, metrics, etc.)
    """
    # Handle both dict and Vocab object
    if hasattr(vocab, "itos"):
        vocab_itos = vocab.itos
    elif isinstance(vocab, dict):
        vocab_itos = vocab.get("itos", [])
    else:
        vocab_itos = []

    checkpoint = {
        "model_state": model.state_dict(),
        "vocab_itos": vocab_itos,
        "dim": config.get("dim", 256),
        "no_pretrain": config.get("no_pretrain", False),
        "img_size": config.get("img_size", 512),
    }

    # Add metadata if provided
    if metadata:
        checkpoint.update(metadata)

    torch.save(checkpoint, save_path)


# Export main model class
__all__ = ["CrossAttnVLM", "create_model", "load_model", "save_model"]
