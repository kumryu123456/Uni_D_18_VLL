"""
Model Interface Contract

Defines the interface for the vision-language model components.
All model implementations must conform to these signatures.
"""

from typing import Protocol, Tuple
import torch
import torch.nn as nn


class TextEncoderProtocol(Protocol):
    """
    Text encoder interface.

    Encodes variable-length token sequences into fixed-dimension embeddings.
    """

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode text queries.

        Args:
            tokens: (B, L) long tensor of token IDs, padded with 0
            lengths: (B,) long tensor of valid lengths

        Returns:
            (B, D) float tensor of query embeddings
        """
        ...


class ImageEncoderProtocol(Protocol):
    """
    Image encoder interface.

    Extracts visual features from document images.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode document images.

        Args:
            images: (B, 3, H, W) float tensor, RGB images normalized to [0,1]

        Returns:
            (B, D, H', W') float tensor of spatial feature maps
            where H', W' depend on backbone downsampling factor
        """
        ...


class FusionModuleProtocol(Protocol):
    """
    Fusion module interface.

    Combines text and visual features to predict bounding boxes.
    """

    def forward(self, query_vec: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Fuse text and visual features, predict bounding box.

        Args:
            query_vec: (B, D) float tensor of text embeddings
            feature_map: (B, D, H', W') float tensor of visual features

        Returns:
            (B, 4) float tensor of predicted bboxes in normalized coords [0,1]
            Format: [cx, cy, w, h] (center x, center y, width, height)
        """
        ...


class VisionLanguageModelProtocol(Protocol):
    """
    Complete vision-language model interface.

    End-to-end model from images + queries to bounding boxes.
    """

    def forward(
        self, images: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict bounding boxes for queries on images.

        Args:
            images: (B, 3, H, W) float tensor, RGB images
            tokens: (B, L) long tensor, query token IDs
            lengths: (B,) long tensor, valid query lengths

        Returns:
            (B, 4) float tensor, predicted bboxes [cx, cy, w, h] in [0,1]
        """
        ...


# Type aliases for clarity
TokenTensor = torch.Tensor  # (B, L) long
LengthTensor = torch.Tensor  # (B,) long
ImageTensor = torch.Tensor  # (B, 3, H, W) float
QueryEmbedding = torch.Tensor  # (B, D) float
FeatureMap = torch.Tensor  # (B, D, H', W') float
BBoxPrediction = torch.Tensor  # (B, 4) float [cx, cy, w, h] normalized


# Example implementation signatures (not enforced, for documentation)

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
        backbone: Image encoder backbone name
        pretrained: Whether to use pretrained weights
        img_size: Input image size (square)

    Returns:
        VisionLanguageModel instance
    """
    ...


def load_model(checkpoint_path: str, device: str = "cpu") -> Tuple[nn.Module, dict]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Target device ("cpu", "cuda", "cuda:0", etc.)

    Returns:
        model: Loaded VisionLanguageModel
        vocab: Dictionary with 'itos' and 'stoi' keys
    """
    ...


def save_model(
    model: nn.Module,
    vocab: dict,
    save_path: str,
    config: dict,
    metadata: dict = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: VisionLanguageModel instance
        vocab: Vocabulary dict with 'itos' and 'stoi'
        save_path: Output .pth file path
        config: Hyperparameters dict
        metadata: Optional training metadata (epoch, metrics, etc.)
    """
    ...
