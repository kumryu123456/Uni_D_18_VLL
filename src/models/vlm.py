"""
Vision-language model for query-based visual element localization.
"""

import logging
import torch
import torch.nn as nn

from src.models.text_encoder import TextEncoder
from src.models.image_encoder import ImageEncoder
from src.models.fusion import CrossAttentionBBox

logger = logging.getLogger(__name__)


class CrossAttnVLM(nn.Module):
    """
    Complete vision-language model.

    Combines text encoder, image encoder, and cross-attention fusion
    to predict bounding boxes from image + query pairs.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        pretrained_backbone: bool = True,
        img_size: int = 512,
    ):
        """
        Initialize vision-language model.

        Args:
            vocab_size: Size of text vocabulary
            dim: Embedding dimension
            pretrained_backbone: Use ImageNet pretrained weights
            img_size: Input image size (square)
        """
        super().__init__()

        self.txt = TextEncoder(vocab_size=vocab_size, emb_dim=dim, hidden=dim)
        self.img = ImageEncoder(out_dim=dim, pretrained=pretrained_backbone, img_size=img_size)
        self.head = CrossAttentionBBox(dim=dim)

        # Log model info
        self._log_model_info()

    def _log_model_info(self):
        """Log model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"CrossAttnVLM initialized:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

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
        # Encode text query
        q = self.txt(tokens, lengths)  # (B, D)

        # Encode image
        fmap = self.img(images)  # (B, D, H', W')

        # Fuse and predict bbox
        pred_norm = self.head(q, fmap)  # (B, 4)

        return pred_norm

    def count_parameters(self):
        """
        Count model parameters.

        Returns:
            Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Per-component counts
        txt_params = sum(p.numel() for p in self.txt.parameters())
        img_params = sum(p.numel() for p in self.img.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "text_encoder": txt_params,
            "image_encoder": img_params,
            "fusion_head": head_params,
        }
