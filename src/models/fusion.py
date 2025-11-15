"""
Fusion module for combining text and visual features.
"""

import math
import torch
import torch.nn as nn


class CrossAttentionBBox(nn.Module):
    """
    Cross-attention fusion for bounding box prediction.

    Uses query vector (text) to attend over image feature map,
    then predicts normalized bounding box coordinates.
    """

    def __init__(self, dim: int = 256):
        """
        Initialize cross-attention module.

        Args:
            dim: Feature dimension
        """
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)

        self.bbox_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 4),  # (cx, cy, w, h)
        )

    def forward(self, q_vec: torch.Tensor, fmap: torch.Tensor) -> torch.Tensor:
        """
        Fuse text and visual features, predict bounding box.

        Args:
            q_vec: (B, D) float tensor of text embeddings
            fmap: (B, D, H', W') float tensor of visual features

        Returns:
            (B, 4) float tensor of predicted bboxes in normalized coords [0,1]
            Format: [cx, cy, w, h] (center x, center y, width, height)
        """
        B, D, H, W = fmap.shape

        # Project query, keys, values
        q = self.q_proj(q_vec)  # (B, D)
        K = self.k_proj(fmap)  # (B, D, H, W)
        V = self.v_proj(fmap)  # (B, D, H, W)

        # Flatten spatial dimensions for attention
        Kf = K.flatten(2).transpose(1, 2)  # (B, HW, D)
        Vf = V.flatten(2).transpose(1, 2)  # (B, HW, D)
        q = q.unsqueeze(1)  # (B, 1, D)

        # Compute attention scores
        attn = torch.matmul(q, Kf.transpose(1, 2)) / math.sqrt(D)  # (B, 1, HW)
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values
        ctx = torch.matmul(attn, Vf).squeeze(1)  # (B, D)

        # Predict bounding box
        pred = self.bbox_head(ctx)  # (B, 4)

        # Normalize to [0, 1] with sigmoid
        pred = torch.sigmoid(pred)

        return pred
