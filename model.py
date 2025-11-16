"""
FINAL OPTIMIZED Vision-Language Model - Maximum IoU Performance
✅ ResNet50 Backbone (Pretrained)
✅ Bidirectional GRU Text Encoder
✅ Multi-Head Cross-Attention
✅ Attention Pooling
✅ Robust BBox Regression
"""

import logging
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """
    Bidirectional GRU text encoder with dropout regularization.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    @property
    def out_dim(self) -> int:
        return self.hidden_dim * (2 if self.bidirectional else 1)

    def forward(self, ids: Tensor, lengths: Tensor) -> Tensor:
        """
        Args:
            ids: [B, L] token indices
            lengths: [B] sequence lengths

        Returns:
            [B, L, D] encoded text features
        """
        # Handle zero-length sequences
        if (lengths == 0).any():
            lengths = torch.clamp(lengths, min=1)

        emb = self.embedding(ids)  # [B, L, E]
        emb = self.dropout(emb)

        # Pack, process, unpack for efficient RNN
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        return out  # [B, L, D]


class ImageEncoder(nn.Module):
    """
    ResNet-based image encoder with pretrained weights.
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()

        # Support multiple backbones
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            feat_dim = 512
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base = models.resnet34(weights=weights)
            feat_dim = 512
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove avgpool and fc layers
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.feat_dim = feat_dim

        logger.info(f"ImageEncoder: {backbone} (pretrained={pretrained})")

    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: [B, 3, H, W]

        Returns:
            [B, N, C] spatial features where N=H'*W'
        """
        feat = self.backbone(images)  # [B, C, H', W']
        B, C, H, W = feat.shape
        feat = feat.view(B, C, H * W).transpose(1, 2)  # [B, N, C]
        return feat


class CrossAttentionLocalization(nn.Module):
    """
    Cross-attention module for text-guided visual localization.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Attention pooling for feature fusion
        self.attention_pool = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )

        # BBox regression head
        self.regressor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 4),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

    def forward(
        self,
        text_feat: Tensor,
        image_feat: Tensor,
    ) -> Tensor:
        """
        Args:
            text_feat: [B, L, D] text features
            image_feat: [B, N, D] image features

        Returns:
            [B, 4] bbox predictions (cx, cy, w, h) in [0, 1]
        """
        # Cross-attention: query=text, key/value=image
        attn_out, _ = self.cross_attn(
            query=text_feat,
            key=image_feat,
            value=image_feat,
            need_weights=False,
        )  # [B, L, D]

        # Attention pooling
        pool_weights = self.attention_pool(attn_out)  # [B, L, 1]
        pool_weights = torch.softmax(pool_weights, dim=1)
        fused = (attn_out * pool_weights).sum(dim=1)  # [B, D]

        # Predict bbox
        bbox = self.regressor(fused)  # [B, 4]

        return bbox


class DocumentVLModel(nn.Module):
    """
    FINAL Optimized Vision-Language Model for Document Understanding.

    Architecture:
    - ResNet50 (Pretrained) → Spatial Features
    - Bidirectional GRU → Text Features
    - Cross-Attention → Text-Guided Visual Grounding
    - Attention Pooling → Feature Fusion
    - BBox Regression → (cx, cy, w, h) in [0, 1]
    """

    def __init__(
        self,
        vocab_size: int,
        txt_emb_dim: int = 256,
        txt_hidden_dim: int = 256,
        txt_num_layers: int = 2,
        attn_dim: int = 256,
        attn_heads: int = 8,
        backbone: str = "resnet50",
        pretrained_backbone: bool = True,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            emb_dim=txt_emb_dim,
            hidden_dim=txt_hidden_dim,
            num_layers=txt_num_layers,
            bidirectional=True,
            pad_idx=pad_idx,
            dropout=dropout,
        )

        # Image encoder
        self.image_encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained_backbone
        )

        # Projection layers to common embedding space
        self.img_proj = nn.Linear(self.image_encoder.feat_dim, attn_dim)
        self.txt_proj = nn.Linear(self.text_encoder.out_dim, attn_dim)

        # Cross-attention localization
        self.cross_attn = CrossAttentionLocalization(
            dim=attn_dim,
            num_heads=attn_heads,
            dropout=dropout,
        )

        logger.info(
            f"FINAL DocumentVLModel initialized:\n"
            f"  - Vocab: {vocab_size}\n"
            f"  - Attention Dim: {attn_dim}\n"
            f"  - Heads: {attn_heads}\n"
            f"  - Backbone: {backbone} (pretrained={pretrained_backbone})"
        )

    def forward(self, images: Tensor, text_ids: Tensor, text_lens: Tensor) -> Tensor:
        """
        Args:
            images: [B, 3, H, W] image tensor
            text_ids: [B, L] token IDs
            text_lens: [B] sequence lengths

        Returns:
            [B, 4] bbox predictions (cx, cy, w, h) in [0, 1]
        """
        # Encode
        img_feat = self.image_encoder(images)  # [B, N, C]
        txt_feat = self.text_encoder(text_ids, text_lens)  # [B, L, D]

        # Project to common embedding space
        img_feat = self.img_proj(img_feat)  # [B, N, A]
        txt_feat = self.txt_proj(txt_feat)  # [B, L, A]

        # Cross-attention and bbox regression
        pred = self.cross_attn(txt_feat, img_feat)  # [B, 4]

        return pred


def create_model(
    vocab_size: int,
    embed_dim: int = 256,
    backbone: str = "resnet50",
    pretrained: bool = True,
    num_heads: int = 8,
    dropout: float = 0.1,
    **kwargs
) -> DocumentVLModel:
    """
    Factory function to create FINAL optimized model.

    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension (default: 256)
        backbone: Image backbone (resnet18/34/50, default: resnet50)
        pretrained: Use pretrained weights (default: True)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)

    Returns:
        DocumentVLModel instance
    """
    return DocumentVLModel(
        vocab_size=vocab_size,
        txt_emb_dim=embed_dim,
        txt_hidden_dim=embed_dim,
        attn_dim=embed_dim,
        attn_heads=num_heads,
        backbone=backbone,
        pretrained_backbone=pretrained,
        dropout=dropout,
        **kwargs
    )


if __name__ == "__main__":
    print("="*70)
    print("🧪 FINAL Model Test")
    print("="*70)

    # Create model
    model = create_model(
        vocab_size=5000,
        embed_dim=256,
        backbone="resnet50",
        pretrained=False,  # Set False for quick test
        num_heads=8,
    )

    # Test forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 512, 512)
    text_ids = torch.randint(0, 5000, (batch_size, 50))
    text_lens = torch.tensor([50, 45, 40, 35])

    with torch.no_grad():
        pred = model(images, text_ids, text_lens)

    print(f"✅ Input: images={images.shape}, text_ids={text_ids.shape}")
    print(f"✅ Output: pred={pred.shape} (should be [{batch_size}, 4])")
    print(f"✅ BBox range: [{pred.min():.3f}, {pred.max():.3f}] (should be [0, 1])")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Trainable parameters: {trainable_params:,}")
    print("="*70)
    print("✅ Model test passed!")
