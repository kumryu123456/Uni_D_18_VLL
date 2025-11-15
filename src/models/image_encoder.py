"""
Image encoder for document visual features.
"""

import torch
import torch.nn as nn

# Check torchvision availability
_BACKBONE_OK = False
try:
    from torchvision.models import resnet18, ResNet18_Weights
    from torchvision import transforms as T

    _BACKBONE_OK = True
except ImportError:
    pass


class TinyCNN(nn.Module):
    """
    Lightweight CNN fallback if torchvision unavailable.
    """

    def __init__(self, out_dim: int = 256):
        """
        Initialize TinyCNN.

        Args:
            out_dim: Output channel dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) input images

        Returns:
            (B, D, H', W') feature maps
        """
        return self.net(x)


class ImageEncoder(nn.Module):
    """
    Image encoder using ResNet18 or TinyCNN fallback.

    Extracts visual features from document images.
    """

    def __init__(
        self, out_dim: int = 256, pretrained: bool = True, img_size: int = 512
    ):
        """
        Initialize image encoder.

        Args:
            out_dim: Output feature dimension
            pretrained: Use ImageNet pretrained weights (if available)
            img_size: Expected input image size
        """
        super().__init__()
        self.resize = None

        if _BACKBONE_OK:
            try:
                # Use ResNet18 backbone
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                m = resnet18(weights=weights)

                # Remove classification head (avgpool + fc)
                layers = list(m.children())[:-2]  # Keep up to conv layers
                self.backbone = nn.Sequential(*layers)

                # Project from 512 (ResNet18 output) to out_dim
                self.proj = nn.Conv2d(512, out_dim, 1)

                # Optional: resize transform
                self.resize = T.Compose(
                    [T.Resize((img_size, img_size)), T.ToTensor()]
                )

            except Exception as e:
                print(f"Warning: Could not load ResNet18, using TinyCNN: {e}")
                self.backbone = TinyCNN(out_dim)
                self.proj = nn.Identity()
        else:
            # Fallback to TinyCNN
            self.backbone = TinyCNN(out_dim)
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode document images.

        Args:
            x: (B, 3, H, W) float tensor, RGB images normalized to [0,1]

        Returns:
            (B, D, H', W') float tensor of spatial feature maps
        """
        f = self.backbone(x)  # (B, 512, H/32, W/32) for ResNet18
        f = self.proj(f)  # (B, D, H', W')
        return f
