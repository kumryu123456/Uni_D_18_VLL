"""
Unit tests for model components.
"""

import pytest
import torch

try:
    from src.models.text_encoder import TextEncoder
    from src.models.image_encoder import ImageEncoder, TinyCNN
    from src.models.fusion import CrossAttentionBBox
    from src.models.vlm import CrossAttnVLM
except ImportError:
    pytest.skip("Model modules not yet implemented", allow_module_level=True)


class TestTextEncoder:
    """Tests for TextEncoder."""

    def test_forward_pass(self):
        """Test TextEncoder forward pass."""
        vocab_size = 1000
        dim = 256
        encoder = TextEncoder(vocab_size=vocab_size, emb_dim=dim, hidden=dim)

        batch_size = 4
        max_len = 10
        tokens = torch.randint(0, vocab_size, (batch_size, max_len))
        lengths = torch.tensor([10, 8, 6, 10])

        output = encoder(tokens, lengths)

        # Should output (B, D)
        assert output.shape == (batch_size, dim)
        assert output.dtype == torch.float32

    def test_variable_length_handling(self):
        """Test handling of variable-length sequences."""
        encoder = TextEncoder(vocab_size=100, emb_dim=64, hidden=64)

        tokens = torch.randint(0, 100, (2, 15))
        lengths = torch.tensor([5, 15])  # Different lengths

        output = encoder(tokens, lengths)

        # Should handle both sequences
        assert output.shape == (2, 64)
        assert not torch.isnan(output).any()


class TestImageEncoder:
    """Tests for ImageEncoder."""

    def test_tinycnn_forward(self):
        """Test TinyCNN fallback forward pass."""
        encoder = TinyCNN(out_dim=256)

        batch_size = 2
        images = torch.randn(batch_size, 3, 512, 512)

        output = encoder(images)

        # Should output (B, D, H', W')
        assert output.dim() == 4
        assert output.size(0) == batch_size
        assert output.size(1) == 256  # out_dim

    def test_image_encoder_forward(self):
        """Test ImageEncoder forward pass."""
        encoder = ImageEncoder(out_dim=256, pretrained=False, img_size=512)

        batch_size = 2
        images = torch.randn(batch_size, 3, 512, 512)

        output = encoder(images)

        # Should output (B, D, H', W')
        assert output.dim() == 4
        assert output.size(0) == batch_size
        assert output.size(1) == 256

    def test_different_image_sizes(self):
        """Test encoder with different image sizes."""
        encoder = ImageEncoder(out_dim=128, pretrained=False, img_size=384)

        images = torch.randn(1, 3, 384, 384)
        output = encoder(images)

        assert output.size(1) == 128  # out_dim


class TestCrossAttentionBBox:
    """Tests for CrossAttentionBBox fusion module."""

    def test_forward_pass(self):
        """Test cross-attention forward pass."""
        dim = 256
        fusion = CrossAttentionBBox(dim=dim)

        batch_size = 4
        q_vec = torch.randn(batch_size, dim)
        fmap = torch.randn(batch_size, dim, 16, 16)

        pred = fusion(q_vec, fmap)

        # Should output (B, 4) normalized bbox
        assert pred.shape == (batch_size, 4)
        assert pred.dtype == torch.float32

        # Should be in [0, 1] range (sigmoid output)
        assert torch.all(pred >= 0)
        assert torch.all(pred <= 1)

    def test_attention_mechanism(self):
        """Test attention mechanism works."""
        fusion = CrossAttentionBBox(dim=128)

        q_vec = torch.randn(2, 128)
        fmap = torch.randn(2, 128, 8, 8)

        pred = fusion(q_vec, fmap)

        # Should produce valid predictions
        assert not torch.isnan(pred).any()
        assert not torch.isinf(pred).any()


class TestCrossAttnVLM:
    """Tests for complete CrossAttnVLM model."""

    def test_end_to_end_forward(self):
        """Test end-to-end model forward pass."""
        vocab_size = 1000
        dim = 256

        model = CrossAttnVLM(
            vocab_size=vocab_size,
            dim=dim,
            pretrained_backbone=False,
            img_size=512,
        )

        batch_size = 4
        images = torch.randn(batch_size, 3, 512, 512)
        tokens = torch.randint(0, vocab_size, (batch_size, 10))
        lengths = torch.tensor([10, 8, 6, 10])

        pred = model(images, tokens, lengths)

        # Should output (B, 4) normalized bbox
        assert pred.shape == (batch_size, 4)
        assert torch.all(pred >= 0)
        assert torch.all(pred <= 1)

    def test_model_trainable(self):
        """Test model parameters are trainable."""
        model = CrossAttnVLM(vocab_size=100, dim=64, pretrained_backbone=False)

        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params > 0

    def test_gradient_flow(self):
        """Test gradients flow through model."""
        model = CrossAttnVLM(vocab_size=100, dim=64, pretrained_backbone=False)

        images = torch.randn(2, 3, 512, 512, requires_grad=True)
        tokens = torch.randint(0, 100, (2, 5))
        lengths = torch.tensor([5, 5])

        pred = model(images, tokens, lengths)
        loss = pred.sum()  # Dummy loss
        loss.backward()

        # Check gradients exist
        assert images.grad is not None
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_batch_size_one(self):
        """Test model works with batch size 1."""
        model = CrossAttnVLM(vocab_size=100, dim=64, pretrained_backbone=False)

        images = torch.randn(1, 3, 512, 512)
        tokens = torch.randint(0, 100, (1, 5))
        lengths = torch.tensor([5])

        pred = model(images, tokens, lengths)

        assert pred.shape == (1, 4)
