"""
Unit tests for Dataset implementation.
"""

import os
import json
import tempfile
import pytest
import torch
from PIL import Image
import numpy as np

# Mock imports for testing without dependencies
try:
    from src.data.dataset import UniDSet, collate_fn
    from src.data.vocab import Vocab
except ImportError:
    pytest.skip("Dataset module not yet implemented", allow_module_level=True)


class TestUniDSet:
    """Tests for UniDSet dataset class."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample JSON and image data for testing."""
        # Create directories
        json_dir = tmp_path / "json"
        jpg_dir = tmp_path / "jpg"
        json_dir.mkdir()
        jpg_dir.mkdir()

        # Create sample image
        img = Image.new("RGB", (800, 600), color="white")
        img_path = jpg_dir / "MI2_00001.jpg"
        img.save(img_path)

        # Create sample JSON
        json_data = {
            "source_data_info": {
                "source_data_name_jpg": "MI2_00001.jpg",
                "source_data_type": "report",
            },
            "learning_data_info": {
                "annotation": [
                    {
                        "instance_id": "Q00001",
                        "class_id": "V01",
                        "class_name": "표_재무제표",
                        "visual_instruction": "매출액은 얼마인가?",
                        "bounding_box": [100, 200, 300, 150],
                        "caption": "재무 표",
                    },
                    {
                        "instance_id": "Q00002",
                        "class_id": "T01",  # Not visual (no V*)
                        "class_name": "텍스트",
                        "visual_instruction": "텍스트는?",
                        "bounding_box": [50, 50, 100, 50],
                    },
                ]
            },
        }

        json_path = json_dir / "MI3_00001.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        return {
            "json_dir": str(json_dir),
            "jpg_dir": str(jpg_dir),
            "json_path": str(json_path),
            "img_path": str(img_path),
        }

    def test_dataset_initialization(self, sample_data):
        """Test dataset initialization and filtering."""
        vocab = Vocab(min_freq=1)
        vocab.build(["매출액은 얼마인가?"])

        dataset = UniDSet(
            [sample_data["json_path"]],
            jpg_dir=sample_data["jpg_dir"],
            vocab=vocab,
            resize_to=(512, 512),
        )

        # Should filter out non-visual annotation (T01)
        assert len(dataset) == 1  # Only V01 should remain

    def test_getitem_returns_correct_structure(self, sample_data):
        """Test __getitem__ returns correct sample structure."""
        vocab = Vocab(min_freq=1)
        vocab.build(["매출액은 얼마인가?"])

        dataset = UniDSet(
            [sample_data["json_path"]],
            jpg_dir=sample_data["jpg_dir"],
            vocab=vocab,
            resize_to=(512, 512),
        )

        sample = dataset[0]

        # Check required keys
        assert "image" in sample
        assert "query_ids" in sample
        assert "length" in sample
        assert "query_text" in sample
        assert "query_id" in sample
        assert "orig_size" in sample
        assert "target" in sample

        # Check types
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["query_ids"], torch.Tensor)
        assert isinstance(sample["length"], torch.Tensor)
        assert isinstance(sample["query_text"], str)
        assert isinstance(sample["query_id"], str)

    def test_getitem_image_shape(self, sample_data):
        """Test image is correctly resized."""
        vocab = Vocab(min_freq=1)
        vocab.build(["test"])

        dataset = UniDSet(
            [sample_data["json_path"]],
            jpg_dir=sample_data["jpg_dir"],
            vocab=vocab,
            resize_to=(512, 512),
        )

        sample = dataset[0]

        # Should be (C, H, W) format
        assert sample["image"].shape == (3, 512, 512)
        assert sample["image"].dtype == torch.float32

    def test_getitem_bbox_normalization(self, sample_data):
        """Test bounding box is normalized."""
        vocab = Vocab(min_freq=1)
        vocab.build(["test"])

        dataset = UniDSet(
            [sample_data["json_path"]],
            jpg_dir=sample_data["jpg_dir"],
            vocab=vocab,
            resize_to=(512, 512),
        )

        sample = dataset[0]

        # Target should be normalized [cx, cy, w, h] in [0, 1]
        assert sample["target"] is not None
        assert sample["target"].shape == (4,)
        assert torch.all(sample["target"] >= 0)
        assert torch.all(sample["target"] <= 1)

    def test_vocab_encoding(self, sample_data):
        """Test query is correctly encoded."""
        vocab = Vocab(min_freq=1)
        vocab.build(["매출액은 얼마인가?"])

        dataset = UniDSet(
            [sample_data["json_path"]],
            jpg_dir=sample_data["jpg_dir"],
            vocab=vocab,
            resize_to=(512, 512),
        )

        sample = dataset[0]

        # Query should be encoded
        assert len(sample["query_ids"]) > 0
        assert sample["length"] > 0
        assert sample["query_text"] == "매출액은 얼마인가?"


class TestCollateFn:
    """Tests for collate_fn batch collation."""

    @pytest.fixture
    def mock_samples(self):
        """Create mock samples for batching."""
        samples = [
            {
                "image": torch.randn(3, 512, 512),
                "query_ids": torch.tensor([1, 2, 3]),
                "length": torch.tensor(3),
                "query_text": "query 1",
                "query_id": "Q001",
                "orig_size": (800, 600),
                "class_name": "표",
                "target": torch.tensor([0.5, 0.5, 0.2, 0.3]),
            },
            {
                "image": torch.randn(3, 512, 512),
                "query_ids": torch.tensor([1, 2]),
                "length": torch.tensor(2),
                "query_text": "query 2",
                "query_id": "Q002",
                "orig_size": (800, 600),
                "class_name": "차트",
                "target": torch.tensor([0.3, 0.4, 0.1, 0.2]),
            },
        ]
        return samples

    def test_collate_batch_shapes(self, mock_samples):
        """Test batch collation produces correct shapes."""
        images, query_ids, lengths, targets, meta = collate_fn(mock_samples)

        batch_size = len(mock_samples)
        max_len = max(int(s["length"]) for s in mock_samples)

        assert images.shape == (batch_size, 3, 512, 512)
        assert query_ids.shape == (batch_size, max_len)
        assert lengths.shape == (batch_size,)
        assert len(targets) == batch_size
        assert len(meta) == batch_size

    def test_collate_query_padding(self, mock_samples):
        """Test queries are correctly padded."""
        images, query_ids, lengths, targets, meta = collate_fn(mock_samples)

        # First sample has length 3, second has length 2
        assert query_ids.shape[1] == 3  # max_len
        assert lengths[0] == 3
        assert lengths[1] == 2

        # Padded positions should be 0
        assert query_ids[1, 2] == 0  # Padded token

    def test_collate_metadata(self, mock_samples):
        """Test metadata is correctly collected."""
        images, query_ids, lengths, targets, meta = collate_fn(mock_samples)

        assert meta[0]["query_id"] == "Q001"
        assert meta[1]["query_id"] == "Q002"
        assert meta[0]["query_text"] == "query 1"
        assert meta[0]["orig_size"] == (800, 600)
