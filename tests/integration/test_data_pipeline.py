"""
Integration tests for data loading pipeline.
"""

import os
import json
import tempfile
import pytest
import torch
from PIL import Image

try:
    from src.data.dataset import UniDSet, make_loader
    from src.data.vocab import Vocab
except ImportError:
    pytest.skip("Dataset module not yet implemented", allow_module_level=True)


class TestDataPipeline:
    """Integration tests for complete data loading pipeline."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a minimal dataset for integration testing."""
        json_dir = tmp_path / "json"
        jpg_dir = tmp_path / "jpg"
        json_dir.mkdir()
        jpg_dir.mkdir()

        # Create multiple sample images and JSONs
        for i in range(3):
            # Create image
            img = Image.new("RGB", (800, 600), color=(100 + i * 50, 100, 100))
            img_path = jpg_dir / f"MI2_0000{i+1}.jpg"
            img.save(img_path)

            # Create JSON
            json_data = {
                "source_data_info": {
                    "source_data_name_jpg": f"MI2_0000{i+1}.jpg",
                },
                "learning_data_info": {
                    "annotation": [
                        {
                            "instance_id": f"Q0000{i+1}",
                            "class_id": "V01",
                            "class_name": "표",
                            "visual_instruction": f"데이터 {i+1}",
                            "bounding_box": [100 + i * 10, 200, 300, 150],
                        }
                    ]
                },
            }

            json_path = json_dir / f"MI3_0000{i+1}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f)

        return {"json_dir": str(json_dir), "jpg_dir": str(jpg_dir)}

    def test_full_pipeline_vocab_building(self, sample_dataset):
        """Test complete pipeline with vocabulary building."""
        dataset, dataloader = make_loader(
            sample_dataset["json_dir"],
            sample_dataset["jpg_dir"],
            vocab=None,
            build_vocab=True,
            batch_size=2,
            shuffle=False,
        )

        # Vocab should be built
        if hasattr(dataset, "dataset"):  # Subset wrapper
            vocab = dataset.dataset.vocab
        else:
            vocab = dataset.vocab

        assert len(vocab) > 2  # More than just <pad>, <unk>
        assert "데이터" in vocab.stoi

    def test_dataloader_iteration(self, sample_dataset):
        """Test iterating through dataloader."""
        dataset, dataloader = make_loader(
            sample_dataset["json_dir"],
            sample_dataset["jpg_dir"],
            vocab=None,
            build_vocab=True,
            batch_size=2,
            shuffle=False,
        )

        # Iterate through all batches
        batch_count = 0
        total_samples = 0

        for images, query_ids, lengths, targets, meta in dataloader:
            batch_count += 1
            batch_size = images.size(0)
            total_samples += batch_size

            # Check batch structure
            assert images.dim() == 4  # (B, C, H, W)
            assert query_ids.dim() == 2  # (B, L)
            assert lengths.dim() == 1  # (B,)
            assert len(targets) == batch_size
            assert len(meta) == batch_size

            # Check all samples have targets (training mode)
            assert all(t is not None for t in targets)

        # Should have processed all samples
        assert total_samples == 3  # 3 samples created

    def test_batch_consistency(self, sample_dataset):
        """Test batch data consistency."""
        vocab = Vocab(min_freq=1)
        vocab.build(["데이터 1", "데이터 2", "데이터 3"])

        dataset, dataloader = make_loader(
            sample_dataset["json_dir"],
            sample_dataset["jpg_dir"],
            vocab=vocab,
            build_vocab=False,
            batch_size=2,
            shuffle=False,
        )

        for images, query_ids, lengths, targets, meta in dataloader:
            # Images should be normalized floats
            assert images.dtype == torch.float32
            assert images.min() >= 0.0
            assert images.max() <= 1.0

            # Query IDs should be valid indices
            assert query_ids.dtype == torch.long
            assert query_ids.min() >= 0
            assert query_ids.max() < len(vocab)

            # Lengths should match actual query lengths
            for i in range(len(lengths)):
                actual_len = int(lengths[i])
                assert actual_len > 0
                assert actual_len <= query_ids.size(1)

            # Targets should be normalized
            for target in targets:
                if target is not None:
                    assert target.shape == (4,)
                    assert torch.all(target >= 0)
                    assert torch.all(target <= 1)

    def test_pipeline_error_handling(self, tmp_path):
        """Test pipeline handles missing data gracefully."""
        # Empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(RuntimeError, match="No supervised samples"):
            make_loader(
                str(empty_dir),
                str(empty_dir),
                vocab=None,
                build_vocab=True,
                batch_size=2,
            )

    def test_inference_mode(self, sample_dataset):
        """Test dataloader in inference mode (no bbox filtering)."""
        # Create JSON without bbox for test scenario
        json_dir = sample_dataset["json_dir"]
        jpg_dir = sample_dataset["jpg_dir"]

        test_json = {
            "source_data_info": {"source_data_name_jpg": "MI2_00001.jpg"},
            "learning_data_info": {
                "annotation": [
                    {
                        "instance_id": "Q99999",
                        "class_id": "V01",
                        "class_name": "표",
                        "visual_instruction": "테스트 쿼리",
                        # No bounding_box (inference mode)
                    }
                ]
            },
        }

        test_json_path = os.path.join(json_dir, "test.json")
        with open(test_json_path, "w", encoding="utf-8") as f:
            json.dump(test_json, f)

        vocab = Vocab(min_freq=1)
        vocab.build(["테스트 쿼리"])

        # Should not filter samples without bbox when not building vocab
        from src.utils.io import find_jsons

        json_files = find_jsons(json_dir)
        dataset = UniDSet(
            json_files, jpg_dir=jpg_dir, vocab=vocab, build_vocab=False
        )

        # Should include all visual annotations
        assert len(dataset) >= 3  # Original 3 + test sample
