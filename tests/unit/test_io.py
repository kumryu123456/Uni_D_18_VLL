"""
Unit tests for I/O utilities.
"""

import os
import json
import tempfile
import pytest
from src.utils.io import (
    read_json,
    find_jsons,
    get_image_path,
    is_visual_ann,
    normalize_bbox,
    denormalize_bbox,
)


class TestJSONUtilities:
    """Tests for JSON reading utilities."""

    def test_read_json_success(self, tmp_path):
        """Test successful JSON reading."""
        json_file = tmp_path / "test.json"
        data = {"key": "value"}
        json_file.write_text(json.dumps(data))

        result = read_json(str(json_file))
        assert result == data

    def test_read_json_not_found(self):
        """Test FileNotFoundError when JSON doesn't exist."""
        with pytest.raises(FileNotFoundError):
            read_json("/nonexistent/file.json")

    def test_find_jsons(self, tmp_path):
        """Test finding JSON files in directory."""
        # Create test JSON files
        (tmp_path / "file1.json").write_text("{}")
        (tmp_path / "file2.json").write_text("{}")
        (tmp_path / "not_json.txt").write_text("")

        result = find_jsons(str(tmp_path))
        assert len(result) == 2
        assert all(f.endswith(".json") for f in result)

    def test_find_jsons_not_found(self):
        """Test FileNotFoundError for non-existent directory."""
        with pytest.raises(FileNotFoundError):
            find_jsons("/nonexistent/directory")


class TestIsVisualAnn:
    """Tests for visual annotation validation."""

    def test_valid_visual_ann_with_v_class(self):
        """Test valid annotation with V* class_id."""
        ann = {
            "class_id": "V01",
            "class_name": "표_재무제표",
            "visual_instruction": "매출액은?",
        }
        assert is_visual_ann(ann) is True

    def test_valid_visual_ann_with_table_keyword(self):
        """Test valid annotation with table keyword in class_name."""
        ann = {
            "class_id": "X01",
            "class_name": "표_데이터",
            "visual_instruction": "데이터는?",
        }
        assert is_visual_ann(ann) is True

    def test_valid_visual_ann_with_chart_keyword(self):
        """Test valid annotation with chart keyword."""
        ann = {
            "class_id": "X01",
            "class_name": "차트_막대그래프",
            "visual_instruction": "그래프는?",
        }
        assert is_visual_ann(ann) is True

    def test_invalid_no_query(self):
        """Test invalid annotation without query."""
        ann = {"class_id": "V01", "class_name": "표", "visual_instruction": ""}
        assert is_visual_ann(ann) is False

    def test_invalid_no_visual_element(self):
        """Test invalid annotation without visual element indicators."""
        ann = {
            "class_id": "T01",
            "class_name": "텍스트",
            "visual_instruction": "텍스트는?",
        }
        assert is_visual_ann(ann) is False


class TestBBoxNormalization:
    """Tests for bounding box normalization."""

    def test_normalize_bbox(self):
        """Test bbox normalization to [0,1] range."""
        bbox = [100, 200, 300, 150]  # x, y, w, h
        img_w, img_h = 800, 600

        cx, cy, nw, nh = normalize_bbox(bbox, img_w, img_h)

        # Check center coordinates
        assert abs(cx - 0.3125) < 1e-6  # (100 + 300/2) / 800
        assert abs(cy - 0.4583) < 1e-4  # (200 + 150/2) / 600

        # Check normalized size
        assert abs(nw - 0.375) < 1e-6  # 300 / 800
        assert abs(nh - 0.25) < 1e-6  # 150 / 600

    def test_denormalize_bbox(self):
        """Test bbox denormalization back to pixels."""
        normalized = (0.3125, 0.4583, 0.375, 0.25)
        img_w, img_h = 800, 600

        x, y, w, h = denormalize_bbox(normalized, img_w, img_h)

        # Check pixel coordinates (with small tolerance for float precision)
        assert abs(x - 100) < 1.0
        assert abs(y - 200) < 1.0
        assert abs(w - 300) < 1.0
        assert abs(h - 150) < 1.0

    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize->denormalize is identity."""
        bbox = [50, 75, 200, 180]
        img_w, img_h = 640, 480

        normalized = normalize_bbox(bbox, img_w, img_h)
        recovered = denormalize_bbox(normalized, img_w, img_h)

        for orig, rec in zip(bbox, recovered):
            assert abs(orig - rec) < 1.0  # Allow 1 pixel error
