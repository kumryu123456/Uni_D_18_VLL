"""
I/O utilities for reading JSON annotations and resolving image paths.
"""

import os
import json
from glob import glob
from typing import List, Dict, Any, Tuple


def read_json(path: str) -> Dict[str, Any]:
    """
    Read and parse JSON annotation file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data as dictionary

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_jsons(json_dir: str) -> List[str]:
    """
    Find all JSON files in directory.

    Args:
        json_dir: Path to JSON directory

    Returns:
        Sorted list of JSON file paths

    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"json_dir not found: {json_dir}")

    return sorted(glob(os.path.join(json_dir, "*.json")))


def get_image_path(json_path: str, data: Dict[str, Any], jpg_dir: str = None) -> str:
    """
    Resolve image path from JSON metadata.

    Resolution strategy (per data-model.md):
    1. Check source_data_name_jpg in JSON
    2. Look in corresponding jpg_dir if provided
    3. Fallback: Replace 'json' with 'jpg' in path
    4. Last resort: Replace MI3 → MI2 in filename

    Args:
        json_path: Path to JSON annotation file
        data: Parsed JSON data
        jpg_dir: Optional explicit JPG directory

    Returns:
        Resolved path to corresponding JPG image

    Raises:
        FileNotFoundError: If image cannot be located
    """
    # Prefer explicit mapping via source_data_name_jpg
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg", None)

    if jpg_dir and jpg_name:
        path = os.path.join(jpg_dir, jpg_name)
        if os.path.exists(path):
            return path

    # Fallback: .../json/... → .../jpg/...
    if jpg_name:
        maybe = json_path.replace(os.sep + "json" + os.sep, os.sep + "jpg" + os.sep)
        maybe = os.path.join(os.path.dirname(maybe), jpg_name) if os.path.isdir(os.path.dirname(maybe)) else maybe
        if os.path.exists(maybe):
            return maybe

    # Last resort: same dir, MI3 → MI2.jpg
    base = os.path.splitext(os.path.basename(json_path))[0]
    sibling = os.path.join(os.path.dirname(json_path), base.replace("MI3", "MI2") + ".jpg")
    if os.path.exists(sibling):
        return sibling

    raise FileNotFoundError(f"Could not resolve JPG for {json_path} (jpg_dir={jpg_dir})")


def is_visual_ann(a: dict) -> bool:
    """
    Check if annotation is valid visual element for training.

    Validation criteria:
    - Has non-empty visual_instruction (query)
    - Is visual element (V* class_id OR table/chart class_name)

    Args:
        annotation: Single annotation dict

    Returns:
        True if annotation is valid visual element
    """
    cid = str(a.get("class_id", "") or "")
    cname = str(a.get("class_name", "") or "")
    has_q = bool(str(a.get("visual_instruction", "") or "").strip())

    looks_visual = cid.startswith("V") or any(
        k in cname for k in ["표", "차트", "그래프", "chart", "table"]
    )

    return has_q and looks_visual


def normalize_bbox(
    bbox: List[float], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Normalize bounding box to [0,1] range.

    Converts from top-left (x, y, w, h) to normalized center (cx, cy, nw, nh).

    Args:
        bbox: [x, y, w, h] in pixels (top-left origin)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        (cx, cy, nw, nh) normalized to [0,1] where:
            cx, cy: center coordinates
            nw, nh: normalized width and height
    """
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_width
    cy = (y + h / 2.0) / img_height
    nw = w / img_width
    nh = h / img_height
    return (cx, cy, nw, nh)


def denormalize_bbox(
    normalized: Tuple[float, float, float, float], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert normalized bbox back to pixel coordinates.

    Converts from normalized center (cx, cy, nw, nh) to top-left (x, y, w, h).

    Args:
        normalized: (cx, cy, nw, nh) in [0,1]
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        (x, y, w, h) in pixels (top-left origin format)
    """
    cx, cy, nw, nh = normalized
    w = nw * img_width
    h = nh * img_height
    x = (cx - nw / 2.0) * img_width
    y = (cy - nh / 2.0) * img_height
    return (x, y, w, h)


def read_json_annotation(json_path: str) -> List[Dict[str, Any]]:
    """
    Read JSON annotation file(s) and return list of all data items.

    This function can handle:
    - Single JSON file path: returns [data]
    - Directory path: reads all .json files in directory
    - Glob pattern: reads all matching files

    Args:
        json_path: Path to JSON file, directory, or glob pattern

    Returns:
        List of parsed JSON data dictionaries

    Raises:
        FileNotFoundError: If path doesn't exist
    """
    results = []

    # Check if it's a directory
    if os.path.isdir(json_path):
        json_files = find_jsons(json_path)
        for jf in json_files:
            data = read_json(jf)
            results.append(data)
    # Check if it's a file
    elif os.path.isfile(json_path):
        data = read_json(json_path)
        results.append(data)
    # Try as glob pattern
    else:
        matching_files = glob(json_path)
        if not matching_files:
            raise FileNotFoundError(f"No files found matching: {json_path}")
        for jf in sorted(matching_files):
            data = read_json(jf)
            results.append(data)

    return results
