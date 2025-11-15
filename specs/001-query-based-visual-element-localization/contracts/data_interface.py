"""
Data Interface Contract

Defines the interface for data loading and processing components.
"""

from typing import Protocol, List, Tuple, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
import torch


class VocabularyProtocol(Protocol):
    """Vocabulary interface for text encoding."""

    itos: List[str]  # Index to string
    stoi: Dict[str, int]  # String to index

    def build(self, texts: List[str]) -> None:
        """Build vocabulary from text corpus."""
        ...

    def encode(self, text: str, max_len: int = 40) -> List[int]:
        """Convert text to token ID list."""
        ...

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        ...

    def __len__(self) -> int:
        """Return vocabulary size."""
        ...


class DocumentDatasetProtocol(Protocol):
    """Dataset interface for document image + query pairs."""

    def __len__(self) -> int:
        """Return number of samples."""
        ...

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get single sample.

        Returns:
            Dictionary with keys:
                - image: (3, H, W) tensor
                - query_ids: (L,) tensor
                - length: scalar tensor
                - query_text: str
                - query_id: str
                - orig_size: (W, H) tuple
                - class_name: str
                - target: (4,) tensor or None
        """
        ...


# Function signatures

def create_dataloader(
    json_dir: str,
    jpg_dir: str,
    vocab: Optional[Any] = None,
    build_vocab: bool = False,
    batch_size: int = 8,
    img_size: int = 512,
    num_workers: int = 2,
    shuffle: bool = False,
) -> Tuple[Dataset, DataLoader]:
    """
    Create dataset and dataloader.

    Args:
        json_dir: Path to JSON annotation directory
        jpg_dir: Path to JPG image directory
        vocab: Existing vocabulary or None (create new)
        build_vocab: Whether to build vocab from this data
        batch_size: Batch size
        img_size: Target image size (square)
        num_workers: DataLoader workers
        shuffle: Whether to shuffle

    Returns:
        dataset: Dataset instance
        dataloader: DataLoader instance
    """
    ...


def collate_fn(
    batch: List[Dict[str, Any]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]], List[Dict]]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from dataset

    Returns:
        images: (B, 3, H, W) tensor
        query_ids: (B, L_max) tensor, padded
        lengths: (B,) tensor
        targets: List of (4,) tensors or None
        metadata: List of dicts with query_id, query_text, etc.
    """
    ...


def find_json_files(json_dir: str) -> List[str]:
    """
    Find all JSON files in directory.

    Args:
        json_dir: Path to JSON directory

    Returns:
        Sorted list of JSON file paths
    """
    ...


def read_json(json_path: str) -> Dict[str, Any]:
    """
    Read and parse JSON annotation file.

    Args:
        json_path: Path to JSON file

    Returns:
        Parsed JSON data as dictionary
    """
    ...


def get_image_path(json_path: str, json_data: Dict, jpg_dir: Optional[str] = None) -> str:
    """
    Resolve image path from JSON metadata.

    Args:
        json_path: Path to JSON annotation file
        json_data: Parsed JSON data
        jpg_dir: Optional explicit JPG directory

    Returns:
        Resolved path to corresponding JPG image

    Raises:
        FileNotFoundError: If image cannot be located
    """
    ...


def parse_annotations(json_data: Dict) -> List[Dict[str, Any]]:
    """
    Extract visual element annotations from JSON.

    Args:
        json_data: Parsed JSON data

    Returns:
        List of annotation dicts with keys:
            - instance_id: Query ID
            - class_id: Element class ID
            - class_name: Element class name
            - visual_instruction: Query text
            - bounding_box: [x, y, w, h] or None
            - caption: Optional caption
    """
    ...


def is_valid_annotation(annotation: Dict) -> bool:
    """
    Check if annotation is valid for training.

    Args:
        annotation: Single annotation dict

    Returns:
        True if annotation has query and is visual element (V* or table/chart)
    """
    ...


def normalize_bbox(
    bbox: List[float], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Normalize bounding box to [0,1] range.

    Args:
        bbox: [x, y, w, h] in pixels
        img_width: Image width
        img_height: Image height

    Returns:
        (cx, cy, nw, nh) normalized center coordinates and size
    """
    ...


def denormalize_bbox(
    normalized: Tuple[float, float, float, float], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert normalized bbox back to pixel coordinates.

    Args:
        normalized: (cx, cy, nw, nh) in [0,1]
        img_width: Image width
        img_height: Image height

    Returns:
        (x, y, w, h) in pixels (top-left corner format)
    """
    ...
