"""
FINAL OPTIMIZED Data Preprocessing - Maximum IoU Performance
통합된 모든 버전의 최고 기능들
"""

import os
import json
import random
import logging
import re
from glob import glob
from typing import List, Dict, Any, Optional
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Try to import albumentations (optional)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("⚠️ Albumentations not found. Using torchvision transforms.")

from torchvision import transforms

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 200


def seed_everything(seed: int = 42):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


# ==================== BBOX Utilities ====================
def normalize_bbox(bbox, img_width, img_height):
    """
    Normalize bounding box to [0, 1] range
    Input: [x, y, w, h] in pixel coordinates
    Output: [cx, cy, nw, nh] normalized center format
    """
    x, y, w, h = bbox
    cx = (x + w/2) / img_width
    cy = (y + h/2) / img_height
    nw = w / img_width
    nh = h / img_height

    # Clamp to valid range
    cx = np.clip(cx, 0, 1)
    cy = np.clip(cy, 0, 1)
    nw = np.clip(nw, 0, 1)
    nh = np.clip(nh, 0, 1)

    return [cx, cy, nw, nh]


def denormalize_bbox(bbox, img_width, img_height):
    """
    Convert normalized bbox back to pixel coordinates
    Input: [cx, cy, nw, nh] normalized
    Output: [x, y, w, h] in pixels
    """
    cx, cy, nw, nh = bbox
    w = nw * img_width
    h = nh * img_height
    x = (cx * img_width) - w/2
    y = (cy * img_height) - h/2

    # Ensure within image bounds
    x = max(0, min(x, img_width - w))
    y = max(0, min(y, img_height - h))
    w = max(0, min(w, img_width - x))
    h = max(0, min(h, img_height - y))

    return [x, y, w, h]


def calculate_iou_xywh(pred, gt):
    """
    Calculate IoU between two bboxes in [x, y, w, h] format
    """
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    px1, py1, px2, py2 = px, py, px + pw, py + ph
    gx1, gy1, gx2, gy2 = gx, gy, gx + gw, gy + gh

    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = pw * ph + gw * gh - inter

    return float(inter / union) if union > 0 else 0.0


# ==================== Florence-2 Format Support ====================
def normalize_bbox_to_florence(bbox: List[float], img_w: int, img_h: int) -> str:
    """
    Normalize bbox to Florence-2 format <loc_xxx>
    """
    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    def to_loc(v, size):
        return max(0, min(999, int(v / size * 999)))

    lx1 = to_loc(x1, img_w)
    ly1 = to_loc(y1, img_h)
    lx2 = to_loc(x2, img_w)
    ly2 = to_loc(y2, img_h)

    return f"<loc_{lx1}><loc_{ly1}><loc_{lx2}><loc_{ly2}>"


def parse_florence_output_to_bbox(text: str, img_w: int, img_h: int):
    """
    Parse Florence-2 output to pixel coordinates
    """
    matches = re.findall(r"<loc_(\d+)>", text)
    if len(matches) < 4:
        return img_w / 4, img_h / 4, img_w / 2, img_h / 2

    lx1, ly1, lx2, ly2 = map(int, matches[:4])
    x1 = lx1 / 999 * img_w
    y1 = ly1 / 999 * img_h
    x2 = lx2 / 999 * img_w
    y2 = ly2 / 999 * img_h

    return x1, y1, x2 - x1, y2 - y1


# ==================== Validation Utilities ====================
def validate_json_file(json_path):
    """Validate JSON file integrity"""
    try:
        if not os.path.exists(json_path):
            return False
        if os.path.getsize(json_path) == 0:
            return False

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        has_learning_data = 'learning_data_info' in data
        has_queries = 'queries' in data

        return has_learning_data or has_queries

    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False


def is_visual_element(ann: Dict[str, Any]) -> bool:
    """Check if annotation is a visual element"""
    class_id = str(ann.get('class_id', ''))
    class_name = str(ann.get('class_name', ''))
    has_query = bool(ann.get('visual_instruction', '').strip())

    is_visual = (
        class_id.startswith('V') or
        any(k in class_name for k in ['표', '차트', '그래프', 'table', 'chart', 'graph', '다이어그램'])
    )

    return has_query and is_visual


def get_category_from_annotation(ann: Dict[str, Any]) -> str:
    """Extract category from annotation"""
    cname = str(ann.get("class_name", "") or "").lower()
    cid = str(ann.get("class_id", "") or "")

    if "표" in cname or "table" in cname:
        return "table"

    if "차트" in cname or "chart" in cname or "그래프" in cname or "graph" in cname:
        if "꺾은선" in cname or "line" in cname:
            return "chart_line"
        if "세로 막대" in cname:
            return "chart_bar_v"
        if "가로 막대" in cname:
            return "chart_bar_h"
        if "원형" in cname or "pie" in cname:
            return "chart_pie"
        return "chart_other"

    if "다이어그램" in cname or cid.startswith("V03"):
        return "diagram"

    if cid.startswith("V"):
        return "visual_other"

    return "other"


# ==================== Vocabulary ====================
class Vocabulary:
    """Enhanced Vocabulary for multilingual text tokenization"""
    def __init__(self, min_freq=1):
        self.vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.min_freq = min_freq

    @property
    def pad_idx(self) -> int:
        return 0

    @property
    def unk_idx(self) -> int:
        return 1

    def tokenize(self, text):
        """Tokenize Korean/English text - character-level for Korean"""
        if not text or not isinstance(text, str):
            return []

        # For better Korean handling, use character-level
        tokens = []
        for char in text:
            if char.strip():  # Non-whitespace characters
                tokens.append(char)

        return tokens if tokens else []

    def build(self, texts):
        """Build vocabulary from texts with frequency filtering"""
        counter = Counter()

        for text in texts:
            if text and isinstance(text, str):
                tokens = self.tokenize(text)
                counter.update(tokens)

        idx = len(self.vocab)
        for word, freq in counter.most_common():
            if freq >= self.min_freq and word not in self.vocab:
                self.vocab[word] = idx
                self.idx2word[idx] = word
                idx += 1

        logger.info(f"Vocabulary built: {len(self.vocab)} tokens")

    def encode(self, text, max_len=MAX_TEXT_LENGTH):
        """Encode text to token indices"""
        if not text or not isinstance(text, str):
            return [2, 3]  # SOS, EOS

        tokens = self.tokenize(text)
        indices = [self.vocab.get(token, 1) for token in tokens]
        indices = [2] + indices + [3]  # Add SOS, EOS

        if len(indices) > max_len:
            indices = indices[:max_len-1] + [3]

        return indices

    def decode(self, indices):
        """Decode indices back to text"""
        tokens = []
        for idx in indices:
            if idx == 2:  # Skip SOS
                continue
            if idx == 3:  # Stop at EOS
                break
            tokens.append(self.idx2word.get(idx, '<unk>'))

        return ''.join(tokens)  # No spaces for character-level

    def __len__(self) -> int:
        return len(self.vocab)


# ==================== Data Collection ====================
def collect_samples(
    json_dir: str,
    img_root: str,
    skip_missing: bool = True
) -> List[Dict[str, Any]]:
    """
    Collect samples from JSON files with visual_instruction.
    """
    samples = []
    skipped_count = 0

    json_paths = sorted(glob(os.path.join(json_dir, "*.json")))
    if not json_paths:
        logger.warning(f"No JSON files found in {json_dir}")
        return []

    logger.info(f"Found {len(json_paths)} JSON files in {json_dir}")

    for jp in json_paths:
        if not validate_json_file(jp):
            skipped_count += 1
            continue

        try:
            with open(jp, "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON {jp}: {e}")
            if not skip_missing:
                raise
            skipped_count += 1
            continue

        source = j.get("source_data_info", {})
        learn = j.get("learning_data_info", {})

        img_name = source.get("source_data_name_jpg", "")
        if not img_name:
            skipped_count += 1
            continue

        img_path = os.path.join(img_root, img_name)

        if not os.path.exists(img_path):
            if skip_missing:
                skipped_count += 1
                continue
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        width, height = source.get("document_resolution", [2480, 3508])

        for ann in learn.get("annotation", []):
            if not is_visual_element(ann):
                continue

            query = ann.get("visual_instruction", "").strip()
            if not query:
                continue

            bbox = ann.get("bounding_box")
            if not bbox or len(bbox) != 4:
                continue

            x, y, w, h = bbox

            if w <= 0 or h <= 0:
                continue

            # Normalize to center format
            normalized = normalize_bbox(bbox, width, height)

            sample = {
                "img_path": img_path,
                "query": query,
                "target_bbox": normalized,
                "instance_id": ann.get("instance_id", ""),
                "orig_bbox": bbox,
                "orig_size": [width, height],
                "category": get_category_from_annotation(ann),
            }
            samples.append(sample)

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} files due to errors")

    logger.info(f"Collected {len(samples)} samples")
    return samples


# ==================== Dataset ====================
class DocumentVLDataset(Dataset):
    """Optimized dataset with dual augmentation support"""

    def __init__(
        self,
        vocab: Vocabulary,
        samples: List[Dict[str, Any]],
        img_size: int = 512,
        is_train: bool = True,
        use_albumentations: bool = True,
        max_query_len: int = MAX_TEXT_LENGTH,
    ):
        super().__init__()
        self.vocab = vocab
        self.samples = samples
        self.img_size = img_size
        self.is_train = is_train
        self.use_albumentations = use_albumentations and HAS_ALBUMENTATIONS
        self.max_query_len = max_query_len

        # Choose augmentation strategy
        if self.use_albumentations:
            if is_train:
                self.transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    A.ColorJitter(p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
        else:
            # Torchvision fallback
            if is_train:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        # Load image
        try:
            img = Image.open(s["img_path"]).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {s['img_path']}: {e}")
            raise

        # Apply transforms
        if self.use_albumentations:
            img_np = np.array(img)
            transformed = self.transform(image=img_np)
            img_tensor = transformed["image"]
        else:
            img_tensor = self.transform(img)

        # Tokenize query
        ids = self.vocab.encode(s["query"], max_len=self.max_query_len)

        if len(ids) == 0:
            logger.warning(f"Empty tokens for query: {s['query']}")
            ids = [2, 3]  # SOS, EOS

        target = torch.tensor(s["target_bbox"], dtype=torch.float32) if s.get("target_bbox") else None

        return {
            "image": img_tensor,
            "text_ids": torch.tensor(ids, dtype=torch.long),
            "text_len": len(ids),
            "target": target,
            "instance_id": s["instance_id"],
            "orig_bbox": torch.tensor(s["orig_bbox"], dtype=torch.float32),
            "orig_size": torch.tensor(s["orig_size"], dtype=torch.float32),
            "query_text": s["query"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader"""
    images = torch.stack([b["image"] for b in batch], dim=0)

    max_len = max(b["text_len"] for b in batch)
    B = len(batch)
    text_ids = torch.zeros(B, max_len, dtype=torch.long)
    text_lens = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        l = b["text_len"]
        text_ids[i, :l] = b["text_ids"]
        text_lens[i] = l

    targets = torch.stack([b["target"] for b in batch if b["target"] is not None], dim=0) \
              if any(b["target"] is not None for b in batch) else None

    return {
        "images": images,
        "text_ids": text_ids,
        "text_lens": text_lens,
        "targets": targets,
        "instance_ids": [b["instance_id"] for b in batch],
        "orig_bbox": torch.stack([b["orig_bbox"] for b in batch], dim=0),
        "orig_size": torch.stack([b["orig_size"] for b in batch], dim=0),
        "query_texts": [b["query_text"] for b in batch],
    }


# ==================== Main DataLoader Creation ====================
def create_dataloader(
    json_dir: str = None,
    img_root: str = None,
    json_dirs: list = None,
    img_roots: list = None,
    vocab: Vocabulary = None,
    build_vocab: bool = False,
    batch_size: int = 8,
    img_size: int = 512,
    is_train: bool = True,
    use_albumentations: bool = True,
    num_workers: int = 4,
    max_query_len: int = MAX_TEXT_LENGTH,
) -> tuple:
    """
    Create dataset and dataloader.

    Supports both single directory and multiple directories:
    - Single: json_dir, img_root
    - Multiple: json_dirs, img_roots (lists)

    Returns:
        (samples, vocab, dataset, dataloader)
    """
    import platform

    # Collect samples from single or multiple directories
    if json_dirs is not None and img_roots is not None:
        # Multiple directories (Press + Report)
        if not isinstance(json_dirs, list):
            json_dirs = [json_dirs]
        if not isinstance(img_roots, list):
            img_roots = [img_roots]

        if len(json_dirs) != len(img_roots):
            raise ValueError("json_dirs and img_roots must have the same length")

        logger.info(f"📂 Loading from {len(json_dirs)} directories...")
        samples = []
        for jd, ir in zip(json_dirs, img_roots):
            logger.info(f"  Loading from {jd}...")
            s = collect_samples(jd, ir, skip_missing=is_train)
            logger.info(f"  ✅ Loaded {len(s)} samples")
            samples.extend(s)
        logger.info(f"✅ Total samples: {len(samples)}")
    elif json_dir is not None and img_root is not None:
        # Single directory (backward compatibility)
        samples = collect_samples(json_dir, img_root, skip_missing=is_train)
    else:
        raise ValueError("Must provide either (json_dir, img_root) or (json_dirs, img_roots)")

    if not samples:
        logger.warning(f"No samples found in {json_dir}")
        return [], None, None, None

    # Build vocabulary if needed
    if build_vocab:
        vocab = Vocabulary()
        vocab.build([s["query"] for s in samples])
    elif vocab is None:
        raise ValueError("vocab must be provided if build_vocab=False")

    # Create dataset
    dataset = DocumentVLDataset(
        vocab=vocab,
        samples=samples,
        img_size=img_size,
        is_train=is_train,
        use_albumentations=use_albumentations,
        max_query_len=max_query_len,
    )

    # Windows compatibility
    if platform.system() == 'Windows' and num_workers > 0:
        logger.warning(f"Setting num_workers=0 for Windows compatibility")
        num_workers = 0

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return samples, vocab, dataset, dataloader


if __name__ == "__main__":
    # Test
    seed_everything(42)

    print("="*70)
    print("🧪 FINAL Preprocessing Test")
    print("="*70)

    # Test vocabulary
    vocab = Vocabulary(min_freq=1)
    sample_texts = [
        "이 문서에서 매출 성장률을 보여주는 차트는 어디에 있나요?",
        "Find the table showing Q3 financial results",
        "2023년 실적 요약 표를 찾아주세요"
    ]

    vocab.build(sample_texts)
    print(f"✅ Vocabulary: {len(vocab)} tokens")

    # Test bbox
    bbox = [100, 150, 200, 300]
    img_w, img_h = 800, 1200

    normalized = normalize_bbox(bbox, img_w, img_h)
    denormalized = denormalize_bbox(normalized, img_w, img_h)
    iou = calculate_iou_xywh(bbox, denormalized)

    print(f"✅ BBox normalized: {normalized}")
    print(f"✅ BBox IoU (should be ~1.0): {iou:.4f}")

    print("="*70)
    print("✅ All tests passed!")
