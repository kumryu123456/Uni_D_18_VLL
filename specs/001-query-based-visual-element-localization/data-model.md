# Data Model: Query-Based Visual Element Localization

**Date**: 2025-11-15
**Phase**: 1 (Design & Contracts)
**Status**: COMPLETED

## Overview

This document defines the data structures, schemas, and representations used throughout the vision-language model project.

---

## 1. Input Data Schema

### 1.1 JSON Annotation Format

**File**: `{report|press}_json/*.json` (e.g., `MI3_00001.json`)

```json
{
  "source_data_info": {
    "source_data_name_jpg": "MI2_00001.jpg",
    "source_data_type": "report",
    "source_data_date": "2025-01-15"
  },
  "learning_data_info": {
    "annotation": [
      {
        "instance_id": "Q00001",
        "class_id": "V01",
        "class_name": "표_재무제표",
        "visual_instruction": "2024년 매출액은 얼마인가?",
        "bounding_box": [120, 340, 560, 180],
        "caption": "연도별 재무 실적 표"
      },
      {
        "instance_id": "Q00002",
        "class_id": "V02",
        "class_name": "차트_막대그래프",
        "visual_instruction": "분기별 성장률 추이는?",
        "bounding_box": [150, 600, 480, 320],
        "caption": "분기별 성장률 그래프"
      }
    ]
  }
}
```

**Field Descriptions**:
- `instance_id`: Unique identifier for each query (Q##### format)
- `class_id`: Visual element type (V01=table, V02=chart, etc.)
- `class_name`: Human-readable class (Korean)
- `visual_instruction`: Natural language query (Korean)
- `bounding_box`: [x, y, w, h] in pixels (top-left origin)
- `caption`: Optional description of visual element

### 1.2 Test Set Query Format

**File**: `test/query/*.json`

```json
{
  "image_id": "TEST_00123",
  "queries": [
    {
      "query_id": "Q12345",
      "query_text": "매출 추이를 보여주는 그래프는?"
    },
    {
      "query_id": "Q12346",
      "query_text": "제품별 판매량 비교 표는?"
    }
  ]
}
```

**Note**: Test set has NO bounding_box field (predictions required).

### 1.3 Image Files

**Format**: JPEG (`.jpg`)
**Path Resolution**:
1. Check `source_data_name_jpg` in JSON
2. Look in corresponding `{type}_jpg/` folder
3. Fallback: Replace `MI3` → `MI2` in JSON filename

**Characteristics** (from research):
- Variable dimensions (scan/photo of documents)
- Aspect ratio: Typically A4 portrait (~1:1.4)
- Resolution: 150-300 DPI
- Color: RGB (may have scanning artifacts)

---

## 2. Internal Data Structures

### 2.1 Vocabulary

**Purpose**: Map Korean text tokens to integer IDs

```python
from typing import Dict, List

class Vocab:
    """
    Simple word-level vocabulary for Korean queries.

    Attributes:
        itos (List[str]): Index-to-string mapping
        stoi (Dict[str, int]): String-to-index mapping
        freq (Dict[str, int]): Token frequency counts
        min_freq (int): Minimum frequency threshold
    """
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.freq: Dict[str, int] = {}
        self.itos: List[str] = ["<pad>", "<unk>"]  # Special tokens
        self.stoi: Dict[str, int] = {"<pad>": 0, "<unk>": 1}

    def build(self, texts: List[str]) -> None:
        """Build vocabulary from list of text strings."""
        for text in texts:
            for token in self._tokenize(text):
                self.freq[token] = self.freq.get(token, 0) + 1

        # Add tokens meeting frequency threshold
        for token, count in sorted(self.freq.items(), key=lambda x: (-x[1], x[0])):
            if count >= self.min_freq and token not in self.stoi:
                idx = len(self.itos)
                self.itos.append(token)
                self.stoi[token] = idx

    def encode(self, text: str, max_len: int = 40) -> List[int]:
        """Convert text to list of token IDs."""
        tokens = self._tokenize(text)[:max_len]
        if not tokens:
            return [self.stoi["<unk>"]]
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return " ".join(self.itos[i] if i < len(self.itos) else "<unk>" for i in ids)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization with punctuation splitting."""
        text = text.replace("##", " ").replace(",", " ").replace("(", " ")
        text = text.replace(")", " ").replace(":", " ").replace("?", " ")
        text = text.replace("!", " ").replace("·", " ")
        return [t for t in text.strip().split() if t]

    def __len__(self) -> int:
        return len(self.itos)
```

**Size Estimation**: ~2,000-5,000 unique tokens (Korean + numbers + punctuation)

### 2.2 Dataset Sample

**Purpose**: Single training/validation/test example

```python
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class Sample:
    """
    Single training/inference sample.

    Attributes:
        image: (C, H, W) tensor, normalized RGB
        query_ids: (L,) tensor of token IDs
        length: Scalar tensor, valid length of query_ids
        query_text: Original query string
        query_id: Unique identifier (e.g., "Q00001")
        orig_size: (width, height) of original image
        class_name: Visual element type (e.g., "표_재무제표")
        target: (4,) tensor [cx, cy, w, h] normalized to [0,1], or None for test
    """
    image: torch.Tensor          # (3, H, W)
    query_ids: torch.Tensor      # (L,)
    length: torch.Tensor         # scalar
    query_text: str
    query_id: str
    orig_size: Tuple[int, int]   # (W, H)
    class_name: str
    target: Optional[torch.Tensor] = None  # (4,) or None
```

### 2.3 Batch Structure

**Purpose**: Collated batch for training/inference

```python
from typing import List, Dict, Any

def collate_fn(samples: List[Sample]) -> Tuple:
    """
    Collate variable-length samples into batch.

    Returns:
        images: (B, 3, H, W) tensor
        query_ids: (B, L_max) tensor, padded with 0
        lengths: (B,) tensor, valid lengths
        targets: List[Tensor | None], length B
        meta: List[Dict], metadata per sample
    """
    batch_size = len(samples)
    max_len = max(int(s.length) for s in samples)

    # Stack images
    images = torch.stack([s.image for s in samples], dim=0)  # (B, 3, H, W)

    # Pad query IDs
    query_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    for i, s in enumerate(samples):
        L = int(s.length)
        query_ids[i, :L] = s.query_ids[:L]
        lengths[i] = L

    # Collect targets and metadata
    targets = [s.target for s in samples]
    meta = [
        {
            "query_id": s.query_id,
            "query_text": s.query_text,
            "orig_size": s.orig_size,
            "class_name": s.class_name,
        }
        for s in samples
    ]

    return images, query_ids, lengths, targets, meta
```

---

## 3. Model State

### 3.1 Checkpoint Format

**Purpose**: Save/load trained model

```python
checkpoint = {
    # Model weights
    "model_state_dict": model.state_dict(),  # OrderedDict of tensors

    # Vocabulary (required for inference)
    "vocab_itos": vocab.itos,  # List[str]
    "vocab_stoi": vocab.stoi,  # Dict[str, int]

    # Hyperparameters (for model reconstruction)
    "config": {
        "img_size": 512,
        "dim": 256,
        "vocab_size": 3245,
        "backbone": "resnet18",
        "pretrained_backbone": True,
        "fusion_heads": 1,
    },

    # Training metadata
    "epoch": 10,
    "best_val_miou": 0.523,
    "optimizer_state_dict": optimizer.state_dict(),  # Optional

    # Reproducibility
    "random_seed": 42,
    "torch_version": "2.0.1",
}

# Save
torch.save(checkpoint, "outputs/ckpt/best_model.pth")

# Load
ckpt = torch.load("outputs/ckpt/best_model.pth", map_location="cpu")
```

**File Naming Convention**:
- `best_model.pth`: Best validation mIoU
- `epoch_{N}.pth`: Checkpoint at epoch N
- `final_model.pth`: After all training

---

## 4. Output Data Schema

### 4.1 Prediction CSV Format

**File**: `submission.csv`

**Required Columns**:
```csv
query_id,query_text,pred_x,pred_y,pred_w,pred_h
Q12345,"매출 추이 그래프는?",120.5,340.2,560.0,180.0
Q12346,"제품별 판매량 표는?",150.0,600.0,480.0,320.0
```

**Field Types**:
- `query_id`: String (Q##### format)
- `query_text`: String (original Korean query)
- `pred_x`: Float (top-left x in pixels)
- `pred_y`: Float (top-left y in pixels)
- `pred_w`: Float (width in pixels)
- `pred_h`: Float (height in pixels)

**Validation Rules**:
- All query_ids from test set must be present
- Bounding boxes non-negative: `pred_x >= 0`, `pred_y >= 0`, `pred_w > 0`, `pred_h > 0`
- No NaN or Inf values
- Encoding: UTF-8 with BOM (`utf-8-sig`)

### 4.2 Evaluation Metrics Output

**Purpose**: Training/validation monitoring

```python
@dataclass
class EvalMetrics:
    """Evaluation metrics for model performance."""
    miou: float                    # Mean IoU (primary metric)
    iou_per_class: Dict[str, float]  # IoU by visual element type
    iou_by_size: Dict[str, float]    # IoU by bbox size (small/medium/large)
    precision_at_50: float           # % predictions with IoU > 0.5
    precision_at_75: float           # % predictions with IoU > 0.75
    mean_center_error: float         # Mean L2 error of bbox centers (pixels)
    mean_size_error: float           # Mean relative size error
```

**Usage**:
```python
metrics = evaluate_model(model, val_loader)
print(f"Validation mIoU: {metrics.miou:.4f}")
```

---

## 5. Configuration Schema

### 5.1 Hyperparameters

```python
from dataclasses import dataclass

@dataclass
class Config:
    """Training and model configuration."""

    # Data
    train_json_dir: str = "./data/train/report_json"
    train_jpg_dir: str = "./data/train/report_jpg"
    val_json_dir: str = "./data/val/report_json"
    val_jpg_dir: str = "./data/val/report_jpg"

    # Model
    img_size: int = 512
    dim: int = 256  # Embedding dimension
    backbone: str = "resnet18"  # resnet18, resnet50, efficientnet_b3
    pretrained_backbone: bool = True
    text_encoder: str = "gru"  # gru, clip
    fusion_type: str = "cross_attn"  # cross_attn, film
    fusion_heads: int = 1  # For multi-head attention

    # Training
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # cosine, step, plateau
    loss_type: str = "smooth_l1"  # smooth_l1, giou, combined

    # Data loading
    num_workers: int = 2
    pin_memory: bool = True

    # Augmentation
    color_jitter: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.1
    rotation_degrees: float = 5.0

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Checkpointing
    ckpt_dir: str = "./outputs/ckpt"
    log_dir: str = "./outputs/logs"
    save_freq: int = 5  # Save every N epochs

    # Inference
    test_json_dir: str = "./data/test/query"
    test_jpg_dir: str = "./data/test/images"
    output_csv: str = "./outputs/preds/submission.csv"
```

### 5.2 Command-Line Interface

**train.py**:
```bash
python train.py \
  --json_dir ./data/train/report_json \
  --jpg_dir ./data/train/report_jpg \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-4 \
  --img_size 512 \
  --save_ckpt ./outputs/ckpt/model.pth
```

**test.py**:
```bash
python test.py \
  --ckpt ./outputs/ckpt/best_model.pth \
  --json_dir ./data/test/query \
  --jpg_dir ./data/test/images \
  --out_csv ./outputs/preds/submission.csv
```

---

## 6. Database/Storage

**Type**: File-based (no database)

**Directory Structure**:
```
project_root/
├── data/                  # Not in version control
│   ├── train/
│   ├── val/
│   └── test/
├── outputs/
│   ├── ckpt/             # Model checkpoints (.pth)
│   ├── preds/            # Prediction CSVs
│   └── logs/             # Training logs (optional)
├── src/                  # Source code
├── tests/                # Test suite
└── specs/                # Documentation
```

**Persistence**:
- **Checkpoints**: PyTorch `.pth` files
- **Predictions**: CSV files
- **Logs**: Text files or TensorBoard events (optional)

**Backup Strategy**:
- Save checkpoints every N epochs
- Keep best model by validation mIoU
- Version control: Only code, not data/checkpoints

---

## 7. Validation & Constraints

### 7.1 Data Validation

**On Dataset Load**:
- ✅ All JSON files have required fields
- ✅ Image files exist and readable
- ✅ Bounding boxes within image bounds
- ✅ Query text non-empty

**On Training**:
- ✅ Batch tensors have correct shapes
- ✅ No NaN/Inf in inputs or targets
- ✅ Labels in expected range ([0,1] for normalized coords)

### 7.2 Constraints

**Memory**:
- Single sample: ~10 MB (512×512 image + metadata)
- Batch of 8: ~80 MB
- Model: ~100 MB (ResNet18 baseline)
- Total training memory: <8 GB

**Disk**:
- Dataset: ~10-20 GB (images + JSON)
- Checkpoints: ~100 MB each
- Outputs: <1 GB

**Computation**:
- Training: 1-2 hours for 10 epochs (GPU)
- Inference: <1 minute for test set (GPU)

---

## 8. Entity Relationships

```
[Document Image] --1:N--> [Visual Elements]
                           |
                           +--1:1--> [Query]
                           |
                           +--1:1--> [Bounding Box]

[Query] --N:1--> [Vocabulary Token]

[Training Sample] --1:1--> [Image Tensor]
                  --1:1--> [Query Tensor]
                  --1:1--> [Target Bbox]

[Model] --1:1--> [Checkpoint]
        --1:1--> [Vocabulary]

[Prediction] --N:1--> [Query]
            --1:1--> [Predicted Bbox]
```

**Key Relationships**:
- Each document page may have multiple visual elements (tables/charts)
- Each visual element has one associated query
- Each query maps to one bounding box (ground truth or predicted)
- Vocabulary is shared across all queries

---

## Conclusion

All data structures defined:
- ✅ Input schemas (JSON, images)
- ✅ Internal representations (Vocab, Sample, Batch)
- ✅ Model state (Checkpoint format)
- ✅ Output schema (CSV predictions)
- ✅ Configuration (Hyperparameters)

**Next**: Create API contracts in `contracts/` directory.
