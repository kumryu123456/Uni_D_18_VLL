# Implementation Plan: Query-Based Visual Element Localization

**Branch**: `001-query-based-visual-element-localization` | **Date**: 2025-11-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-query-based-visual-element-localization/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build a vision-language model for Dacon competition that accepts document images and natural language queries to predict bounding box locations of visual elements (tables, charts). The system must achieve competitive mIoU scores while complying with strict rules: no LLM/VLM pretrained weights, no external data, no APIs. Core approach uses separate text and image encoders with cross-attention fusion for bounding box regression.

## Technical Context

**Language/Version**: Python 3.8+
**Primary Dependencies**: PyTorch, torchvision, PIL, numpy, pandas
**Storage**: File-based (JPG images, JSON metadata, CSV output, PTH checkpoints)
**Testing**: pytest (unit tests), integration tests for data pipeline and inference
**Target Platform**: Linux (WSL2), CUDA-enabled GPU for training
**Project Type**: Single (ML model with CLI interface)
**Performance Goals**: Competitive mIoU (target >0.5 based on task difficulty), batch training on GPU
**Constraints**:
- No LLM/VLM pretrained weights (CLIP allowed, LLaVA forbidden)
- No external data beyond competition dataset
- No API calls
- Large dataset volume (requires efficient data loading)
- Must reproduce Private Score exactly
**Scale/Scope**:
- Training samples: ~thousands (NEEDS CLARIFICATION from data exploration)
- Image size: Variable document pages (resize to 512x512 or NEEDS CLARIFICATION)
- Vocabulary size: Built from training queries (~hundreds-thousands tokens)
- Model size: Lightweight enough for training within time constraints (NEEDS CLARIFICATION on computational budget)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Constitution Status**: No constitution file found (template only). Creating project-specific principles:

### Core Principles for This Competition Project

1. **Reproducibility First**: All code must produce deterministic results with fixed seeds
2. **Compliance**: Strict adherence to competition rules (no forbidden models, no data leakage)
3. **Modularity**: Clear separation between model.py, train.py, test.py, preprocess.py
4. **Documentation**: README with setup, pretrained model sources, execution instructions
5. **Code Quality**: Error-free execution, proper dependency management

**Gates**:
- ✅ Single project structure (src/ + tests/)
- ✅ No external APIs
- ✅ No forbidden pretrained models
- ✅ Test-before-implement workflow applicable
- ✅ File-based I/O (images, JSON, CSV)

**Status**: PASS - No violations. Project aligns with competition requirements and standard ML project practices.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
├── data/                      # Dataset directory (not in git)
│   ├── train/
│   │   ├── report_jpg/
│   │   ├── report_json/
│   │   ├── press_jpg/
│   │   └── press_json/
│   ├── val/
│   │   └── [same structure]
│   └── test/
│       ├── images/
│       └── query/
├── src/
│   ├── data/
│   │   ├── dataset.py         # UniDSet, collate_fn, data loading
│   │   ├── vocab.py           # Vocab class
│   │   └── augmentation.py    # Data augmentation (optional)
│   ├── models/
│   │   ├── text_encoder.py    # TextEncoder (GRU-based)
│   │   ├── image_encoder.py   # ImageEncoder (ResNet/other)
│   │   ├── fusion.py          # CrossAttentionBBox
│   │   └── vlm.py             # CrossAttnVLM (main model)
│   ├── training/
│   │   ├── trainer.py         # Training loop logic
│   │   ├── loss.py            # Loss functions (smooth L1, GIoU, etc.)
│   │   └── metrics.py         # IoU, mIoU calculations
│   └── utils/
│       ├── config.py          # CFG class, hyperparameters
│       ├── io.py              # JSON/image path resolution
│       └── seed.py            # Random seed fixing
├── model.py                   # Model definition (imports from src.models)
├── train.py                   # Training script (CLI)
├── test.py                    # Inference script (CLI)
├── preprocess.py              # Data preprocessing utilities
├── outputs/
│   ├── ckpt/                  # Model checkpoints
│   └── preds/                 # Prediction CSVs
├── tests/
│   ├── unit/
│   │   ├── test_vocab.py
│   │   ├── test_dataset.py
│   │   ├── test_models.py
│   │   └── test_metrics.py
│   └── integration/
│       ├── test_training.py
│       └── test_inference.py
├── requirements.txt
└── README.md
```

**Structure Decision**: Single project structure selected. This is a standalone ML model with CLI interface, following competition requirements for model.py, train.py, test.py, preprocess.py deliverables. Source code organized by responsibility (data, models, training, utils) with clear separation of concerns.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

N/A - No constitution violations detected.

---

## Phase 0: Outline & Research

**Status**: COMPLETED
**Output**: `research.md`

### Research Tasks Identified

From Technical Context NEEDS CLARIFICATION items:

1. **Dataset Size & Characteristics**
   - Exact number of training/validation samples
   - Distribution of document types (report vs press)
   - Query length statistics
   - Visual element size distribution
   - Image dimensions and aspect ratios

2. **Optimal Image Size**
   - Balance between detail preservation and computational cost
   - Impact on small visual elements (tables/charts)
   - Memory constraints during training
   - Common choices: 512x512, 768x768, 1024x1024

3. **Model Capacity & Computational Budget**
   - Target model size for competition timeline
   - Expected training time per epoch
   - GPU memory requirements
   - Batch size feasibility

4. **Advanced Architecture Options**
   - CLIP text encoder (allowed) vs custom GRU
   - Better vision backbones: EfficientNet, ConvNeXt, Swin Transformer
   - Fusion mechanisms: FiLM, multi-head cross-attention, bilinear pooling
   - Multi-scale feature extraction

5. **Loss Function Improvements**
   - Smooth L1 (baseline) vs GIoU/DIoU/CIoU
   - Auxiliary losses: attention supervision, classification
   - Loss weighting strategies

6. **Data Augmentation Strategies**
   - Document-specific augmentations: brightness, contrast, slight rotation
   - Avoid breaking spatial relationships
   - Mixup/CutMix applicability

---

## Phase 1: Design & Contracts

**Status**: COMPLETED
**Prerequisites**: `research.md` complete ✅
**Outputs**: `data-model.md`, `contracts/`, `quickstart.md`

### Design Tasks

1. **Data Model** (`data-model.md`):
   - Dataset schema (JSON structure)
   - Sample dataclass definitions
   - Vocabulary representation
   - Checkpoint format

2. **API Contracts** (`contracts/`):
   - Model forward signature
   - Training function interface
   - Inference function interface
   - Data loader interface
   - Metrics computation interface

3. **Quickstart Guide** (`quickstart.md`):
   - Setup instructions
   - Data preparation steps
   - Training command examples
   - Inference command examples
   - Troubleshooting common issues

---

## Phase 2: Task Planning

**Status**: NOT STARTED (executed by `/speckit.tasks`)
**Prerequisites**: Phase 1 complete
**Output**: `tasks.md`

This phase is handled by the `/speckit.tasks` command and is not part of `/speckit.plan`.
