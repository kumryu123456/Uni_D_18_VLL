# Feature Specification: Query-Based Visual Element Localization

**Competition**: Dacon Vision-Language Model Challenge
**Branch**: `001-query-based-visual-element-localization`
**Date**: 2025-11-15

## Overview

Develop a vision-language model that predicts the location of visual elements (tables, charts) in document images based on natural language queries. The model must understand both the content meaning (text) and visual structure (layout) to locate where users' desired information is visually positioned.

## Background

Complex documents (reports, papers, financial statements) often contain key information in tables and charts rather than plain text. Current systems primarily understand documents through text, forcing users to manually scan pages to find specific visual elements like "graph showing company A's revenue trend" or "table comparing model performance."

## Objectives

1. Accept document image and natural language query as input
2. Predict bounding box location (x, y, w, h) of visual elements semantically related to the query
3. Achieve competitive mIoU (Mean Intersection over Union) score on test data
4. Comply with competition rules: no LLM/VLM pretrained weights, no external data, no APIs

## Competition Constraints

### Data
- **Training**: `train_valid/` folder with report_jpg, report_json, press_jpg, press_json
- **Validation**: `val/` folder with same structure
- **Test**: `test/` folder with images and query JSON (no ground truth)
- **Output**: CSV with columns: `query_id`, `query_text`, `pred_x`, `pred_y`, `pred_w`, `pred_h`
- Bounding boxes: top-left origin (x, y, w, h) format

### Rules
- No external data
- No API usage
- Pre-trained models allowed EXCEPT LLM/VLM with language modeling
- CLIP allowed, LLaVA forbidden
- Python only
- No pseudo-labeling on test data
- Code must reproduce Private Score

### Evaluation
- **Metric**: mIoU (Mean Intersection over Union)
- **Public Score**: 50% of test samples
- **Private Score**: 100% of test samples
- **Final Scoring**: 80% accuracy + 20% model/data understanding

### Deliverables
- `model.py`: Model implementation
- `train.py`: Training script
- `test.py`: Inference script
- `preprocess.py`: Data preprocessing
- `README.md`: Execution instructions, environment, pretrained model sources
- `requirements.txt`: Dependencies

## Functional Requirements

### FR1: Data Loading
- Load document images (JPG) and metadata (JSON)
- Parse visual annotations with queries from JSON
- Extract bounding boxes for training samples
- Handle both report and press release document types

### FR2: Model Architecture
- Text encoder: Process natural language queries
- Image encoder: Extract visual features from document images
- Fusion mechanism: Combine text and visual features
- Bounding box prediction head: Output (cx, cy, w, h) normalized coordinates

### FR3: Training Pipeline
- Build vocabulary from training queries
- Resize images to consistent size
- Implement data augmentation (if applicable)
- Use smooth L1 loss for bounding box regression
- Support mixed precision training
- Save model checkpoints with vocabulary

### FR4: Inference Pipeline
- Load trained model and vocabulary
- Process test images and queries
- Predict bounding boxes in original image coordinates
- Output CSV in required format

### FR5: Evaluation
- Compute IoU between predicted and ground truth boxes
- Calculate mIoU across all validation samples
- Support evaluation mode with ground truth

## Non-Functional Requirements

### NFR1: Performance
- Target: Competitive mIoU score (baseline reference available)
- Training time: Reasonable on GPU (data volume is large)
- Inference speed: Batch processing support

### NFR2: Reproducibility
- Fixed random seeds
- Deterministic training
- Version-controlled dependencies

### NFR3: Code Quality
- Modular structure (separate model, train, test, preprocess)
- Clear documentation
- Error handling
- Logging for training progress

### NFR4: Compliance
- No forbidden pretrained weights (no LLM/VLM language models)
- All processing in code (no manual intervention)
- No data leakage from test set

## Technical Approach (Baseline Reference)

The competition provides baseline code with:
- ResNet18 image encoder (ImageNet weights allowed)
- GRU text encoder with custom vocabulary
- Cross-attention fusion
- Direct bounding box regression

Improvements to consider:
- Better image backbones (EfficientNet, ConvNeXt)
- Better text encoding (Transformer, CLIP text encoder)
- Advanced fusion (multi-head attention, FiLM layers)
- Multi-scale features
- Data augmentation strategies
- Loss function improvements (GIoU, DIoU)

## Success Criteria

1. Model trains without errors
2. Achieves mIoU > baseline on validation set
3. Generates valid submission CSV
4. Code passes reproducibility check
5. All deliverables meet competition requirements

## Out of Scope

- OCR or text extraction from images
- Multi-language support beyond provided data
- Real-time inference optimization
- Web UI or deployment infrastructure
