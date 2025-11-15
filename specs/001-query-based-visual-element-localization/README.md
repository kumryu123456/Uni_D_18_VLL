# Feature 001: Query-Based Visual Element Localization

**Branch**: `001-query-based-visual-element-localization`
**Status**: Planning Complete ✅
**Created**: 2025-11-15

## Overview

Vision-language model for Dacon competition that predicts bounding box locations of visual elements (tables, charts) in document images based on natural language queries.

## Competition Details

- **Metric**: mIoU (Mean Intersection over Union)
- **Constraints**: No LLM/VLM pretrained weights, no external data, no APIs
- **Deliverables**: model.py, train.py, test.py, preprocess.py, README.md, requirements.txt
- **Scoring**: 80% accuracy + 20% model/data understanding

## Documentation

This feature directory contains comprehensive planning documentation:

### Core Documents

1. **[spec.md](./spec.md)** - Feature specification
   - Competition requirements and constraints
   - Functional and non-functional requirements
   - Success criteria

2. **[plan.md](./plan.md)** - Implementation plan
   - Technical context and decisions
   - Project structure
   - Phase breakdown

3. **[research.md](./research.md)** - Research findings (Phase 0)
   - Dataset analysis decisions
   - Architecture choices (ResNet18 + GRU baseline)
   - Loss functions (Smooth L1 → GIoU upgrade path)
   - Training strategy (3-phase approach)
   - Risk mitigation

4. **[data-model.md](./data-model.md)** - Data structures (Phase 1)
   - JSON schema for annotations
   - Internal data structures (Vocab, Sample, Batch)
   - Checkpoint format
   - Output CSV schema

5. **[quickstart.md](./quickstart.md)** - Getting started guide (Phase 1)
   - Installation instructions
   - Data setup
   - Training commands
   - Troubleshooting

### API Contracts (Phase 1)

Located in `contracts/`:

- **[model_interface.py](./contracts/model_interface.py)** - Model component interfaces
- **[data_interface.py](./contracts/data_interface.py)** - Data loading interfaces
- **[training_interface.py](./contracts/training_interface.py)** - Training loop interfaces
- **[evaluation_interface.py](./contracts/evaluation_interface.py)** - Metrics and evaluation interfaces

## Key Decisions

### Architecture (from research.md)

**Baseline (Phase 1)**:
- Text Encoder: GRU with learned embeddings
- Image Encoder: ResNet18 (ImageNet pretrained)
- Fusion: Single-head cross-attention
- Bbox Head: MLP with sigmoid output

**Upgrade Path (Phase 2+)**:
- CLIP text encoder (allowed by rules)
- EfficientNet-B3 or ConvNeXt backbone
- Multi-head cross-attention
- GIoU loss

### Data

- **Image Size**: 512×512 baseline, 768×768 option
- **Batch Size**: 8 (adjust based on GPU)
- **Augmentation**: ColorJitter + mild rotation
- **Vocabulary**: Custom word-level tokenization for Korean

### Training

- **Optimizer**: AdamW (lr=1e-4, wd=1e-4)
- **Scheduler**: Cosine annealing
- **Loss**: Smooth L1 baseline, GIoU upgrade
- **Epochs**: 10-20 with early stopping

## Next Steps

1. **Run `/speckit.tasks`** to generate tasks.md
2. **Implement baseline** following contracts/
3. **Train and validate** on provided data
4. **Iterate** based on validation performance
5. **Submit** to leaderboard

## Project Structure

```
specs/001-query-based-visual-element-localization/
├── README.md              # This file
├── spec.md                # Feature specification
├── plan.md                # Implementation plan
├── research.md            # Phase 0 research
├── data-model.md          # Phase 1 data model
├── quickstart.md          # Phase 1 quickstart
└── contracts/             # Phase 1 API contracts
    ├── model_interface.py
    ├── data_interface.py
    ├── training_interface.py
    └── evaluation_interface.py
```

## Timeline Estimate

- **Phase 0 (Research)**: ✅ Complete
- **Phase 1 (Design)**: ✅ Complete
- **Phase 2 (Tasks)**: Next step - run `/speckit.tasks`
- **Implementation**: 1-2 weeks
  - Week 1: Baseline implementation and training
  - Week 2: Improvements and submission

## Success Criteria

- ✅ Planning documentation complete
- ⏳ Baseline achieves mIoU > 0.4
- ⏳ Final model achieves competitive score
- ⏳ Code passes reproducibility check
- ⏳ All deliverables submitted

## Notes

- Constitution check passed (no violations)
- All research findings documented
- Design contracts defined
- Ready for task generation and implementation
