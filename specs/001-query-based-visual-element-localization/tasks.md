# Tasks: Query-Based Visual Element Localization

**Input**: Design documents from `/specs/001-query-based-visual-element-localization/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Unit and integration tests included following TDD principles per constitution
**Organization**: Tasks grouped by functional requirement (mapped as user stories) for independent implementation

## Format: `- [ ] [ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story/functional requirement (FR1-FR5)
- Include exact file paths in descriptions

## Path Conventions

Single project structure (from plan.md):
- Source: `src/` at repository root
- Tests: `tests/` at repository root
- Deliverables: `model.py`, `train.py`, `test.py`, `preprocess.py` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project directory structure per plan.md (src/data, src/models, src/training, src/utils, tests/unit, tests/integration, outputs/ckpt, outputs/preds)
- [X] T002 [P] Initialize requirements.txt with PyTorch, torchvision, PIL, numpy, pandas, pytest, tqdm
- [X] T003 [P] Create .gitignore for Python project (data/, outputs/, __pycache__, *.pyc, .pytest_cache, *.pth)
- [X] T004 [P] Implement seed_everything() function in src/utils/seed.py for reproducibility
- [X] T005 [P] Implement CFG configuration class in src/utils/config.py with hyperparameters from research.md
- [X] T006 [P] Create README.md template with competition requirements sections

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY functional requirement can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T007 Implement JSON reading utility in src/utils/io.py (read_json, find_jsons functions)
- [X] T008 [P] Implement image path resolution in src/utils/io.py (get_image_path function per data-model.md)
- [X] T009 [P] Create unit test for JSON utilities in tests/unit/test_io.py
- [X] T010 [P] Create simple_tokenize function in src/data/vocab.py for Korean text processing
- [X] T011 Implement Vocab class in src/data/vocab.py (build, encode, decode methods per data-model.md)
- [X] T012 [P] Create unit test for Vocab class in tests/unit/test_vocab.py
- [X] T013 [P] Implement is_visual_ann validation function in src/utils/io.py
- [X] T014 [P] Implement bbox normalization/denormalization in src/utils/io.py (normalize_bbox, denormalize_bbox)

**Checkpoint**: Foundation ready - functional requirement implementation can now begin in parallel

---

## Phase 3: FR1 - Data Loading (Priority: P1) 🎯 MVP Core

**Goal**: Load and parse document images with JSON annotations for training

**Independent Test**: Successfully load sample from train set, verify image tensor shape, query encoding, bbox normalization

### Tests for FR1 (TDD)

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T015 [P] [FR1] Unit test for is_visual_ann filter in tests/unit/test_io.py
- [X] T016 [P] [FR1] Unit test for bbox normalization in tests/unit/test_io.py
- [X] T017 [P] [FR1] Unit test for UniDSet.__getitem__ in tests/unit/test_dataset.py
- [X] T018 [FR1] Integration test for data loading pipeline in tests/integration/test_data_pipeline.py

### Implementation for FR1

- [X] T019 [P] [FR1] Implement UniDSet.__init__ in src/data/dataset.py (load JSON files, filter visual annotations)
- [X] T020 [P] [FR1] Implement UniDSet.__len__ in src/data/dataset.py
- [X] T021 [FR1] Implement UniDSet.__getitem__ in src/data/dataset.py (load image, encode query, normalize bbox per data-model.md Sample structure)
- [X] T022 [FR1] Implement collate_fn in src/data/dataset.py (pad queries, stack images, collect metadata per data-model.md)
- [X] T023 [FR1] Implement make_loader helper in src/data/dataset.py (create DataLoader with proper settings)
- [X] T024 [FR1] Add error handling for missing images and corrupted JSON files in src/data/dataset.py
- [X] T025 [FR1] Add logging for dataset statistics (total samples, vocab size) in src/data/dataset.py

**Checkpoint**: At this point, data loading should be fully functional - can iterate through batches

---

## Phase 4: FR2 - Model Architecture (Priority: P1) 🎯 MVP Core

**Goal**: Implement baseline vision-language model (ResNet18 + GRU + Cross-Attention)

**Independent Test**: Forward pass with dummy batch produces correct output shape (B, 4)

### Tests for FR2 (TDD)

- [X] T026 [P] [FR2] Unit test for TextEncoder forward pass in tests/unit/test_models.py
- [X] T027 [P] [FR2] Unit test for ImageEncoder forward pass in tests/unit/test_models.py
- [X] T028 [P] [FR2] Unit test for CrossAttentionBBox forward pass in tests/unit/test_models.py
- [X] T029 [FR2] Integration test for end-to-end model forward in tests/unit/test_models.py

### Implementation for FR2

- [X] T030 [P] [FR2] Implement TextEncoder (GRU-based) in src/models/text_encoder.py per contracts/model_interface.py
- [X] T031 [P] [FR2] Implement TinyCNN fallback in src/models/image_encoder.py
- [X] T032 [FR2] Implement ImageEncoder (ResNet18 backbone + projection) in src/models/image_encoder.py per contracts/model_interface.py
- [X] T033 [FR2] Implement CrossAttentionBBox fusion module in src/models/fusion.py per contracts/model_interface.py
- [X] T034 [FR2] Implement CrossAttnVLM main model in src/models/vlm.py (compose text, image, fusion)
- [X] T035 [FR2] Create model.py at repository root (import CrossAttnVLM, create_model factory function)
- [X] T036 [FR2] Add model summary logging (parameter count, architecture) in src/models/vlm.py

**Checkpoint**: Model architecture complete - can run forward pass and get bbox predictions

---

## Phase 5: FR3 - Training Pipeline (Priority: P1) 🎯 MVP Core

**Goal**: Train baseline model on training data with validation monitoring

**Independent Test**: Run training for 2 epochs, verify loss decreases, checkpoint saved

### Tests for FR3 (TDD)

- [ ] T037 [P] [FR3] Unit test for smooth_l1_loss in tests/unit/test_loss.py
- [ ] T038 [P] [FR3] Unit test for IoU computation in tests/unit/test_metrics.py
- [ ] T039 [FR3] Integration test for single training epoch in tests/integration/test_training.py
- [ ] T040 [FR3] Integration test for checkpoint save/load in tests/integration/test_training.py

### Implementation for FR3

- [ ] T041 [P] [FR3] Implement smooth_l1_loss in src/training/loss.py per contracts/training_interface.py
- [ ] T042 [P] [FR3] Implement iou_xywh_pixel in src/training/metrics.py per contracts/evaluation_interface.py
- [ ] T043 [P] [FR3] Implement compute_miou in src/training/metrics.py
- [ ] T044 [FR3] Implement train_epoch function in src/training/trainer.py (forward, loss, backward, optimizer step)
- [ ] T045 [FR3] Implement validate_epoch function in src/training/trainer.py (forward, compute mIoU)
- [ ] T046 [FR3] Implement train_loop in src/training/trainer.py (epoch loop, scheduler, early stopping, checkpointing)
- [ ] T047 [FR3] Implement save_checkpoint in src/training/trainer.py per data-model.md checkpoint format
- [ ] T048 [FR3] Implement load_checkpoint in src/training/trainer.py
- [ ] T049 [FR3] Create train.py at repository root (CLI with argparse, call train_loop)
- [ ] T050 [FR3] Add mixed precision training support (torch.amp.autocast) in src/training/trainer.py
- [ ] T051 [FR3] Add training progress logging (loss, mIoU, lr per epoch) in src/training/trainer.py
- [ ] T052 [FR3] Add vocabulary building from training data in train.py

**Checkpoint**: Training pipeline complete - can train model and save checkpoints

---

## Phase 6: FR4 - Inference Pipeline (Priority: P1) 🎯 MVP Core

**Goal**: Load trained model and generate predictions on test data

**Independent Test**: Load checkpoint, run inference on sample test data, produce valid CSV

### Tests for FR4 (TDD)

- [ ] T053 [P] [FR4] Unit test for model loading in tests/unit/test_inference.py
- [ ] T054 [P] [FR4] Unit test for bbox denormalization in tests/unit/test_io.py
- [ ] T055 [FR4] Integration test for inference pipeline in tests/integration/test_inference.py

### Implementation for FR4

- [ ] T056 [P] [FR4] Implement _load_model_from_ckpt in src/training/trainer.py (restore model + vocab from checkpoint)
- [ ] T057 [FR4] Implement predict_on_dataset in src/training/trainer.py (inference loop, denormalize bboxes)
- [ ] T058 [FR4] Implement save_predictions_csv in src/training/metrics.py per data-model.md CSV format
- [ ] T059 [FR4] Create test.py at repository root (CLI for inference, call predict_on_dataset)
- [ ] T060 [FR4] Add CSV validation (required columns, no NaN) in test.py
- [ ] T061 [FR4] Add batch processing with progress bar in src/training/trainer.py

**Checkpoint**: Inference complete - can generate submission CSV from trained model

---

## Phase 7: FR5 - Evaluation (Priority: P2)

**Goal**: Comprehensive evaluation metrics for validation set

**Independent Test**: Run evaluation on validation set, get per-class and per-size mIoU breakdown

### Tests for FR5 (TDD)

- [ ] T062 [P] [FR5] Unit test for precision_at_threshold in tests/unit/test_metrics.py
- [ ] T063 [P] [FR5] Unit test for categorize_bbox_size in tests/unit/test_metrics.py
- [ ] T064 [FR5] Integration test for full evaluation in tests/integration/test_evaluation.py

### Implementation for FR5

- [ ] T065 [P] [FR5] Implement compute_precision_at_threshold in src/training/metrics.py
- [ ] T066 [P] [FR5] Implement categorize_bbox_size in src/training/metrics.py (small/medium/large)
- [ ] T067 [P] [FR5] Implement compute_center_error in src/training/metrics.py
- [ ] T068 [P] [FR5] Implement compute_size_error in src/training/metrics.py
- [ ] T069 [FR5] Implement evaluate_predictions in src/training/metrics.py (comprehensive metrics per contracts/evaluation_interface.py)
- [ ] T070 [FR5] Add evaluation CLI mode to train.py (--eval flag, call evaluate_predictions)
- [ ] T071 [FR5] Add per-class IoU logging (tables vs charts) in src/training/metrics.py
- [ ] T072 [FR5] Add worst predictions analysis in src/training/metrics.py

**Checkpoint**: Full evaluation suite available for model analysis

---

## Phase 8: Data Preprocessing & Utilities (Priority: P2)

**Goal**: Data exploration and preprocessing utilities

**Independent Test**: Run preprocessing script to get dataset statistics

### Implementation for Data Preprocessing

- [ ] T073 [P] [P8] Implement data verification in preprocess.py (check JSON-image pairs, bbox validity)
- [ ] T074 [P] [P8] Implement dataset statistics in preprocess.py (sample counts, query lengths, bbox sizes)
- [ ] T075 [P] [P8] Implement submission validation in preprocess.py (validate_submission_format per contracts/evaluation_interface.py)
- [ ] T076 [P] [P8] Implement analysis mode in preprocess.py (per-class stats, size distribution)
- [ ] T077 [P8] Add CLI interface to preprocess.py (--verify, --stats, --validate-submission, --analyze)

**Checkpoint**: Data utilities available for debugging and analysis

---

## Phase 9: Advanced Features (Priority: P3) - Optional Improvements

**Goal**: Improvements beyond baseline for better accuracy

**Independent Test**: Train with advanced features, verify mIoU improvement

### Implementation for Advanced Features

- [ ] T078 [P] [P9] Implement GIoU loss in src/training/loss.py per research.md upgrade path
- [ ] T079 [P] [P9] Implement combined loss (Smooth L1 + GIoU) in src/training/loss.py
- [ ] T080 [P] [P9] Implement ColorJitter augmentation in src/data/dataset.py per research.md
- [ ] T081 [P] [P9] Implement rotation augmentation with bbox adjustment in src/data/augmentation.py
- [ ] T082 [P9] Add multi-head cross-attention option in src/models/fusion.py
- [ ] T083 [P9] Add EfficientNet backbone option in src/models/image_encoder.py
- [ ] T084 [P9] Add CLIP text encoder option in src/models/text_encoder.py (if baseline insufficient)

**Checkpoint**: Advanced features available for experimentation

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Finalization for competition submission

- [ ] T085 [P] Complete README.md with setup instructions, training commands, pretrained model sources per quickstart.md
- [ ] T086 [P] Verify all code follows PEP8 style guidelines
- [ ] T087 [P] Add docstrings to all public functions and classes
- [ ] T088 [P] Test reproducibility (run training twice with same seed, verify identical checkpoints)
- [ ] T089 [P] Verify no forbidden dependencies (no LLaVA, no external data)
- [ ] T090 [P] Create submission.zip generation script in train.py (zip command)
- [ ] T091 Run full quickstart.md validation (install, train 2 epochs, inference, verify CSV)
- [ ] T092 Clean up debug code and temporary files
- [ ] T093 Final submission checklist: model.py, train.py, test.py, preprocess.py, README.md, requirements.txt all present

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all functional requirements
- **FR1-FR5 (Phases 3-7)**: All depend on Foundational phase completion
  - FR1 (Data Loading): Can start after Foundational
  - FR2 (Model): Can start after Foundational (parallel with FR1)
  - FR3 (Training): Depends on FR1 + FR2 complete
  - FR4 (Inference): Depends on FR2 complete (can develop before FR3)
  - FR5 (Evaluation): Depends on FR1 + FR2 complete
- **Phase 8 (Preprocessing)**: Can start after Foundational (parallel with FR1-FR5)
- **Phase 9 (Advanced)**: Depends on FR1-FR5 baseline working
- **Polish (Phase 10)**: Depends on all desired features complete

### Critical Path (MVP)

**Minimum for working submission**:
1. Setup (Phase 1) →
2. Foundational (Phase 2) →
3. FR1 (Data) + FR2 (Model) [parallel] →
4. FR3 (Training) →
5. FR4 (Inference) →
6. Basic Polish (T085, T091, T093)

**Estimated**: ~50-60 tasks for MVP (T001-T061 + key polish tasks)

### Functional Requirement Dependencies

- **FR1 (Data Loading)**: No dependencies on other FRs - independently testable
- **FR2 (Model)**: No dependencies on other FRs - can test with dummy data
- **FR3 (Training)**: Depends on FR1 + FR2 - needs data and model
- **FR4 (Inference)**: Depends on FR2 - can test with dummy checkpoint
- **FR5 (Evaluation)**: Depends on FR1 + FR2 - needs data and predictions

### Within Each Functional Requirement

- Tests MUST be written and FAIL before implementation
- Utility functions before main components
- Core implementation before extensions
- Error handling and logging after core functionality

### Parallel Opportunities

**Phase 1 (Setup)**: Tasks T002-T006 can all run in parallel

**Phase 2 (Foundational)**: Tasks T008-T014 can run in parallel after T007

**Phase 3 (FR1)**:
- Tests T015-T017 can run in parallel
- Implementation T019-T020 can run in parallel

**Phase 4 (FR2)**:
- Tests T026-T028 can run in parallel
- Implementation T030-T031 can run in parallel

**Phase 5 (FR3)**:
- Tests T037-T038 can run in parallel
- Implementation T041-T043 can run in parallel

**Phase 6 (FR4)**:
- Tests T053-T054 can run in parallel
- Implementation T056-T058 can run in parallel

**Phase 7 (FR5)**:
- Tests T062-T063 can run in parallel
- Implementation T065-T068 can run in parallel

**Phase 8**: All tasks T073-T076 can run in parallel

**Phase 9**: All tasks T078-T081 can run in parallel

**Phase 10**: All tasks T085-T090 can run in parallel

---

## Parallel Example: FR2 (Model Architecture)

```bash
# Launch all tests for FR2 together:
Task: "Unit test for TextEncoder forward pass in tests/unit/test_models.py"
Task: "Unit test for ImageEncoder forward pass in tests/unit/test_models.py"
Task: "Unit test for CrossAttentionBBox forward pass in tests/unit/test_models.py"

# Launch model components in parallel:
Task: "Implement TextEncoder in src/models/text_encoder.py"
Task: "Implement TinyCNN fallback in src/models/image_encoder.py"
```

---

## Implementation Strategy

### MVP First (FR1-FR4 Only)

1. Complete Phase 1: Setup → ~30 min
2. Complete Phase 2: Foundational → ~2 hours
3. Complete Phase 3: FR1 (Data Loading) → ~4 hours
4. Complete Phase 4: FR2 (Model) → ~4 hours
5. Complete Phase 5: FR3 (Training) → ~6 hours
6. Complete Phase 6: FR4 (Inference) → ~2 hours
7. **STOP and VALIDATE**: Train for 10 epochs, generate submission
8. Submit to leaderboard for baseline score

**Estimated MVP Time**: ~20-24 hours of focused development

### Incremental Delivery

1. **Day 1-2**: Setup + Foundational + FR1 + FR2 → Can test model forward pass
2. **Day 3-4**: FR3 → Can train baseline model
3. **Day 5**: FR4 + submission → First leaderboard submission
4. **Day 6-7**: FR5 + analysis → Understand model weaknesses
5. **Day 8-10**: Phase 9 (Advanced features) → Improve mIoU
6. **Day 11-12**: Phase 10 (Polish) → Final submission

### Parallel Team Strategy

With 2 developers:

1. Both complete Setup + Foundational together
2. Once Foundational done:
   - **Developer A**: FR1 (Data) + FR3 (Training)
   - **Developer B**: FR2 (Model) + FR4 (Inference)
3. Integrate for full training pipeline
4. Both work on Phase 9 improvements
5. Both polish for submission

---

## Task Breakdown Summary

| Phase | Task Range | Count | Purpose | Estimated Time |
|-------|------------|-------|---------|----------------|
| 1: Setup | T001-T006 | 6 | Project init | 30 min |
| 2: Foundational | T007-T014 | 8 | Core utilities | 2 hours |
| 3: FR1 (Data) | T015-T025 | 11 | Data loading | 4 hours |
| 4: FR2 (Model) | T026-T036 | 11 | Model architecture | 4 hours |
| 5: FR3 (Training) | T037-T052 | 16 | Training pipeline | 6 hours |
| 6: FR4 (Inference) | T053-T061 | 9 | Inference & CSV | 2 hours |
| 7: FR5 (Evaluation) | T062-T072 | 11 | Metrics suite | 3 hours |
| 8: Preprocessing | T073-T077 | 5 | Data utilities | 2 hours |
| 9: Advanced | T078-T084 | 7 | Improvements | 4 hours |
| 10: Polish | T085-T093 | 9 | Finalization | 3 hours |
| **TOTAL** | T001-T093 | **93** | **Full project** | **~30 hours** |

### MVP Scope (First Submission)

**Tasks**: T001-T061 + T085 + T091 + T093 = **64 tasks**
**Estimated**: ~20-24 hours
**Output**: Baseline model, first leaderboard submission

### Parallel Execution Potential

- **Setup**: 5/6 tasks parallel = 1.2x speedup
- **Foundational**: 7/8 tasks parallel = 1.9x speedup
- **FR1-FR5**: ~40% of tasks marked [P] = 1.4x average speedup
- **Overall**: ~1.5x speedup with 2 developers, ~2x with 3-4 developers

---

## Notes

- **[P] marker**: Tasks working on different files, can run in parallel
- **[Story] label**: Maps task to functional requirement (FR1-FR5, P8-P10)
- **TDD approach**: All tests written before implementation per constitution
- **Competition compliance**: No forbidden dependencies, reproducible, documented
- **Checkpoints**: Each phase has validation criteria for independent testing
- **Incremental value**: Each FR complete = testable functionality
- **MVP focus**: FR1-FR4 = minimum for leaderboard submission
- **Avoid**: Same file conflicts, cross-FR dependencies that break independence
- Commit after each task or logical group
- Stop at any checkpoint to validate independently
