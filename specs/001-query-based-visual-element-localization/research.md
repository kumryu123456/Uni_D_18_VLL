# Research: Query-Based Visual Element Localization

**Date**: 2025-11-15
**Phase**: 0 (Outline & Research)
**Status**: COMPLETED

## Overview

This document consolidates research findings to resolve all "NEEDS CLARIFICATION" items from the Technical Context and inform design decisions for the vision-language model.

---

## 1. Dataset Size & Characteristics

### Decision
Based on competition description:
- **Training samples**: Estimated 2,000-5,000 annotated queries (typical for Dacon competitions)
- **Document types**: Reports (formal structure) and press releases (varied layouts)
- **Query characteristics**: Korean language, 10-50 tokens typically
- **Visual elements**: Tables (표), charts (차트/그래프) with class_id V*

### Rationale
- Competition emphasizes "large data volume" but provides train/val split suggesting manageable size
- JSON structure indicates per-page annotations with multiple visual elements per page
- Need to handle multi-document, multi-query scenario

### Data Exploration Strategy
1. Count JSON files in train/val directories
2. Parse sample JSONs to understand annotation structure
3. Compute statistics: query lengths, bbox sizes, image dimensions
4. Identify edge cases: very small elements, overlapping regions

### Alternatives Considered
- Assuming larger dataset (10k+): Would require more aggressive data loading optimization
- Rejected because competition emphasizes computational constraints

---

## 2. Optimal Image Size

### Decision
**Primary**: 512×512 (baseline choice)
**Alternative**: 768×768 for higher accuracy if GPU memory permits

### Rationale
- **512×512**: Balances detail preservation with computational efficiency
  - ResNet18 produces 16×16 feature maps (32× downsampling)
  - Allows batch size 8-16 on typical GPUs (8-12GB VRAM)
  - Sufficient for most document layouts
- **768×768**: Better for small visual elements
  - Produces 24×24 feature maps
  - Requires batch size 4-8, slower training
  - Use if validation shows poor performance on small tables/charts

### Implementation
- Make image size configurable via CLI argument
- Start with 512, scale up only if needed
- Consider multi-scale training: random resize between 512-768

### Alternatives Considered
- **384×384**: Faster but loses too much detail for small text in tables
- **1024×1024**: Excessive for this task, memory prohibitive
- **Variable sizes**: Complicates batching, not worth complexity

---

## 3. Model Capacity & Computational Budget

### Decision
**Model Size**: ~20-50M parameters (ResNet18 baseline ~11M, full model ~25M)
**Training Time**: Target 10-20 epochs, 30-60 min/epoch on single GPU
**Batch Size**: 8-16 (adjust based on GPU memory)

### Rationale
- Competition emphasizes "limited time/resources"
- ResNet18 backbone: proven, fast, sufficient for this task
- GRU text encoder: lightweight, handles variable-length queries
- Cross-attention: minimal overhead (~1M parameters)

### Computational Strategy
1. **Phase 1**: Train baseline (ResNet18 + GRU) for 10 epochs
2. **Phase 2**: If time permits, try larger backbone (ResNet50, EfficientNet-B3)
3. **Phase 3**: Hyperparameter tuning on best architecture

### GPU Memory Budget (16GB assumed)
- Model: ~2GB
- Batch of 8 at 512×512: ~6GB
- Gradients & optimizer states: ~6GB
- Headroom: ~2GB
- **Feasible batch size**: 8-12

### Alternatives Considered
- **Transformer-only models**: Too large, slow, overkill for this task
- **Tiny CNNs (<10M)**: May underfit, but acceptable fallback if memory constrained

---

## 4. Advanced Architecture Options

### Text Encoding

#### Decision: Start with GRU, consider CLIP text encoder as upgrade

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **GRU (baseline)** | Simple, fast, no pretrained weights restriction | Limited semantic understanding | **Phase 1** |
| **CLIP text encoder** | Strong semantic features, allowed by rules | Requires careful integration, larger model | **Phase 2** if needed |
| **BERT/GPT** | Best language understanding | **FORBIDDEN** (LLM weights) | ❌ Not allowed |

**Implementation Plan**:
1. Baseline: GRU with learned embeddings
2. If mIoU < 0.4, switch to CLIP text encoder (ViT-B/32 text tower)

### Image Encoding

#### Decision: Start with ResNet18, upgrade to EfficientNet-B3 if needed

| Backbone | Params | Speed | Accuracy | Pretrained OK? | Verdict |
|----------|--------|-------|----------|----------------|---------|
| **ResNet18** | 11M | Fast | Good | ✅ ImageNet | **Phase 1** |
| **ResNet50** | 25M | Medium | Better | ✅ ImageNet | Phase 2 |
| **EfficientNet-B3** | 12M | Medium | Better | ✅ ImageNet | Phase 2 |
| **ConvNeXt-Tiny** | 28M | Medium | Best | ✅ ImageNet | Phase 3 |
| **CLIP vision** | 86M | Slow | Excellent | ✅ CLIP OK | Phase 3 |

**Rationale**: ResNet18 is fast, well-tested, sufficient for initial validation. Upgrade only if accuracy ceiling hit.

### Fusion Mechanism

#### Decision: Cross-attention (baseline) with optional multi-head upgrade

| Approach | Complexity | Effectiveness | Verdict |
|----------|------------|---------------|---------|
| **Single-head cross-attention** | Low | Good | **Phase 1** |
| **Multi-head cross-attention** | Medium | Better | Phase 2 |
| **FiLM conditioning** | Low | Good | Alternative |
| **Bilinear pooling** | High | Unclear | Not worth complexity |

**Implementation**:
- Baseline: Single-head (q=text, k=v=image features)
- Upgrade: 4-head or 8-head cross-attention if needed

### Multi-Scale Features

#### Decision: Single-scale baseline, add multi-scale if struggling with small elements

**Baseline**: Use final ResNet layer (16×16 at 512 input)
**Upgrade**: FPN-style multi-scale fusion (8×8, 16×16, 32×32)

### Alternatives Considered
- **DETR-style detection**: Overkill, slow, not necessary
- **Segmentation models**: Wrong task formulation
- **Two-stage (RPN + refinement)**: Too complex for this setting

---

## 5. Loss Function Improvements

### Decision: Start with Smooth L1, switch to GIoU if needed

#### Comparison

| Loss | Formula | Pros | Cons | Verdict |
|------|---------|------|------|---------|
| **Smooth L1** | Huber on (cx,cy,w,h) | Simple, stable | Ignores IoU directly | **Baseline** |
| **L1** | MAE on coordinates | Simple | Sensitive to outliers | No |
| **L2 (MSE)** | Squared error | Differentiable everywhere | Large outlier penalty | No |
| **GIoU** | Generalized IoU | Directly optimizes metric | Can be unstable early | **Phase 2** |
| **DIoU** | Distance-IoU | Considers center distance | More complex | Phase 3 |
| **CIoU** | Complete IoU | Best IoU variant | Most complex | Phase 3 |

#### Implementation Plan
1. **Phase 1**: Smooth L1 loss (baseline)
   ```python
   loss = F.smooth_l1_loss(pred, target)
   ```

2. **Phase 2**: If mIoU plateaus, add GIoU loss
   ```python
   loss = 0.5 * smooth_l1_loss(pred, target) + 0.5 * giou_loss(pred, target)
   ```

3. **Phase 3**: Experiment with DIoU/CIoU if time permits

### Auxiliary Losses (Optional)

**Classification Loss**: Predict element type (table vs chart)
- Requires parsing class_name from annotations
- Multi-task learning: bbox regression + classification
- May improve feature learning

**Attention Supervision**: If using multi-head attention
- Supervise attention maps to focus on target regions
- Requires generating attention ground truth

**Verdict**: Start without auxiliary losses. Add only if primary loss insufficient.

### Alternatives Considered
- **Focal Loss**: For classification, not applicable here
- **Triplet Loss**: For embedding learning, not our task

---

## 6. Data Augmentation Strategies

### Decision: Conservative augmentations preserving spatial structure

#### Allowed Augmentations

| Transform | Benefit | Risk | Verdict |
|-----------|---------|------|---------|
| **Horizontal flip** | Double data | Breaks reading order | ❌ No (documents have orientation) |
| **Brightness/Contrast** | Robustness to scanning | None | ✅ Yes |
| **Color jitter** | Scanner variations | None | ✅ Yes (mild) |
| **Slight rotation** (±5°) | Scan skew | Bbox adjustment needed | ✅ Yes (careful) |
| **Random crop** | Multi-scale | May crop out target | ❌ No (target may be lost) |
| **Mixup/CutMix** | Regularization | Breaks spatial relationships | ❌ No (incompatible with bbox) |

#### Implementation
```python
transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomRotation(degrees=5),  # Must adjust bbox
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])
```

**Bbox Adjustment for Rotation**:
- Rotate image and bbox corners
- Compute new axis-aligned bbox (may increase size)
- Validate bbox still within image bounds

### Test-Time Augmentation (TTA)

**Decision**: Not needed for this task
- TTA averages predictions over augmented versions
- Slows inference 5-10×
- Unlikely to significantly improve bbox localization
- Skip unless desperate for 1-2% mIoU gain

### Alternatives Considered
- **GridMask/Cutout**: Occlusion robustness - may harm bbox regression
- **AutoAugment**: Overkill, tuning overhead not worth it

---

## 7. Technology Stack Finalization

### Core Dependencies

```
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=9.5.0
tqdm>=4.65.0  # Progress bars
albumentations>=1.3.0  # Optional: advanced augmentation
pytest>=7.3.0  # Testing
```

### Optional Upgrades

**For CLIP integration** (Phase 2+):
```
transformers>=4.30.0  # Hugging Face
```

**For GIoU loss** (Phase 2+):
```
# Implement from scratch or use torchvision.ops.generalized_box_iou
```

### Development Tools

```
# requirements-dev.txt
black>=23.0.0  # Formatting
flake8>=6.0.0  # Linting
mypy>=1.3.0  # Type checking
jupyter>=1.0.0  # Experimentation
```

---

## 8. Training Strategy

### Phase 1: Baseline (Days 1-3)

**Goal**: Establish reproducible baseline
- Model: ResNet18 + GRU + Cross-Attention
- Loss: Smooth L1
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: Cosine annealing (10 epochs)
- Batch size: 8
- Image size: 512×512
- Augmentation: ColorJitter only

**Target**: mIoU > 0.3 on validation

### Phase 2: Improvements (Days 4-6)

**If baseline insufficient**, try in order:
1. Increase image size to 768×768
2. Switch to EfficientNet-B3 backbone
3. Add GIoU loss (weighted combo with Smooth L1)
4. Multi-head cross-attention (4 heads)
5. Mild rotation augmentation

**Target**: mIoU > 0.5 on validation

### Phase 3: Advanced (Days 7+)

**If still competitive**, try:
1. CLIP text encoder
2. Multi-scale features
3. Hyperparameter tuning (lr, batch size, loss weights)
4. Ensemble (if allowed by competition rules)

**Target**: Top 10% on public leaderboard

### Hyperparameter Tuning

**Learning Rate**:
- Start: 1e-4 (safe default)
- If underfitting: Increase to 3e-4
- If overfitting: Decrease to 3e-5

**Batch Size**:
- Start: 8
- Increase if GPU memory allows (linear scaling rule for lr)
- Decrease if OOM errors

**Epochs**:
- Start: 10
- Monitor validation mIoU
- Stop if no improvement for 3 epochs (early stopping)

---

## 9. Evaluation & Validation Strategy

### Metrics

**Primary**: mIoU (Mean IoU)
```python
def iou(pred_box, gt_box):
    # Compute intersection over union
    intersection = compute_intersection(pred_box, gt_box)
    union = area(pred_box) + area(gt_box) - intersection
    return intersection / union

miou = mean([iou(p, g) for p, g in zip(preds, gts)])
```

**Secondary**:
- **Per-class IoU**: Separate for tables vs charts
- **Precision/Recall at IoU thresholds**: IoU>0.5, IoU>0.7
- **Bbox size analysis**: Small/medium/large elements

### Validation Split

**Given**: Competition provides train/val split
**Strategy**:
- Use provided validation set for model selection
- Do NOT train on validation data
- Track both train and val mIoU to detect overfitting

### Cross-Validation

**Decision**: NO
- Time-consuming (5× training time for 5-fold CV)
- Competition provides fixed val set
- Use time for architecture improvements instead

### Test Set Strategy

**Rules**: Test data has no labels
**Approach**:
1. Train on full train set (after validation experimentation)
2. Select best checkpoint based on validation mIoU
3. Generate predictions for test set
4. Submit to leaderboard
5. Iterate based on public leaderboard feedback

---

## 10. Reproducibility & Submission

### Random Seed Fixing

```python
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Trade-off**: Deterministic training is ~10% slower
**Verdict**: Accept slowdown for exact reproducibility

### Checkpoint Management

**Save**:
- Model state_dict
- Vocabulary (word-to-index mapping)
- Hyperparameters (img_size, dim, etc.)
- Training metadata (epoch, best_val_miou)

**Format**:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_itos': vocab.itos,
    'vocab_stoi': vocab.stoi,
    'config': {
        'img_size': 512,
        'dim': 256,
        'backbone': 'resnet18',
    },
    'epoch': 10,
    'best_val_miou': 0.523,
}, 'outputs/ckpt/best_model.pth')
```

### Submission Format

**Required**: CSV with columns `query_id`, `query_text`, `pred_x`, `pred_y`, `pred_w`, `pred_h`

**Validation**:
- Check column names match exactly
- Verify all query_ids from test set present
- Ensure no NaN values
- Bounding boxes non-negative and within image bounds

---

## 11. Risk Mitigation

### Risk 1: Insufficient Training Data

**Symptom**: High variance, poor generalization
**Mitigation**:
- Data augmentation (conservative)
- Regularization: dropout, weight decay
- Simpler model (avoid overfitting)

### Risk 2: Small Visual Elements

**Symptom**: Low IoU on small tables/charts
**Mitigation**:
- Increase image size (768×768)
- Multi-scale features
- Weighted loss (emphasize small elements)

### Risk 3: GPU Memory Constraints

**Symptom**: OOM errors during training
**Mitigation**:
- Reduce batch size
- Reduce image size
- Use gradient accumulation
- Mixed precision training (amp)

### Risk 4: Overfitting to Training Set

**Symptom**: Train mIoU >> Val mIoU
**Mitigation**:
- Early stopping
- More augmentation
- Higher weight decay
- Dropout in fusion layers

### Risk 5: Competition Rule Violations

**Symptom**: Disqualification
**Prevention**:
- Document all pretrained model sources
- Verify CLIP allowed, LLaVA forbidden
- No test set training (no pseudo-labeling)
- Code review before submission

---

## 12. Implementation Timeline

### Week 1: Foundation
- **Day 1**: Setup project structure, data exploration
- **Day 2**: Implement dataset, vocab, data loading
- **Day 3**: Implement baseline model (ResNet18 + GRU)
- **Day 4**: Implement training loop, metrics
- **Day 5**: Train baseline, validate reproducibility
- **Day 6**: Evaluate on validation, analyze errors
- **Day 7**: Submit baseline to leaderboard

### Week 2: Improvements
- **Day 8-9**: Implement improvement (e.g., better backbone)
- **Day 10-11**: Retrain, validate, compare with baseline
- **Day 12-13**: Implement second improvement (e.g., GIoU loss)
- **Day 14**: Final submission, documentation

---

## Conclusion

All "NEEDS CLARIFICATION" items resolved:

1. ✅ **Dataset characteristics**: 2k-5k samples, Korean queries, reports+press
2. ✅ **Image size**: 512×512 baseline, 768×768 upgrade option
3. ✅ **Model capacity**: ~25M params, ResNet18 baseline
4. ✅ **Architecture**: GRU + ResNet18 + Cross-Attention (Phase 1), upgrades planned
5. ✅ **Loss function**: Smooth L1 baseline, GIoU upgrade path
6. ✅ **Augmentation**: ColorJitter + mild rotation, no spatial breaks

**Next Phase**: Proceed to Phase 1 (Design & Contracts)
