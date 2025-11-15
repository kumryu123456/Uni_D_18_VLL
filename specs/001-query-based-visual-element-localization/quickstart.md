# Quickstart Guide: Query-Based Visual Element Localization

**Last Updated**: 2025-11-15
**Target Audience**: Competition participants, developers

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Setup](#data-setup)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Inference](#inference)
7. [Submission](#submission)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), Windows with WSL2, or macOS
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
  - CUDA 11.8+ or compatible
  - Without GPU: Training will be very slow
- **RAM**: 16GB+ recommended
- **Disk Space**: 30GB+ (dataset + outputs)

### Software Requirements

- **Python**: 3.8 or higher
- **Git**: For version control
- **CUDA Toolkit**: If using GPU

---

## Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd <project-directory>
```

### Step 2: Create Virtual Environment

**Using venv**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# Or on Windows WSL:
source venv/bin/activate
```

**Using conda** (alternative):
```bash
conda create -n vlm-competition python=3.9
conda activate vlm-competition
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt**:
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=9.5.0
tqdm>=4.65.0
pytest>=7.3.0
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch 2.0.1
CUDA available: True
```

---

## Data Setup

### Step 1: Download Dataset

Download from competition platform:
- `train_valid.zip` (training and validation data)
- `test.zip` (test data)

### Step 2: Extract Data

```bash
# Create data directory
mkdir -p data

# Extract training/validation data
unzip train_valid.zip -d data/
# Expected structure:
# data/
# ├── train/
# │   ├── report_jpg/
# │   ├── report_json/
# │   ├── press_jpg/
# │   └── press_json/
# └── val/
#     └── [same structure]

# Extract test data
unzip test.zip -d data/
# Expected structure:
# data/test/
# ├── images/
# └── query/
```

### Step 3: Verify Data Structure

```bash
python preprocess.py --verify
```

This will check:
- All JSON files have corresponding images
- Bounding boxes are valid
- Required fields are present

Expected output:
```
✓ Found 2,543 training samples
✓ Found 634 validation samples
✓ Found 1,200 test queries
✓ All images accessible
✓ All bounding boxes valid
```

---

## Training

### Quick Start (Baseline Model)

```bash
python train.py \
  --json_dir ./data/train/report_json \
  --jpg_dir ./data/train/report_jpg \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_ckpt ./outputs/ckpt/baseline.pth
```

**Parameters**:
- `--json_dir`: Path to JSON annotations
- `--jpg_dir`: Path to JPG images
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 8, reduce if OOM)
- `--lr`: Learning rate (default: 1e-4)
- `--img_size`: Input image size (default: 512)
- `--save_ckpt`: Output checkpoint path

### Training with Both Report and Press Data

```bash
# Train on reports
python train.py \
  --json_dir ./data/train/report_json \
  --jpg_dir ./data/train/report_jpg \
  --epochs 5 \
  --save_ckpt ./outputs/ckpt/report_model.pth

# Fine-tune on press releases
python train.py \
  --json_dir ./data/train/press_json \
  --jpg_dir ./data/train/press_jpg \
  --epochs 5 \
  --resume ./outputs/ckpt/report_model.pth \
  --save_ckpt ./outputs/ckpt/combined_model.pth
```

### Advanced Training Options

```bash
python train.py \
  --json_dir ./data/train/report_json \
  --jpg_dir ./data/train/report_jpg \
  --epochs 20 \
  --batch_size 12 \
  --lr 1e-4 \
  --img_size 768 \
  --dim 512 \
  --backbone efficientnet_b3 \
  --loss_type combined \
  --scheduler cosine \
  --augment \
  --save_ckpt ./outputs/ckpt/advanced.pth
```

**Advanced Parameters**:
- `--dim`: Embedding dimension (default: 256)
- `--backbone`: Image encoder (resnet18, resnet50, efficientnet_b3)
- `--loss_type`: Loss function (smooth_l1, giou, combined)
- `--scheduler`: LR scheduler (cosine, step, plateau)
- `--augment`: Enable data augmentation
- `--resume`: Resume from checkpoint

### Monitoring Training

Training will output progress every epoch:

```
[Epoch 1/10] loss=0.0523  val_miou=0.312  lr=0.000100
[Epoch 2/10] loss=0.0387  val_miou=0.398  lr=0.000095
[Epoch 3/10] loss=0.0301  val_miou=0.445  lr=0.000087
...
[Epoch 10/10] loss=0.0156  val_miou=0.532  lr=0.000010
✓ Best model saved: ./outputs/ckpt/baseline.pth (mIoU=0.532)
```

---

## Evaluation

### Evaluate on Validation Set

```bash
python train.py eval \
  --ckpt ./outputs/ckpt/baseline.pth \
  --json_dir ./data/val/report_json \
  --jpg_dir ./data/val/report_jpg \
  --out_csv ./outputs/preds/val_predictions.csv
```

**Output**:
```
[Eval] mIoU=0.532
       Precision@0.5: 0.723
       Precision@0.7: 0.512
       Mean center error: 23.4 pixels
       Mean size error: 0.156
✓ Predictions saved: ./outputs/preds/val_predictions.csv
```

### Analyze Validation Results

```bash
python preprocess.py --analyze \
  --predictions ./outputs/preds/val_predictions.csv \
  --ground_truth ./data/val/report_json
```

This will show:
- Per-class IoU (tables vs charts)
- Per-size IoU (small/medium/large)
- Worst predictions for debugging

---

## Inference

### Generate Test Predictions

```bash
python test.py \
  --ckpt ./outputs/ckpt/baseline.pth \
  --json_dir ./data/test/query \
  --jpg_dir ./data/test/images \
  --out_csv ./outputs/preds/submission.csv
```

**Output**:
```
Processing 1,200 queries...
100%|████████████████████| 150/150 [00:45<00:00,  3.31it/s]
✓ Predictions saved: ./outputs/preds/submission.csv
```

### Validate Submission Format

```bash
python preprocess.py --validate-submission \
  --csv ./outputs/preds/submission.csv \
  --test_queries ./data/test/query
```

**Checks**:
- All required columns present
- All test query IDs included
- No missing or invalid values
- UTF-8 encoding with BOM

---

## Submission

### Step 1: Review Predictions

```bash
head -n 5 ./outputs/preds/submission.csv
```

Expected format:
```csv
query_id,query_text,pred_x,pred_y,pred_w,pred_h
Q12345,"매출 추이 그래프는?",120.5,340.2,560.0,180.0
Q12346,"제품별 판매량 표는?",150.0,600.0,480.0,320.0
...
```

### Step 2: Create Submission ZIP

```bash
python train.py zip \
  --csv ./outputs/preds/submission.csv \
  --out_zip ./outputs/submission.zip
```

### Step 3: Submit to Platform

1. Go to competition submission page
2. Upload `outputs/submission.zip`
3. Wait for public score
4. Iterate based on feedback

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
python train.py --batch_size 4 ...

# Reduce image size
python train.py --img_size 384 ...

# Use gradient accumulation
python train.py --batch_size 4 --accumulation_steps 2 ...
```

#### 2. Image Not Found

**Error**: `FileNotFoundError: Could not resolve JPG for ...`

**Solutions**:
- Verify data extraction: `ls data/train/report_jpg/`
- Explicitly set jpg_dir: `--jpg_dir ./data/train/report_jpg`
- Check JSON source_data_name_jpg field matches actual filename

#### 3. Low Validation mIoU

**Problem**: mIoU < 0.3 after training

**Solutions**:
```bash
# Increase image size
python train.py --img_size 768 ...

# Try better backbone
python train.py --backbone resnet50 ...

# Train longer
python train.py --epochs 20 ...

# Switch to GIoU loss
python train.py --loss_type giou ...
```

#### 4. Slow Training

**Problem**: Training very slow (>10 min/epoch)

**Solutions**:
- Reduce num_workers: `--num_workers 0`
- Use smaller image size: `--img_size 384`
- Reduce batch size and enable mixed precision
- Check GPU utilization: `nvidia-smi`

#### 5. Validation Error: NaN Loss

**Problem**: Loss becomes NaN during training

**Solutions**:
- Lower learning rate: `--lr 1e-5`
- Use gradient clipping: `--grad_clip 1.0`
- Check data for corrupted images
- Verify bounding boxes in valid range

---

## Performance Benchmarks

### Expected Training Times (Single RTX 3090)

| Configuration | Epochs | Time/Epoch | Final mIoU |
|---------------|--------|------------|------------|
| Baseline (512×512, batch 8) | 10 | 3 min | 0.45-0.52 |
| Medium (768×768, batch 4) | 10 | 8 min | 0.52-0.58 |
| Advanced (768×768, EfficientNet) | 20 | 12 min | 0.55-0.62 |

### Memory Usage

| Configuration | GPU Memory | Disk Space |
|---------------|------------|------------|
| Baseline | 6 GB | 500 MB |
| Medium | 10 GB | 800 MB |
| Advanced | 14 GB | 1.2 GB |

---

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes
2. **Try advanced architectures**: EfficientNet, multi-scale features
3. **Ensemble models**: Average predictions from multiple checkpoints
4. **Error analysis**: Study validation failures to guide improvements
5. **Submit to leaderboard**: Iterate based on public score

---

## Additional Resources

- **Competition Rules**: [Link to competition page]
- **Baseline Code**: See provided baseline in competition description
- **Model Documentation**: See `contracts/` directory
- **Data Schema**: See `data-model.md`

---

## Support

For issues:
1. Check this guide and troubleshooting section
2. Review error messages carefully
3. Verify data setup and paths
4. Check competition forum for similar issues
5. Contact via competition platform

Good luck with the competition! 🚀
