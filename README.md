# Query-Based Visual Element Localization

Vision-language model for Dacon competition that predicts bounding box locations of visual elements (tables, charts) in document images based on natural language queries.

## Competition Details

- **Metric**: mIoU (Mean Intersection over Union)
- **Target**: Competitive accuracy on test set
- **Constraints**: No LLM/VLM pretrained weights, no external data, no APIs

## Development Environment

- **OS**: Linux (Ubuntu 20.04+ / WSL2)
- **Python**: 3.8+
- **GPU**: NVIDIA with CUDA 11.8+ (8GB+ VRAM recommended)

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows WSL: source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dataset Setup

1. Download competition data from Dacon platform
2. Extract to `data/` directory:

```
data/
├── train/
│   ├── report_jpg/
│   ├── report_json/
│   ├── press_jpg/
│   └── press_json/
├── val/
│   └── [same structure]
└── test/
    ├── images/
    └── query/
```

3. Verify data structure:

```bash
python preprocess.py --verify
```

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

### Training Options

- `--json_dir`: Path to JSON annotations
- `--jpg_dir`: Path to JPG images
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--img_size`: Input image size (default: 512)
- `--dim`: Embedding dimension (default: 256)
- `--save_ckpt`: Output checkpoint path

## Inference

### Generate Predictions

```bash
python test.py \
  --ckpt ./outputs/ckpt/baseline.pth \
  --json_dir ./data/test/query \
  --jpg_dir ./data/test/images \
  --out_csv ./outputs/preds/submission.csv
```

### Validate Submission Format

```bash
python preprocess.py --validate-submission \
  --csv ./outputs/preds/submission.csv \
  --test_queries ./data/test/query
```

## Evaluation

### Evaluate on Validation Set

```bash
python train.py eval \
  --ckpt ./outputs/ckpt/baseline.pth \
  --json_dir ./data/val/report_json \
  --jpg_dir ./data/val/report_jpg \
  --out_csv ./outputs/preds/val_predictions.csv
```

## Model Architecture

### Baseline (Phase 1)

- **Text Encoder**: GRU with learned embeddings (Korean tokenization)
- **Image Encoder**: ResNet18 (ImageNet pretrained)
- **Fusion**: Single-head cross-attention
- **Output**: Bounding box regression [cx, cy, w, h]

### Pretrained Models

- **ResNet18**: ImageNet weights from `torchvision.models`
  - Source: PyTorch official
  - License: BSD
  - Usage: Allowed per competition rules

## Project Structure

```
├── src/
│   ├── data/          # Dataset and data loading
│   ├── models/        # Model architecture
│   ├── training/      # Training loop and metrics
│   └── utils/         # Utilities (config, seed, I/O)
├── tests/
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
├── model.py           # Model definition (competition deliverable)
├── train.py           # Training script (competition deliverable)
├── test.py            # Inference script (competition deliverable)
├── preprocess.py      # Data preprocessing (competition deliverable)
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py --batch_size 4 ...

# Reduce image size
python train.py --img_size 384 ...
```

### Slow Training

```bash
# Reduce num_workers if data loading is bottleneck
python train.py --num_workers 0 ...
```

### Low Validation mIoU

- Try larger image size: `--img_size 768`
- Train longer: `--epochs 20`
- Use better backbone: `--backbone resnet50`

## Competition Submission

1. Train model
2. Generate predictions on test set
3. Validate CSV format
4. Create submission zip
5. Upload to Dacon platform

## License

This project is for Dacon competition purposes only.

## References

- Competition: Dacon Vision-Language Model Challenge
- Baseline code: Provided by competition organizers
