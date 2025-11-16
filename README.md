# 문서 내 시각요소 위치 예측 모델

질의기반 비전-언어 모델을 이용한 문서 내 표·차트 위치 예측

---

## 📋 개요

문서 이미지와 자연어 질의를 입력받아, 질의와 관련된 시각요소(표, 차트)의 위치를 예측하는 Vision-Language 모델입니다.

**평가 지표**: mIoU (Mean Intersection over Union)

---

## 🖥️ 개발 환경

- **OS**: Linux (Ubuntu 20.04+)
- **Python**: 3.8+
- **GPU**: CUDA 11.0+ (권장)

---

## 📦 사전 학습 모델

본 프로젝트는 다음 사전학습 모델을 사용합니다:

### ResNet50 (torchvision)
- **용도**: 이미지 특징 추출 백본
- **출처**: PyTorch 공식 torchvision
- **가중치**: ImageNet-1K pretrained weights
- **다운로드**: 자동 다운로드 (`torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`)
- **라이선스**: BSD 3-Clause License
- **참고**: https://pytorch.org/vision/stable/models.html

---

## 📁 파일 구조

```
.
├── model.py          # 모델 정의 (ResNet50 + BiGRU + Cross-Attention)
├── preprocess.py     # 데이터 전처리 및 로딩
├── train.py          # 학습 스크립트
├── test.py           # 추론 스크립트
├── requirements.txt  # 의존성 패키지
├── README.md         # 본 문서
└── data/             # 데이터 디렉토리
    ├── train/
    │   ├── press_json/
    │   ├── press_jpg/
    │   ├── report_json/
    │   └── report_jpg/
    └── valid/
        ├── press_json/
        ├── press_jpg/
        ├── report_json/
        └── report_jpg/
```

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt
```

### 2. 학습

**Press + Report 데이터 동시 학습** (권장):

```bash
python train.py \
  --train_json_dirs ./data/train/press_json ./data/train/report_json \
  --train_img_roots ./data/train/press_jpg ./data/train/report_jpg \
  --val_json_dirs ./data/valid/press_json ./data/valid/report_json \
  --val_img_roots ./data/valid/press_jpg ./data/valid/report_jpg \
  --epochs 50 \
  --batch_size 16 \
  --accumulation_steps 2 \
  --warmup_epochs 5 \
  --use_ema \
  --use_amp \
  --pretrained \
  --patience 15 \
  --save_dir ./checkpoints \
  --log_dir ./logs
```

**단일 디렉토리 학습** (호환성):

```bash
python train.py \
  --train_json_dir ./data/train/press_json \
  --train_img_root ./data/train/press_jpg \
  --val_json_dir ./data/valid/press_json \
  --val_img_root ./data/valid/press_jpg \
  --epochs 50 \
  --batch_size 16 \
  --use_ema \
  --use_amp \
  --pretrained
```

### 3. 추론

```bash
python test.py \
  --test_dir ./data/test \
  --checkpoint ./checkpoints/best_model.pt \
  --output_csv submission.csv \
  --enable_tta
```

**출력**: `submission.csv` 파일 생성
- 열 구성: `query_id`, `query_text`, `pred_x`, `pred_y`, `pred_w`, `pred_h`
- 좌표 형식: (x, y, w, h) - 좌상단 기준 픽셀 좌표

---

## 🏗️ 모델 아키텍처

### 주요 구성 요소

1. **이미지 인코더**: ResNet50 (Pretrained on ImageNet)
   - 문서 이미지 → 2D Feature Map

2. **텍스트 인코더**: Bidirectional GRU
   - 자연어 질의 → 텍스트 임베딩
   - Character-level tokenization (한국어/영어 지원)

3. **Cross-Attention**: Multi-Head Attention (8 heads)
   - 질의와 이미지 특징 융합

4. **BBox Regressor**: 2-layer MLP
   - 정규화된 BBox 좌표 예측 (cx, cy, w, h)

### 손실 함수

- **CIoU Loss**: Complete IoU Loss (weight=2.0)
- **L1 Loss**: Smooth L1 Loss (weight=1.0)
- **Combined Loss**: `2.0 * CIoU + 1.0 * L1`

### 학습 기법

- ✅ EMA (Exponential Moving Average, decay=0.9999)
- ✅ Cosine Annealing LR with Warmup (5 epochs)
- ✅ Gradient Clipping (max_norm=1.0)
- ✅ Gradient Accumulation (steps=2)
- ✅ Mixed Precision Training (AMP)
- ✅ Early Stopping (patience=15)

### 데이터 증강

- **Training**: ColorJitter, GaussianBlur, RandomRotation
- **Validation**: Resize + Normalize only

### Test Time Augmentation (TTA)

- Horizontal Flip
- Prediction Averaging

---

## 📊 예상 성능

| 구성 | mIoU | 특징 |
|------|------|------|
| 단일 모델 (Press) | 0.72-0.76 | EMA, CIoU Loss |
| Press + Report | 0.77-0.81 | 데이터 2배 |
| + TTA | 0.78-0.82 | 수평 뒤집기 |

---

## 💡 주요 특징

### 1. 멀티 소스 데이터 처리
- Press + Report 데이터 동시 학습
- 자동 데이터 통합 및 Vocabulary 생성

### 2. 안정적인 학습
- EMA로 모델 가중치 안정화
- Gradient Clipping으로 Exploding Gradient 방지
- Mixed Precision으로 메모리 효율성 향상

### 3. 강력한 손실 함수
- CIoU Loss로 BBox 위치, 크기, 비율 동시 최적화
- L1 Loss로 smooth regression

### 4. Character-level Tokenization
- 한국어 문자 단위 토크나이징
- OOV (Out-of-Vocabulary) 문제 해결

---

## 🔧 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--epochs` | 50 | 총 학습 에폭 |
| `--batch_size` | 16 | 배치 크기 |
| `--lr` | 1e-4 | 학습률 |
| `--warmup_epochs` | 5 | Warmup 에폭 수 |
| `--accumulation_steps` | 2 | Gradient Accumulation |
| `--ciou_weight` | 2.0 | CIoU Loss 가중치 |
| `--patience` | 15 | Early Stopping patience |
| `--img_size` | 512 | 입력 이미지 크기 |
| `--embed_dim` | 256 | 임베딩 차원 |
| `--num_heads` | 8 | Attention Head 수 |

---

## 📝 제출 형식

**CSV 파일 구조**:

```csv
query_id,query_text,pred_x,pred_y,pred_w,pred_h
MI2_240725_TY2_0001_1.jpg,감염병전문병원 추진 개요,512.34,345.67,234.12,156.89
```

- `query_id`: 이미지 파일명
- `query_text`: 질의 텍스트
- `pred_x`, `pred_y`: BBox 좌상단 좌표 (픽셀)
- `pred_w`, `pred_h`: BBox 너비/높이 (픽셀)

---

## ⚠️ 주의사항

1. **데이터 경로**: 실제 환경에 맞게 경로 수정 필요
2. **GPU 메모리**: batch_size 조정 (OOM 발생 시 줄이기)
3. **학습 시간**: 전체 데이터 50 epoch 학습 시 약 10-14시간 소요
4. **TTA**: 추론 시간 2배 증가하지만 성능 향상

---

## 📚 참고 자료

- **ResNet**: Deep Residual Learning for Image Recognition (He et al., 2015)
- **CIoU Loss**: Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation (Zheng et al., 2020)
- **EMA**: Mean teachers are better role models (Tarvainen & Valpola, 2017)

---

## 👥 개발자

Uni_D_18_VLL Team

---

## 📄 라이선스

본 프로젝트는 대회 제출용 코드입니다.
