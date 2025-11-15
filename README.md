# Query-Based Visual Element Localization

Vision-Language Model for predicting bounding box locations of visual elements (tables, charts) in document images based on natural language queries.

## 대회 정보
- **대회명**: 문서 내 시각요소(표·차트) 위치 예측을 위한 질의기반 비전-언어 모델 개발
- **평가지표**: mIoU (Mean Intersection over Union)
- **모델**: Florence-2-large-ft (Microsoft)

---

## 개발 환경

### 하드웨어
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) 1장 권장
- **RAM**: 32GB 이상 권장
- **Storage**: 50GB 이상 (데이터셋 포함)

### 소프트웨어
- **OS**: Linux (Ubuntu 20.04+) 또는 Windows WSL2
- **Python**: 3.8 이상
- **CUDA**: 11.8 이상
- **PyTorch**: 2.0 이상

---

## 설치 방법

### 1. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 2. 의존성 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 데이터셋 준비
대회에서 제공하는 데이터를 다운로드하여 다음과 같이 배치:

```
data/
├── train/
│   ├── report_json/
│   ├── report_jpg/
│   ├── press_json/
│   └── press_jpg/
├── valid/
│   ├── report_json/
│   ├── report_jpg/
│   ├── press_json/
│   └── press_jpg/
└── test/
    ├── images/
    └── query/
```

---

## 사용 방법

### 학습 (Training)

#### 기본 학습
```bash
python train.py \
  --train_press_json ./data/train/press_json \
  --train_press_jpg ./data/train/press_jpg \
  --train_report_json ./data/train/report_json \
  --train_report_jpg ./data/train/report_jpg \
  --valid_press_json ./data/valid/press_json \
  --valid_press_jpg ./data/valid/press_jpg \
  --valid_report_json ./data/valid/report_json \
  --valid_report_jpg ./data/valid/report_jpg \
  --epochs 10 \
  --batch_size 32 \
  --lr 5e-5 \
  --output_dir ./outputs/florence2_bbox
```

#### 주요 하이퍼파라미터
- `--epochs`: 학습 에포크 수 (기본값: 10)
- `--batch_size`: 배치 크기 (기본값: 32, VRAM 부족 시 16 또는 8로 조정)
- `--lr`: 학습률 (기본값: 5e-5)
- `--max_train_samples`: 학습 샘플 수 제한 (기본값: 8000)
- `--max_valid_samples`: 검증 샘플 수 제한 (기본값: 2000)

### 추론 (Inference)

#### 테스트 데이터 예측
```bash
python test.py \
  --test_dir ./data/test \
  --model_dir ./outputs/florence2_bbox/best \
  --batch_size 16 \
  --output_csv ./submission.csv
```

#### 출력 형식
생성된 CSV 파일은 다음과 같은 형식을 가집니다:
```csv
query_id,query_text,pred_x,pred_y,pred_w,pred_h
instance_001,2023년 매출 추이 표,125.3,456.7,890.2,345.6
```

---

## 모델 아키텍처

### Florence-2-large-ft
- **개발사**: Microsoft
- **유형**: Vision-Language Foundation Model
- **특징**:
  - 멀티모달 비전-언어 그라운딩
  - 한국어 쿼리 지원
  - Document understanding 특화
  - Phrase grounding task 수행

### 학습 전략
1. **데이터 전처리**
   - 문서 이미지 로딩 및 리사이징
   - 한국어 자연어 쿼리 토크나이징
   - Bounding box 정규화 (Florence-2 loc 토큰 형식)

2. **카테고리 균형 샘플링**
   - 표, 차트 종류별 균등 분포 유지
   - 과적합 방지

3. **학습**
   - AdamW optimizer
   - Cosine learning rate scheduling with warmup
   - Gradient clipping (max_norm=1.0)
   - Mixed precision training 지원

4. **평가**
   - Validation set에서 실시간 mIoU 계산
   - Best model checkpoint 저장

---

## 사전학습 모델 정보

### Florence-2-large-ft
- **출처**: [microsoft/Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft)
- **라이센스**: MIT License
- **다운로드**: Transformers library를 통해 자동 다운로드
- **사용 목적**: Vision-language grounding for document understanding

```python
# 모델 자동 다운로드 (train.py 실행 시)
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large-ft",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large-ft",
    trust_remote_code=True
)
```

---

## 파일 구조

```
.
├── train.py           # 학습 스크립트
├── test.py            # 추론 스크립트
├── requirements.txt   # Python 의존성
├── README.md          # 본 문서
└── outputs/           # 학습 결과물 (생성됨)
    └── florence2_bbox/
        └── best/      # 최고 성능 모델 체크포인트
```

---

## 예상 학습 시간

### RTX 3090 (24GB) 기준
- **1 Epoch**: 약 15-20분 (train 8000 samples, batch_size=32)
- **전체 학습 (10 epochs)**: 약 2.5-3시간
- **Validation**: 약 3-5분 per epoch

### 메모리 사용량
- **Batch size 32**: ~22GB VRAM
- **Batch size 16**: ~12GB VRAM
- **Batch size 8**: ~7GB VRAM

---

## 트러블슈팅

### CUDA Out of Memory
```bash
# Batch size 줄이기
python train.py --batch_size 16  # 또는 8
```

### 학습 속도 개선
```bash
# num_workers 조정
python train.py --num_workers 4  # CPU 코어 수에 맞게 조정
```

### 모델 로딩 에러
```bash
# 캐시 삭제 후 재다운로드
rm -rf ~/.cache/huggingface/
python train.py  # 모델 재다운로드
```

---

## 성능 향상 팁

1. **데이터 증강**: 현재 코드에는 기본 증강만 포함. 추가 증강 고려
2. **하이퍼파라미터 튜닝**: Learning rate, batch size, epochs 조정
3. **앙상블**: 여러 체크포인트 결과 평균
4. **후처리**: 예측된 bbox의 이상치 필터링

---

## 제출 방법

1. 학습 완료 후 best model 확인
2. 테스트 데이터로 추론 실행
3. 생성된 CSV 파일 제출

```bash
# 전체 파이프라인
python train.py [학습 인자]
python test.py --model_dir ./outputs/florence2_bbox/best --output_csv submission.csv
# submission.csv 제출
```

---

## 라이센스

본 프로젝트는 대회 목적으로만 사용됩니다.

## 참고 자료
- [Florence-2 Paper](https://arxiv.org/abs/2311.06242)
- [Hugging Face Model Card](https://huggingface.co/microsoft/Florence-2-large-ft)
- [대회 페이지](https://dacon.io)
