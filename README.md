# LoRA Character Training Pipeline

만화/웹툰 캐릭터를 학습시키는 자동화된 LoRA 파인튜닝 파이프라인

## 주요 기능

- **자동 데이터셋 전처리**: 만화 스크린샷에서 캐릭터 자동 크롭
- **텍스트/말풍선 제거**: OCR 기반 텍스트 영역 회피
- **캐릭터 전신 감지**: 배경 제거 기반 전신 크롭
- **LoRA 파인튜닝**: Stable Diffusion 모델 경량화 학습
- **자동 추론**: 학습된 모델로 이미지 생성

## 설치

```bash
# 1. 레포지토리 클론
git clone <repo-url>
cd lora

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirement.txt
```

## 사용 방법

### 1단계: 데이터셋 준비

```bash
# dataset 폴더에 만화/웹툰 스크린샷 넣기
mkdir -p dataset
cp /path/to/screenshots/*.png dataset/
```

**데이터셋 요구사항:**
- 형식: PNG, JPG, JPEG, WEBP
- 권장 개수: 20-50장
- 내용: 같은 캐릭터가 나오는 만화 컷
- 자동 처리: 텍스트, 말풍선, 여러 캐릭터 → 자동으로 크롭됨

### 2단계: 학습

```bash
python train.py
```

**학습 과정:**
1. 자동 전처리 (캐릭터 크롭, 텍스트 제거)
2. 모델 로딩 (Anything v4.0)
3. LoRA 파인튜닝 (100 epochs)
4. 모델 저장 (`my_lora_model/`)

**학습 설정 변경:**

`train.py`의 `Config` 클래스 수정:

```python
class Config:
    # 학습
    num_epochs = 100          # 에폭 수 (50-200 권장)
    learning_rate = 5e-5      # 학습률

    # LoRA
    lora_r = 32               # LoRA rank (8-64)
    lora_alpha = 64           # LoRA alpha

    # 데이터
    raw_dataset_path = "./dataset"
    clean_dataset_path = "./dataset_clean"
```

### 3단계: 이미지 생성

```bash
# 기본 사용
python inference.py

# 커스텀 프롬프트
python inference.py --prompt "smiling, outdoor, running"

# 여러 이미지 생성
python inference.py --num_images 5

# 고품질 생성
python inference.py --steps 50 --guidance_scale 9.0
```

**주요 옵션:**
- `--prompt`: 프롬프트 (자동으로 "sks girl" 추가됨)
- `--negative_prompt`: 네거티브 프롬프트
- `--num_images`: 생성할 이미지 수
- `--steps`: 추론 스텝 (20-50 권장)
- `--guidance_scale`: CFG scale (7-10 권장)
- `--seed`: 랜덤 시드 (재현성)
- `--lora_path`: LoRA 모델 경로

**예시:**
```bash
# 웃고 있는 캐릭터
python inference.py --prompt "smiling, happy expression"

# 특정 시드로 재생성
python inference.py --seed 42

# 다른 체크포인트 사용
python inference.py --lora_path my_lora_model_epoch50
```

## 프로젝트 구조

```
lora/
├── dataset/              # 원본 데이터셋 (만화 스크린샷)
├── dataset_clean/        # 전처리된 데이터셋 (자동 생성)
├── outputs/              # 생성된 이미지
├── my_lora_model/        # 학습된 LoRA 모델
├── train.py              # 학습 스크립트 (전처리 포함)
├── inference.py          # 추론 스크립트
├── preprocess_dataset.py # 전처리 모듈
├── requirement.txt       # 의존성
└── README.md
```

## 전처리 동작 방식

### 자동 처리 단계

1. **캐릭터 감지**: 배경 제거 (rembg)로 캐릭터 영역 탐지
2. **텍스트 감지**: OCR (EasyOCR)로 말풍선/텍스트 위치 파악
3. **스마트 크롭**: 텍스트 영역을 피해서 캐릭터 중심 크롭
4. **리사이즈**: 512x512 정사각형 (종횡비 유지, 패딩 추가)


## 학습 팁

### 데이터셋 품질

- ✅ **좋은 데이터**: 캐릭터 얼굴/전신이 잘 보이는 컷
- ❌ **나쁜 데이터**: 캐릭터가 가려지거나 흐릿한 컷

### 하이퍼파라미터 튜닝

**작은 데이터셋 (10-20장):**
- `num_epochs = 150-200`
- `lora_r = 32-64`
- `learning_rate = 5e-5`

**큰 데이터셋 (50-100장):**
- `num_epochs = 50-100`
- `lora_r = 16-32`
- `learning_rate = 1e-5`

### 모델 버전 관리

```bash
# 버전별 저장
my_lora_model_v1/
my_lora_model_v2/
my_lora_model_epoch50/
my_lora_model_epoch100/
```

## 기술 스택

- **Base Model**: Stable Diffusion (Anything v4.0)
- **Fine-tuning**: LoRA (PEFT)
- **Preprocessing**:
  - rembg (배경 제거)
  - EasyOCR (텍스트 감지)
  - OpenCV (이미지 처리)
- **Framework**: PyTorch, Diffusers, Transformers

## 라이센스
MIT License

## 참고 자료
- [Diffusers 문서](https://huggingface.co/docs/diffusers)
- [Anything v4.0 모델](https://huggingface.co/xyn-ai/anything-v4.0)
