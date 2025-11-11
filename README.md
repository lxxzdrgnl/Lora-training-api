# LoRA Character Training Pipeline

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

만화/웹툰 캐릭터를 학습시키는 **자동화된 LoRA 파인튜닝 파이프라인**

> 만화 스크린샷만 넣으면 자동으로 텍스트 제거, 캐릭터 크롭, LoRA 학습까지 원클릭!

## 주요 기능

- **자동 데이터셋 전처리**: 만화 스크린샷에서 캐릭터 자동 크롭
- **텍스트/말풍선 제거**: OCR 기반 텍스트 감지 + Inpainting 제거
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



## FastAPI 서버

LoRA 학습 및 이미지 생성 기능을 RESTful API로 제공합니다.

### 설치

`requirements.txt`에 추가된 `fastapi`와 `uvicorn`을 설치합니다.

```bash
pip install -r requirements.txt
```

### 서버 실행

프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다.

```bash
uvicorn main:app --reload
```

서버는 `http://127.0.0.1:8000`에서 실행되며, API 문서는 `http://127.0.0.1:8000/docs`에서 확인할 수 있습니다.

### API 엔드포인트 상세

*   **`GET /`**
    *   **설명**: 서버 상태를 확인합니다.
    *   **성공 응답 (200)**:
        ```json
        {
          "message": "LoRA FastAPI server is running."
        }
        ```

*   **`POST /train`**
    *   **설명**: LoRA 모델 학습을 시작합니다. 학습은 백그라운드에서 실행됩니다.
    *   **요청 본문**:
        ```json
        {
          "raw_dataset_path": "./dataset",
          "output_dir": "my_lora_model",
          "skip_preprocessing": false
        }
        ```
    *   **성공 응답 (200)**:
        ```json
        {
          "message": "Training started in the background. Check /train/status for progress."
        }
        ```
    *   **에러 응답 (400)**: 이미 학습이 진행 중일 때 발생합니다.
        ```json
        {
          "message": "Training is already in progress."
        }
        ```

*   **`GET /train/status`**
    *   **설명**: 현재 학습 진행 상태를 확인합니다.
    *   **성공 응답 (200)**: 여러 상태에 대한 예시입니다.
        *   학습 진행 중:
            ```json
            {
              "is_training": true,
              "message": "Training in progress..."
            }
            ```
        *   학습 완료:
            ```json
            {
              "is_training": false,
              "message": "Training completed successfully."
            }
            ```
        *   학습 중 아님 (초기 상태):
            ```json
            {
              "is_training": false,
              "message": "Not training"
            }
            ```

*   **`POST /generate`**
    *   **설명**: 프롬프트를 기반으로 이미지를 생성하고, 생성된 이미지의 URL을 반환합니다.
    *   **요청 본문**:
        ```json
        {
          "prompt": "1girl, black hair, long hair, black and white manga style",
          "lora_path": "my_lora_model"
        }
        ```
    *   **성공 응답 (200)**:
        ```json
        {
          "image_url": "http://127.0.0.1:8000/static/20251111_123456_1.png"
        }
        ```
    *   **에러 응답 (404)**: LoRA 모델을 찾을 수 없을 때 발생합니다.
        ```json
        {
          "message": "LoRA model not found at my_lora_model. Please train the model first."
        }
        ```
    *   **에러 응답 (422)**: 요청 본문의 내용이 유효하지 않을 때 발생합니다. (예: `prompt` 필드 누락)
        ```json
        {
          "detail": [
            {
              "loc": [
                "body",
                "prompt"
              ],
              "msg": "field required",
              "type": "value_error.missing"
            }
          ]
        }
        ```
    *   **에러 응답 (500)**: 서버 내부에서 이미지 생성 중 오류가 발생했을 때입니다.
        ```json
        {
          "message": "An error occurred during image generation: Some internal server error."
        }
        ```



## 프로젝트 구조



```

lora/

├── .git/                 # Git 저장소 파일

├── .gitignore            # Git 무시 파일

├── core/                 # 학습, 생성, 전처리를 위한 핵심 모듈

│   ├── __init__.py

│   ├── config.py         # 설정 파일

│   ├── generate.py       # 이미지 생성 로직

│   ├── preprocess.py     # 데이터셋 전처리 로직

│   └── train.py          # 학습 로직

├── dataset/              # 원본 데이터셋 (만화 스크린샷)

├── dataset_clean/        # 전처리된 데이터셋 (자동 생성)

├── generate.py           # 이미지 생성을 위한 메인 스크립트

├── main.py               # FastAPI 애플리케이션 엔트리포인트

├── outputs/              # 생성된 이미지

├── my_lora_model/        # 학습된 LoRA 모델 (예: my_lora_model_epoch100)

├── README.md             # 프로젝트 README 파일

├── requirements.txt      # Python 의존성

├── train.py              # 학습을 위한 메인 스크립트 (전처리 포함)

└── venv/                 # Python 가상 환경

```



## 전처리 동작 방식



### 자동 처리 단계



1.  **텍스트 감지**: OCR (EasyOCR)로 말풍선/텍스트 위치 파악

2.  **텍스트 제거**: Inpainting으로 텍스트 영역 자동 제거

3.  **캐릭터 감지**: 배경 제거 (rembg)로 캐릭터 영역 탐지

4.  **스마트 크롭**: 캐릭터 중심으로 bbox 확장 (전신 포함)

5.  **리사이즈**: 512x512 정사각형 (종횡비 유지, 패딩 추가)





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



**과적합 증상:**

- Loss가 계속 감소하지만 생성 이미지 품질이 떨어짐

- 해결: Epoch 수 줄이기, Learning rate 낮추기



**과소적합 증상:**

- 캐릭터가 원본과 많이 다름

- 해결: Epoch 수 늘리기, LoRA rank 높이기





## 기술 스택



### Deep Learning & AI

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

![Hugging Face](https://img.shields.io/badge/🤗_Diffusers-Latest-FFD21E?style=for-the-badge)

![Transformers](https://img.shields.io/badge/Transformers-Latest-FF6F00?style=for-the-badge)



### Model & Fine-tuning

![Stable Diffusion](https://img.shields.io/badge/Stable_Diffusion-Anything_v4.0-9C27B0?style=for-the-badge)

![LoRA](https://img.shields.io/badge/LoRA-PEFT-00C853?style=for-the-badge)



### Preprocessing

![OpenCV](https://img.shields.io/badge/OpenCV-Latest-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

![EasyOCR](https://img.shields.io/badge/EasyOCR-Text_Detection-FF6B6B?style=for-the-badge)

![rembg](https://img.shields.io/badge/rembg-Background_Removal-00D9FF?style=for-the-badge)



## 트러블슈팅



### Q: 전처리가 실패함 (캐릭터 감지 안됨)

**A:** 배경이 너무 복잡하거나 캐릭터가 작은 경우 발생합니다.

- 해결: 수동으로 크롭하거나 다른 이미지를 사용하세요.



### Q: 생성된 이미지가 깨져있거나 노이즈가 많음

**A:** 데이터셋에 텍스트/말풍선이 제대로 제거되지 않았을 수 있습니다.

- 해결: `dataset_clean/` 폴더를 확인해 전처리가 제대로 됐는지 체크하세요.

- 전처리를 다시 실행하려면: `rm -rf dataset_clean && python train.py`



### Q: CUDA out of memory 에러

**A:** GPU 메모리 부족입니다.

```python

# train.py Config 수정

gradient_accumulation_steps = 2  # 1 → 2로 변경

# 또는 이미지 크기 축소

image_size = 512  # → 384

```



### Q: 학습이 너무 느림

**A:** GPU 사용을 확인하세요:

```bash

nvidia-smi  # GPU 사용 확인

```



## 라이센스



MIT License



## 참고 자료



- [LoRA 논문](https://arxiv.org/abs/2106.09685)

- [Diffusers 문서](https://huggingface.co/docs/diffusers)

- [PEFT 라이브러리](https://github.com/huggingface/peft)

- [Anything v4.0 모델](https://huggingface.co/xyn-ai/anything-v4.0)

- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

- [rembg](https://github.com/danielgatis/rembg)


