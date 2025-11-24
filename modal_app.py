"""
Modal 배포 파일
LoRA 학습 및 이미지 생성을 위한 서버리스 GPU 애플리케이션
"""

import modal
import os
from pathlib import Path

# Modal 앱 생성
app = modal.App("lora-training-inference")

# 베이스 이미지 빌드 함수 (베이스 모델 포함)
def download_base_model_to_image():
    """
    이미지 빌드 시 베이스 모델을 다운로드하여 이미지에 포함시킵니다.
    컨테이너 시작 시간을 대폭 단축시킵니다.
    """
    from diffusers import StableDiffusionPipeline
    import torch

    model_id = "stablediffusionapi/anything-v5"
    cache_dir = "/base_models/anything-v5"

    print(f"📥 Downloading base model: {model_id}")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=cache_dir
    )

    # 모델을 로컬에 저장
    pipe.save_pretrained(cache_dir)

    print(f"✅ Base model cached to: {cache_dir}")


# 베이스 이미지 (라이브러리 + 베이스 모델 포함 - 캐싱 최적화)
# 패키지와 베이스 모델은 거의 변경되지 않으므로 한 번 빌드되면 계속 재사용됨
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "diffusers>=0.28.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "peft>=0.7.0",
        "safetensors>=0.4.0",
        "bitsandbytes>=0.41.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "rembg>=2.0.0",
        "onnxruntime>=1.16.0",
        "easyocr>=1.7.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "datasets>=2.14.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "boto3>=1.34.0",
        "requests>=2.31.0",
    )
    # 베이스 모델을 이미지에 포함 (첫 빌드 시 시간 걸리지만 이후 매우 빠름)
    .run_function(download_base_model_to_image)
)

# 최종 이미지 (core 디렉토리 추가 - 자주 변경되는 파일)
# core가 변경되어도 base_image는 재사용됨
image = base_image.add_local_dir(
    local_path=str(Path(__file__).parent / "core"),
    remote_path="/root/core"
)

# Modal Volume 설정 (모델 캐싱용)
volume = modal.Volume.from_name("lora-models", create_if_missing=True)
BASE_MODEL_PATH = "/base_models"
CACHE_DIR = "/cache"

# AWS Secrets 설정
secrets = modal.Secret.from_name("lora-secrets")

# 베이스 모델 다운로드 함수 (앱 시작 시 한 번만 실행)
@app.function(
    image=image,
    volumes={BASE_MODEL_PATH: volume},
    timeout=3600,  # 1시간
    secrets=[secrets]
)
def download_base_model():
    """
    베이스 Stable Diffusion 모델을 다운로드하여 캐싱합니다.
    """
    from diffusers import StableDiffusionPipeline
    import torch

    model_id = "stablediffusionapi/anything-v5"
    local_path = f"{BASE_MODEL_PATH}/anything-v5"

    print(f"Downloading base model: {model_id}")

    # 이미 다운로드된 경우 스킵
    if os.path.exists(local_path):
        print(f"✅ Base model already cached at {local_path}")
        return local_path

    # 다운로드
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    # 저장
    pipe.save_pretrained(local_path)
    volume.commit()

    print(f"✅ Base model downloaded to {local_path}")
    return local_path


# LoRA 학습 클래스 (GPU A10G 사용)
@app.cls(
    image=image,
    gpu="A10G",  # 학습용 GPU
    timeout=7200,  # 2시간
    volumes={
        CACHE_DIR: modal.Volume.from_name("lora-cache", create_if_missing=True)
    },
    secrets=[secrets],
    memory=32768,  # 32GB RAM
    enable_memory_snapshot=True,  # 메모리 스냅샷 활성화 - 부팅 시간 획기적 단축!
)
class LoraTrainer:
    """
    LoRA 학습을 처리하는 클래스

    enable_memory_snapshot=True로 인해:
    - 첫 번째 컨테이너 부팅 시: 모델 로드 + 메모리 스냅샷 생성 (느림)
    - 이후 컨테이너들: 메모리 스냅샷에서 복원 (매우 빠름, 수 초 이내)
    """

    @modal.enter()
    def load_models(self):
        """
        컨테이너 시작 시 베이스 모델을 메모리에 로드합니다.
        메모리 스냅샷이 활성화되어 있어 이 초기화는 한 번만 수행됩니다.
        """
        from diffusers import StableDiffusionPipeline
        import torch

        print("🚀 Initializing LoRA Trainer...")

        # 베이스 모델 경로
        self.base_model_path = "/base_models/anything-v5"

        if not os.path.exists(self.base_model_path):
            print("⚠️ Base model not found in image, using HuggingFace")
            self.base_model_path = "stablediffusionapi/anything-v5"

        print(f"📦 Loading base model from: {self.base_model_path}")

        # 파이프라인 로드 (메모리 스냅샷에 포함됨)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipe.to("cuda")

        print("✅ Base model loaded and ready!")
        print("💾 Memory snapshot will be created after this initialization")

    @modal.method()
    def train_lora(
        self,
        user_id: str,
        model_id: int,
        model_name: str,
        training_image_urls: list[str],
        callback_url: str = None
    ):
        """
        LoRA 학습을 실행합니다.

        Args:
            user_id: 사용자 ID
            model_id: 모델 ID
            model_name: 모델 이름
            training_image_urls: S3 presigned URL 리스트
            callback_url: 완료 시 호출할 콜백 URL

        Returns:
            dict: 학습 결과
        """
        from core.config import TrainingConfig
        from core.train import train_with_preprocessing
        import requests
        import shutil
        import boto3
        import time

        print(f"Starting training for user: {user_id}, model: {model_name}")
        print(f"Number of training images: {len(training_image_urls)}")

        # 진행률 콜백 함수
        def send_progress_callback(status, message):
            """백엔드로 진행률 전송"""
            if callback_url:
                try:
                    progress_data = {
                        "userId": user_id,
                        "modelId": model_id,
                        "status": status,
                        "message": message
                    }
                    requests.post(callback_url, json=progress_data, timeout=2)
                    print(f"📊 Progress sent: {status} - {message}")
                except Exception as e:
                    print(f"⚠️ Failed to send progress: {e}")

        # 1. 서버 로드 중
        send_progress_callback("LOADING", "Loading server")
        time.sleep(0.5)

        # 2. 이미지 전처리 중 (다운로드)
        send_progress_callback("PREPROCESSING", "Preprocessing images")

        # S3 이미지 다운로드
        temp_dataset_path = f"{CACHE_DIR}/dataset_{user_id}_{model_name}"
        os.makedirs(temp_dataset_path, exist_ok=True)

        print("Downloading training images from S3...")
        for idx, url in enumerate(training_image_urls):
            ext = ".jpg"
            if ".png" in url.lower():
                ext = ".png"

            local_path = os.path.join(temp_dataset_path, f"image_{idx:04d}{ext}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Downloaded {len(training_image_urls)} images")

        # 학습 설정
        output_dir = f"{CACHE_DIR}/models/{user_id}/{model_name}"

        config = TrainingConfig(
            raw_dataset_path=temp_dataset_path,
            output_dir=output_dir,
            model_id=self.base_model_path
        )

        # 학습 콜백 함수 (에포크별 진행률)
        def training_callback(status, phase, current_epoch, total_epochs, message):
            """학습 진행률 콜백"""
            if phase == "training" and current_epoch > 0:
                progress_message = f"Training {current_epoch}/{total_epochs}"
                send_progress_callback("TRAINING", progress_message)

        # 학습 실행
        try:
            # 3. 학습 시작
            send_progress_callback("TRAINING", "Training 0/{}".format(config.num_epochs))

            train_result = train_with_preprocessing(
                raw_dataset_path=temp_dataset_path,
                output_dir=output_dir,
                config=config,
                skip_preprocessing=False,
                callback=training_callback
            )

            # 학습된 모델 파일 경로 찾기 (최종 checkpoint의 safetensors 파일)
            # output_dir/checkpoint-{최종epoch}/lora_weights.safetensors
            import glob
            checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
            if not checkpoint_dirs:
                raise FileNotFoundError(f"No checkpoint found in {output_dir}")

            # 가장 높은 에포크 번호의 checkpoint 찾기
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
            model_file_path = os.path.join(latest_checkpoint, "lora_weights.safetensors")

            if not os.path.exists(model_file_path):
                raise FileNotFoundError(f"LoRA weights not found at {model_file_path}")

            print(f"Found trained model: {model_file_path}")

            # 4. 모델 업로드 중
            send_progress_callback("UPLOADING", "Uploading model")

            # S3 업로드 (단일 .safetensors 파일만 - Civitai 방식)
            print("Uploading trained model to S3...")
            s3_client = boto3.client('s3')
            bucket_name = os.environ.get("AWS_S3_MODELS_BUCKET", "lora-models-bucket")

            # S3 키: model-{modelId}/{modelName}.safetensors
            s3_model_key = f"model-{model_id}/{model_name}.safetensors"
            s3_client.upload_file(
                model_file_path,
                bucket_name,
                s3_model_key,
                ExtraArgs={'ContentType': 'application/octet-stream'}
            )
            print(f"✅ Uploaded model (WebUI format): s3://{bucket_name}/{s3_model_key}")

            file_size = os.path.getsize(model_file_path)

            # 콜백 호출
            if callback_url:
                try:
                    callback_data = {
                        "userId": user_id,
                        "modelId": model_id,
                        "modelName": model_name,
                        "s3ModelKey": s3_model_key,
                        "fileSize": file_size,
                        "status": "SUCCESS"
                    }
                    response = requests.post(callback_url, json=callback_data, timeout=10)
                    response.raise_for_status()
                    print(f"✅ Callback successful: {callback_url}")
                except Exception as e:
                    print(f"❌ Callback failed: {e}")

            # 임시 파일 정리
            shutil.rmtree(temp_dataset_path, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

            return {
                "status": "SUCCESS",
                "s3_model_key": s3_model_key,
                "file_size": file_size,
                **train_result
            }

        except Exception as e:
            print(f"❌ Training failed: {e}")

            # 실패 콜백
            if callback_url:
                try:
                    callback_data = {
                        "userId": user_id,
                        "modelId": model_id,
                        "modelName": model_name,
                        "status": "FAIL",
                        "error": str(e)
                    }
                    requests.post(callback_url, json=callback_data, timeout=10)
                    print(f"❌ Failure callback sent to backend for model {model_id}.")
                except Exception as cb_e:
                    print(f"⚠️ Failed to send failure callback: {cb_e}")

            # 정리
            shutil.rmtree(temp_dataset_path, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

            raise


# 이미지 생성 클래스 (GPU T4 사용)
@app.cls(
    image=image,
    gpu="T4",  # 생성용 GPU (T4 사용)
    timeout=600,  # 10분
    volumes={
        CACHE_DIR: modal.Volume.from_name("lora-cache", create_if_missing=True)
    },
    secrets=[secrets],
    memory=16384,  # 16GB RAM
    enable_memory_snapshot=True,  # 메모리 스냅샷 활성화 - 부팅 시간 획기적 단축!
)
class ImageGenerator:
    """
    이미지 생성을 처리하는 클래스

    enable_memory_snapshot=True로 인해:
    - 첫 번째 컨테이너 부팅 시: 모델 로드 + 메모리 스냅샷 생성 (느림)
    - 이후 컨테이너들: 메모리 스냅샷에서 복원 (매우 빠름, 수 초 이내)
    """

    @modal.enter()
    def load_models(self):
        """
        컨테이너 시작 시 베이스 모델을 메모리에 로드합니다.
        메모리 스냅샷이 활성화되어 있어 이 초기화는 한 번만 수행됩니다.
        """
        from diffusers import StableDiffusionPipeline
        import torch

        print("🚀 Initializing Image Generator...")

        # 베이스 모델 경로
        self.base_model_path = "/base_models/anything-v5"

        if not os.path.exists(self.base_model_path):
            print("⚠️ Base model not found in image, using HuggingFace")
            self.base_model_path = "stablediffusionapi/anything-v5"

        print(f"📦 Loading base model from: {self.base_model_path}")

        # 파이프라인 로드 (메모리 스냅샷에 포함됨)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipe.to("cuda")

        print("✅ Base model loaded and ready!")
        print("💾 Memory snapshot will be created after this initialization")

    @modal.method()
    def generate_images(
        self,
        user_id: str,
        prompt: str,
        lora_model_url: str,
        model_id: int,
        history_id: int = None,
        negative_prompt: str = "low quality, blurry, ugly, distorted, deformed",
        num_images: int = 1,
        steps: int = 40,
        guidance_scale: float = 7.5,
        seed: int = None,
        callback_url: str = None
    ):
        """
        이미지 생성을 실행합니다.

        Args:
            user_id: 사용자 ID
            prompt: 이미지 생성 프롬프트
            lora_model_url: S3 LoRA 모델 URL
            model_id: 모델 ID
            history_id: GenerationHistory ID
            negative_prompt: 네거티브 프롬프트
            num_images: 생성할 이미지 수
            steps: 추론 스텝
            guidance_scale: CFG scale
            seed: 랜덤 시드
            callback_url: 완료 시 콜백 URL

        Returns:
            list: S3 키 리스트
        """
        from core.config import InferenceConfig
        from core.generate import generate_images
        from urllib.parse import urlparse, unquote
        import requests
        import boto3
        import shutil

        print(f"Generating images for user: {user_id}")
        print(f"Prompt: {prompt}")
        print(f"Number of images: {num_images}")

        # LoRA 모델 캐시 경로
        temp_lora_path = f"{CACHE_DIR}/lora_models/model_{model_id}"

        # 이미 캐시된 모델이 있는지 확인
        if os.path.exists(temp_lora_path):
            safetensors_files = [f for f in os.listdir(temp_lora_path) if f.endswith('.safetensors')]
            if safetensors_files:
                print(f"✅ Using cached LoRA model (shared): {temp_lora_path}")
                print(f"Cached files: {os.listdir(temp_lora_path)}")
                downloaded_files = [os.path.join(temp_lora_path, f) for f in os.listdir(temp_lora_path)]
            else:
                print(f"⚠️ Cache corrupted, re-downloading...")
                downloaded_files = None
        else:
            downloaded_files = None

        if downloaded_files is None:
            # 캐시가 없으면 S3에서 다운로드
            os.makedirs(temp_lora_path, exist_ok=True)
            print(f"Downloading LoRA model from S3...")

            s3_client = boto3.client('s3')
            bucket_name = os.environ.get("AWS_S3_MODELS_BUCKET", "lora-models-bucket")

            # lora_model_url 파싱 (단일 .safetensors 파일 경로)
            if lora_model_url.startswith('http'):
                # HTTP URL에서 S3 키 추출
                parsed = urlparse(lora_model_url)
                path_parts = parsed.path.strip('/').split('/')
                # bucket 이름 다음부터가 S3 키
                if len(path_parts) >= 2:
                    s3_key = '/'.join(path_parts[1:])
                else:
                    s3_key = path_parts[-1]
                s3_key = unquote(s3_key)
            else:
                # s3://bucket/key 형식
                s3_key = lora_model_url.replace(f's3://{bucket_name}/', '')
                s3_key = unquote(s3_key)

            print(f"S3 key: {s3_key}")

            # 단일 safetensors 파일 다운로드
            local_file = os.path.join(temp_lora_path, os.path.basename(s3_key))

            print(f"Downloading: s3://{bucket_name}/{s3_key} -> {local_file}")
            s3_client.download_file(bucket_name, s3_key, local_file)
            downloaded_files = [local_file]

            print(f"✅ LoRA model downloaded to {temp_lora_path}")

        # .safetensors 파일 확인
        safetensors_files = [f for f in os.listdir(temp_lora_path) if f.endswith('.safetensors')]
        if not safetensors_files:
            print(f"❌ ERROR: No .safetensors file found in {temp_lora_path}")
            raise FileNotFoundError(f"No .safetensors file found in {temp_lora_path}")

        # 출력 디렉토리
        output_dir = f"{CACHE_DIR}/outputs/{user_id}"
        os.makedirs(output_dir, exist_ok=True)

        config = InferenceConfig(
            model_id=self.base_model_path,
            lora_path=temp_lora_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            output_dir=output_dir
        )

        # 진행률 콜백 함수 (백엔드로 POST 요청)
        import time
        last_update_time = [0]  # mutable object to track last update time

        def progress_callback(status, current_image, total_images, current_step, total_steps, message):
            """1초마다 백엔드로 진행률 전송"""
            current_time = time.time()

            # 1초마다만 전송 (첫 호출은 즉시 전송)
            if current_time - last_update_time[0] < 1.0 and last_update_time[0] != 0:
                return

            last_update_time[0] = current_time

            if history_id and callback_url:
                try:
                    progress_data = {
                        "historyId": history_id,
                        "status": "GENERATING",
                        "currentStep": current_step,
                        "totalSteps": total_steps,
                        "message": message
                    }
                    requests.post(callback_url, json=progress_data, timeout=2)
                    print(f"📊 Progress sent to backend: {message} ({current_step}/{total_steps})")
                except Exception as e:
                    print(f"⚠️ Failed to send progress: {e}")

        # 이미지 생성
        try:
            generated_files = generate_images(
                lora_path=temp_lora_path,
                config=config,
                callback=progress_callback
            )

            # S3 업로드
            print("Uploading generated images to S3...")
            s3_client = boto3.client('s3')
            bucket_name = os.environ.get("AWS_S3_IMAGES_BUCKET", "lora-generated-image-bucket")

            s3_keys = []
            for generated_file in generated_files:
                filename = os.path.basename(generated_file)
                s3_key = f"user-{user_id}/{filename}"

                s3_client.upload_file(
                    generated_file,
                    bucket_name,
                    s3_key,
                    ExtraArgs={'ContentType': 'image/png'}
                )

                s3_keys.append(s3_key)
                print(f"✅ Uploaded: s3://{bucket_name}/{s3_key}")

            # 콜백 호출
            if callback_url:
                try:
                    callback_data = {
                        "historyId": history_id,
                        "userId": user_id,
                        "modelId": model_id,
                        "prompt": prompt,
                        "negativePrompt": negative_prompt,
                        "steps": steps,
                        "guidanceScale": guidance_scale,
                        "seed": seed,
                        "numImages": num_images,
                        "imageS3Keys": s3_keys,
                        "status": "SUCCESS"
                    }
                    response = requests.post(callback_url, json=callback_data, timeout=10)
                    response.raise_for_status()
                    print(f"✅ Callback successful: {callback_url}")
                except Exception as e:
                    print(f"❌ Callback failed: {e}")

            # 진행률 딕셔너리에서 제거 (완료됨)
            if history_id and history_id in generation_progress:
                del generation_progress[history_id]

            # 임시 파일 정리 (LoRA 모델은 캐시로 유지)
            shutil.rmtree(output_dir, ignore_errors=True)

            return s3_keys

        except Exception as e:
            print(f"❌ Image generation failed: {e}")

            # 실패 콜백
            if callback_url:
                try:
                    callback_data = {
                        "historyId": history_id,
                        "userId": user_id,
                        "modelId": model_id,
                        "status": "FAIL",
                        "error": str(e)
                    }
                    requests.post(callback_url, json=callback_data, timeout=10)
                except Exception as cb_e:
                    print(f"⚠️ Failed to send failure callback: {cb_e}")
            print(f"❌ Failure callback sent to backend for history {history_id}.")

            # 진행률 딕셔너리에서 제거 (실패함)
            if history_id and history_id in generation_progress:
                del generation_progress[history_id]

            shutil.rmtree(output_dir, ignore_errors=True)

            raise


# 전역 상태 저장용 딕셔너리 (SSE 진행률 스트리밍)
# {history_id: {"status": "GENERATING", "current_step": 1, "total_steps": 30, ...}}
generation_progress = {}

# FastAPI 웹 서버
@app.function(
    image=image,
    secrets=[secrets],
    min_containers=1,  # 항상 1개 인스턴스 유지
)
@modal.asgi_app()
def fastapi_app():
    """
    FastAPI 웹 서버 (Modal 배포)
    """
    from fastapi import FastAPI, BackgroundTasks, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from typing import Optional, List
    import asyncio
    import json

    web_app = FastAPI(
        title="LoRA Training and Inference API (Modal)",
        description="Modal에서 실행되는 서버리스 GPU 기반 LoRA 학습 및 이미지 생성 API",
        version="2.0.0",
    )

    # CORS 설정
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 모든 도메인 허용
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic 모델
    class TrainRequest(BaseModel):
        user_id: str = Field(..., description="사용자 ID")
        model_id: int = Field(..., description="모델 ID")
        model_name: str = Field(..., description="모델 이름")
        training_image_urls: List[str] = Field(..., description="S3 학습 이미지 URL 리스트")
        callback_url: Optional[str] = Field(None, description="학습 완료 시 콜백 URL")

    class GenerateRequest(BaseModel):
        user_id: str = Field(..., description="사용자 ID")
        model_id: int = Field(..., description="모델 ID")
        history_id: Optional[int] = Field(None, description="GenerationHistory ID")
        prompt: str = Field(..., description="이미지 생성 프롬프트")
        lora_model_url: str = Field(..., description="S3 LoRA 모델 URL")
        negative_prompt: Optional[str] = Field("low quality, blurry, ugly, distorted, deformed")
        num_images: int = Field(1, description="생성할 이미지 수")
        steps: int = Field(40, description="추론 스텝")
        guidance_scale: float = Field(7.5, description="CFG scale")
        seed: Optional[int] = Field(None, description="랜덤 시드")
        callback_url: Optional[str] = Field(None, description="완료 시 콜백 URL")

    class MessageResponse(BaseModel):
        message: str

    # 엔드포인트
    @web_app.get("/")
    def root():
        return {"message": "LoRA Modal API is running"}

    @web_app.post("/train")
    async def start_training(req: TrainRequest):
        """학습 시작 (비동기)"""
        try:
            # LoraTrainer 클래스 메서드 호출 (spawn으로 비동기 실행)
            trainer = LoraTrainer()
            trainer.train_lora.spawn(
                user_id=req.user_id,
                model_id=req.model_id,
                model_name=req.model_name,
                training_image_urls=req.training_image_urls,
                callback_url=req.callback_url
            )

            return {"message": "Training started on Modal GPU (A10G)"}

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"message": f"Failed to start training: {str(e)}"}
            )

    @web_app.post("/generate")
    async def generate_images_api(req: GenerateRequest):
        """이미지 생성 (비동기)"""
        try:
            # ImageGenerator 클래스 메서드 호출 (spawn으로 비동기 실행)
            generator = ImageGenerator()
            call = generator.generate_images.spawn(
                user_id=req.user_id,
                model_id=req.model_id,
                history_id=req.history_id,
                prompt=req.prompt,
                lora_model_url=req.lora_model_url,
                negative_prompt=req.negative_prompt,
                num_images=req.num_images,
                steps=req.steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                callback_url=req.callback_url
            )

            return {"message": "Image generation started on Modal GPU (T4)", "call_id": call.object_id}

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"message": f"Failed to start generation: {str(e)}"}
            )

    @web_app.get("/generate/stream")
    async def stream_generation_progress():
        """SSE 스트리밍 - 이미지 생성 진행률"""
        async def event_generator():
            last_sent = {}

            while True:
                # 진행 중인 작업이 있는지 확인
                if generation_progress:
                    for history_id, progress in generation_progress.items():
                        # 변경사항이 있을 때만 전송
                        if last_sent.get(history_id) != progress:
                            data = {
                                "status": "IN_PROGRESS",
                                "historyId": history_id,
                                "current_step": progress.get("current_step", 0),
                                "total_steps": progress.get("total_steps", 0),
                                "message": progress.get("message", "Generating...")
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                            last_sent[history_id] = progress.copy()

                await asyncio.sleep(0.5)  # 0.5초마다 체크

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Nginx 버퍼링 비활성화
            }
        )

    @web_app.get("/health")
    def health_check():
        return {"status": "healthy"}

    return web_app


# CLI 명령어
@app.local_entrypoint()
def main():
    """
    로컬에서 Modal 함수 테스트
    """
    print("Starting Modal deployment...")

    # 베이스 모델 다운로드 (최초 1회)
    print("Downloading base model...")
    download_base_model.remote()

    print("✅ Modal deployment ready!")
    print("Deploy with: modal deploy modal_app.py")
