"""
FastAPI 애플리케이션
LoRA 학습 및 이미지 생성을 위한 API
"""

import os
import json
import asyncio
import requests
import shutil
from pathlib import Path
from threading import Lock
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Union

from core.config import TrainingConfig, InferenceConfig
from core.train import train_with_preprocessing
from core.generate import generate_images

app = FastAPI(
    title="LoRA Training and Inference API",
    description="""
LoRA 모델 학습 및 이미지 생성을 위한 RESTful API 서버입니다.

## 주요 기능
- **학습**: 백그라운드에서 비동기 LoRA 모델 학습
- **이미지 생성**: 학습된 모델로 프롬프트 기반 이미지 생성
- **정적 파일 서빙**: 생성된 이미지를 `/static/` 경로로 제공
- **CORS 지원**: Vue.js 등 프론트엔드에서 직접 접근 가능

## 이미지 저장 및 접근
- 생성된 이미지는 `outputs/` 폴더에 저장됩니다
- 브라우저에서 `http://localhost:8000/static/이미지명.png` 로 직접 접근 가능
- CORS가 설정되어 있어 다른 도메인에서도 이미지 로드 가능
    """,
    version="1.0.0",
)

# CORS 설정 - Vue에서 API 및 정적 파일 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://bluemingai.ap-northeast-2.elasticbeanstalk.com"
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:5173",  # Vite 기본 포트
        "http://blueming-front.s3-website.ap-northeast-2.amazonaws.com/"
        ,  # localhost 대신 127.0.0.1
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# --- 정적 파일 마운트 ---
# 'outputs' 디렉토리를 '/static' 경로에 마운트하여 이미지 URL로 접근 가능하게 합니다.
os.makedirs("outputs", exist_ok=True)
os.makedirs("temp", exist_ok=True)  # 임시 파일 저장용 디렉토리
app.mount("/static", StaticFiles(directory="outputs"), name="static")

# --- S3 다운로드/업로드 유틸리티 ---
def download_from_s3_url(s3_url: str, local_path: str) -> str:
    """
    S3 Presigned URL에서 파일을 다운로드합니다.

    Args:
        s3_url: S3 Presigned URL
        local_path: 로컬 저장 경로

    Returns:
        다운로드된 파일의 경로
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    response = requests.get(s3_url, stream=True)
    response.raise_for_status()

    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_path


def upload_to_s3(local_file_path: str, s3_key: str, bucket_name: str = "lora-training-data-bucket", content_type: str = None) -> str:
    """
    로컬 파일을 S3에 업로드합니다.

    Args:
        local_file_path: 업로드할 로컬 파일 경로
        s3_key: S3 키 (예: models/user-123/model.safetensors)
        bucket_name: S3 버킷 이름
        content_type: Content-Type (예: image/png, application/octet-stream)

    Returns:
        S3 Key
    """
    import boto3
    from botocore.exceptions import ClientError

    try:
        s3_client = boto3.client('s3')

        # Content-Type 자동 감지
        if content_type is None:
            if local_file_path.endswith('.png'):
                content_type = 'image/png'
            elif local_file_path.endswith('.jpg') or local_file_path.endswith('.jpeg'):
                content_type = 'image/jpeg'
            else:
                content_type = 'application/octet-stream'

        # 파일 업로드
        s3_client.upload_file(
            local_file_path,
            bucket_name,
            s3_key,
            ExtraArgs={'ContentType': content_type}
        )

        print(f"✅ Uploaded to S3: s3://{bucket_name}/{s3_key}")
        return s3_key

    except ClientError as e:
        print(f"❌ S3 upload failed: {e}")
        raise

def download_training_images(s3_urls: List[str]) -> str:
    """
    여러 S3 URL에서 학습용 이미지들을 다운로드합니다.

    Args:
        s3_urls: S3 Presigned URL 리스트

    Returns:
        데이터셋 경로 (temp/dataset_xxxxx)
    """
    # temp 디렉토리 안에 고유한 데이터셋 폴더 생성
    dataset_id = os.urandom(8).hex()
    dataset_path = os.path.join("temp", f"dataset_{dataset_id}")
    os.makedirs(dataset_path, exist_ok=True)

    for idx, s3_url in enumerate(s3_urls):
        # 파일 확장자 추출 (URL에서)
        ext = ".jpg"  # 기본값
        if ".png" in s3_url.lower():
            ext = ".png"
        elif ".jpeg" in s3_url.lower() or ".jpg" in s3_url.lower():
            ext = ".jpg"

        local_path = os.path.join(dataset_path, f"image_{idx:04d}{ext}")
        download_from_s3_url(s3_url, local_path)

    return dataset_path

# --- 모델 및 상태 관리 ---
training_status = {
    "status": "IDLE",  # IDLE, PREPROCESSING, TRAINING, SUCCESS, FAIL
    "progress": {
        "phase": "",  # "preprocessing" or "training"
        "current_epoch": 0,
        "total_epochs": 0
    },
    "message": "대기 중"
}
training_lock = Lock()

generation_status = {
    "status": "IDLE",  # IDLE, GENERATING, SUCCESS, FAIL
    "progress": {
        "current_image": 0,
        "total_images": 0,
        "current_step": 0,
        "total_steps": 0
    },
    "message": "대기 중"
}
generation_lock = Lock()

# --- Pydantic 모델 정의 ---

# 요청 모델
class TrainRequest(BaseModel):
    user_id: str = Field(..., description="사용자 ID")
    model_name: str = Field(..., description="모델 이름")
    training_image_urls: Optional[List[str]] = Field(None, description="S3에 업로드된 학습 이미지 URL 리스트")
    raw_dataset_path: Optional[str] = Field(None, description="원본 데이터셋 경로 (로컬 파일 사용 시)")
    output_dir: Optional[str] = Field(None, description="학습된 모델이 저장될 경로 (자동 생성됨)")
    skip_preprocessing: bool = Field(False, description="전처리 과정 스킵 여부")
    callback_url: Optional[str] = Field(None, description="학습 완료 시 호출할 Spring Boot API URL")

class GenerateRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="사용자 ID (S3 경로 생성용)")
    prompt: str = Field(..., description="이미지 생성을 위한 프롬프트")
    negative_prompt: Optional[str] = Field("low quality, blurry, ugly, distorted, deformed", description="이미지 생성 시 제외할 요소들에 대한 프롬프트")
    lora_model_url: Optional[str] = Field(None, description="S3에 저장된 LoRA 모델 파일 URL (.safetensors)")
    lora_path: Optional[str] = Field(None, description="사용할 LoRA 모델 경로 (로컬 파일 사용 시)")
    num_images: int = Field(1, description="생성할 이미지 개수")
    steps: int = Field(40, description="이미지 생성 스텝 수")
    guidance_scale: float = Field(7.5, description="프롬프트 충실도 (CFG Scale)")
    seed: Optional[int] = Field(None, description="재현성을 위한 랜덤 시드")

# 응답 모델
class MessageResponse(BaseModel):
    message: str

class ProgressInfo(BaseModel):
    phase: str = Field("", description="현재 단계 (preprocessing 또는 training)")
    current_epoch: int = Field(0, description="현재 완료된 에포크 수")
    total_epochs: int = Field(0, description="총 에포크 수")

class TrainStatusResponse(BaseModel):
    status: str = Field(..., description="학습 상태: IDLE, PREPROCESSING, TRAINING, SUCCESS, FAIL")
    progress: ProgressInfo = Field(..., description="진행도 정보")
    message: str = Field(..., description="상태 메시지")

class GenerateResponse(BaseModel):
    image_urls: List[str]

class GenerationProgressInfo(BaseModel):
    current_image: int = Field(0, description="현재 생성 중인 이미지 번호")
    total_images: int = Field(0, description="총 생성할 이미지 수")
    current_step: int = Field(0, description="현재 스텝")
    total_steps: int = Field(0, description="총 스텝 수")

class GenerationStatusResponse(BaseModel):
    status: str = Field(..., description="생성 상태: IDLE, GENERATING, SUCCESS, FAIL")
    progress: GenerationProgressInfo = Field(..., description="진행도 정보")
    message: str = Field(..., description="상태 메시지")

class VErrorLocation(BaseModel):
    loc: List[Union[str, int]]
    msg: str
    type: str

class ValidationErrorResponse(BaseModel):
    detail: List[VErrorLocation]


# --- 백그라운드 작업 함수 ---
def run_training_task(req: TrainRequest):
    """백그라운드에서 학습을 실행하는 함수"""
    global training_status

    # 콜백 함수 정의
    def update_status(status: str, phase: str = "", current_epoch: int = 0, total_epochs: int = 0, message: str = ""):
        """학습 상태 업데이트 콜백"""
        global training_status
        training_status["status"] = status
        training_status["progress"]["phase"] = phase
        training_status["progress"]["current_epoch"] = current_epoch
        training_status["progress"]["total_epochs"] = total_epochs
        training_status["message"] = message

    temp_dataset_path = None  # 임시 파일 경로 추적
    model_file_path = None
    s3_key = None

    try:
        # S3 URL이 제공된 경우 이미지 다운로드
        if req.training_image_urls:
            update_status(
                status="PREPROCESSING",
                phase="downloading",
                message="S3에서 학습 이미지 다운로드 중..."
            )
            # temp/dataset_xxxxx에 다운로드
            temp_dataset_path = download_training_images(req.training_image_urls)
            raw_dataset_path = temp_dataset_path
        else:
            raw_dataset_path = req.raw_dataset_path or "./dataset"

        # 출력 디렉토리 자동 생성 (models/{user_id}/{model_name})
        output_dir = req.output_dir or f"models/{req.user_id}/{req.model_name}"

        config = TrainingConfig(
            raw_dataset_path=raw_dataset_path,
            output_dir=output_dir
        )
        train_with_preprocessing(
            raw_dataset_path=raw_dataset_path,
            output_dir=output_dir,
            config=config,
            skip_preprocessing=req.skip_preprocessing,
            callback=update_status
        )

        # 학습 완료 후 S3에 모델 업로드
        update_status(
            status="SUCCESS",
            phase="uploading",
            message="S3에 모델 업로드 중..."
        )

        # 모델 파일 찾기 (pytorch_lora_weights.safetensors)
        model_file_path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")

        # S3 키 생성 (models/{user_id}/{model_name}.safetensors)
        s3_key = f"models/{req.user_id}/{req.model_name}.safetensors"

        # S3 업로드
        upload_to_s3(model_file_path, s3_key)

        # 파일 크기 계산
        file_size = os.path.getsize(model_file_path)

        update_status(
            status="SUCCESS",
            phase="",
            current_epoch=0,
            total_epochs=0,
            message="학습이 성공적으로 완료되었습니다."
        )

        # Spring Boot 콜백 호출
        if req.callback_url:
            try:
                callback_data = {
                    "userId": req.user_id,
                    "modelName": req.model_name,
                    "s3Key": s3_key,
                    "fileSize": file_size,
                    "status": "SUCCESS"
                }
                response = requests.post(req.callback_url, json=callback_data, timeout=10)
                response.raise_for_status()
                print(f"✅ Callback to Spring Boot successful: {req.callback_url}")
            except Exception as callback_error:
                print(f"❌ Callback to Spring Boot failed: {callback_error}")

    except Exception as e:
        update_status(
            status="FAIL",
            phase="",
            current_epoch=0,
            total_epochs=0,
            message=f"학습 실패: {str(e)}"
        )

        # 실패 시에도 콜백 호출
        if req.callback_url:
            try:
                callback_data = {
                    "userId": req.user_id,
                    "modelName": req.model_name,
                    "status": "FAIL",
                    "error": str(e)
                }
                requests.post(req.callback_url, json=callback_data, timeout=10)
            except:
                pass

    finally:
        # S3에서 다운로드한 임시 파일 정리
        if temp_dataset_path and os.path.exists(temp_dataset_path):
            try:
                shutil.rmtree(temp_dataset_path)
            except Exception as e:
                print(f"임시 파일 정리 실패: {e}")

def run_generation_task(req: GenerateRequest, base_url: str, user_id: str = None):
    """백그라운드에서 이미지 생성을 실행하는 함수"""
    global generation_status

    # 콜백 함수 정의
    def update_generation_status(status: str, current_image: int = 0, total_images: int = 0,
                                  current_step: int = 0, total_steps: int = 0, message: str = ""):
        """이미지 생성 상태 업데이트 콜백"""
        global generation_status
        generation_status["status"] = status
        generation_status["progress"]["current_image"] = current_image
        generation_status["progress"]["total_images"] = total_images
        generation_status["progress"]["current_step"] = current_step
        generation_status["progress"]["total_steps"] = total_steps
        generation_status["message"] = message

    temp_lora_path = None  # 임시 파일 경로 추적
    try:
        # S3 URL에서 LoRA 모델 다운로드
        if req.lora_model_url:
            update_generation_status(
                status="GENERATING",
                message="S3에서 LoRA 모델 다운로드 중..."
            )
            # temp/lora_xxxxx에 다운로드
            lora_id = os.urandom(8).hex()
            temp_lora_path = os.path.join("temp", f"lora_{lora_id}")
            os.makedirs(temp_lora_path, exist_ok=True)
            lora_model_path = os.path.join(temp_lora_path, "pytorch_lora_weights.safetensors")
            download_from_s3_url(req.lora_model_url, lora_model_path)
            lora_path = temp_lora_path
        else:
            lora_path = req.lora_path or "my_lora_model"

        config = InferenceConfig(
            lora_path=lora_path,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_images=req.num_images,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed
        )

        generated_files = generate_images(
            lora_path=config.lora_path,
            config=config,
            callback=update_generation_status
        )

        if not generated_files:
            update_generation_status(
                status="FAIL",
                message="이미지 생성 실패"
            )
        else:
            # S3에 생성된 이미지 업로드
            update_generation_status(
                status="SUCCESS",
                message="S3에 이미지 업로드 중..."
            )

            s3_keys = []
            for generated_file in generated_files:
                # S3 키 생성 (user-{user_id}/{filename}.png)
                filename = os.path.basename(generated_file)
                s3_key = f"user-{user_id}/{filename}" if user_id else f"generated/{filename}"

                # S3 업로드
                upload_to_s3(
                    local_file_path=generated_file,
                    s3_key=s3_key,
                    bucket_name="lora-generated-image-bucket"
                )

                s3_keys.append(s3_key)

            update_generation_status(
                status="SUCCESS",
                current_image=0,
                total_images=0,
                current_step=0,
                total_steps=0,
                message=f"이미지 생성 완료 ({len(s3_keys)}개)"
            )

            # S3 키 목록 저장 (Spring Boot에서 조회할 수 있도록)
            generation_status["s3_keys"] = s3_keys

    except Exception as e:
        update_generation_status(
            status="FAIL",
            current_image=0,
            total_images=0,
            current_step=0,
            total_steps=0,
            message=f"이미지 생성 실패: {str(e)}"
        )
    finally:
        # S3에서 다운로드한 임시 모델 파일 정리
        if temp_lora_path and os.path.exists(temp_lora_path):
            try:
                shutil.rmtree(temp_lora_path)
            except Exception as e:
                print(f"임시 파일 정리 실패: {e}")

# --- API 엔드포인트 ---
@app.get(
    "/",
    response_model=MessageResponse,
    summary="API 서버 상태 확인",
    responses={
        200: {
            "description": "서버가 정상적으로 실행 중일 때의 응답입니다.",
            "content": {
                "application/json": {
                    "example": {"message": "LoRA FastAPI server is running."}
                }
            },
        }
    },
)
def read_root():
    """API 서버가 정상적으로 실행 중인지 확인합니다."""
    return {"message": "LoRA FastAPI server is running."}

@app.post(
    "/train",
    response_model=MessageResponse,
    summary="LoRA 모델 학습 시작",
    responses={
        200: {
            "description": "학습이 성공적으로 시작되었을 때의 응답입니다.",
            "content": {
                "application/json": {
                    "example": {"message": "Training started in the background. Check /train/status for progress."}
                }
            },
        },
        400: {
            "description": "이미 학습이 진행 중일 때의 응답입니다.",
            "content": {
                "application/json": {
                    "example": {"message": "Training is already in progress."}
                }
            },
        },
    },
)
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    LoRA 모델 학습을 시작합니다.
    - 학습은 백그라운드에서 실행되며, 완료까지 시간이 소요될 수 있습니다.
    - 학습 진행 상태는 `/train/status` 엔드포인트로 확인할 수 있습니다.
    """
    with training_lock:
        if training_status["status"] in ["PREPROCESSING", "TRAINING"]:
            return JSONResponse(
                status_code=400,
                content={"message": "Training is already in progress."}
            )
        # 학습 시작 상태로 초기화
        training_status["status"] = "PREPROCESSING"
        training_status["progress"]["phase"] = "preprocessing"
        training_status["progress"]["current_epoch"] = 0
        training_status["progress"]["total_epochs"] = 0
        training_status["message"] = "학습 준비 중..."

    background_tasks.add_task(run_training_task, req)
    return {"message": "Training started in the background. Check /train/status for progress."}

@app.get(
    "/train/status",
    response_model=TrainStatusResponse,
    summary="학습 상태 확인",
    responses={
        200: {
            "description": "현재 학습 상태에 대한 응답입니다.",
            "content": {
                "application/json": {
                    "examples": {
                        "idle": {
                            "summary": "대기 중",
                            "value": {
                                "status": "IDLE",
                                "progress": {
                                    "phase": "",
                                    "current_epoch": 0,
                                    "total_epochs": 0
                                },
                                "message": "대기 중"
                            }
                        },
                        "preprocessing": {
                            "summary": "전처리 중",
                            "value": {
                                "status": "PREPROCESSING",
                                "progress": {
                                    "phase": "preprocessing",
                                    "current_epoch": 0,
                                    "total_epochs": 0
                                },
                                "message": "데이터셋 전처리 중..."
                            }
                        },
                        "training": {
                            "summary": "학습 진행 중",
                            "value": {
                                "status": "TRAINING",
                                "progress": {
                                    "phase": "training",
                                    "current_epoch": 50,
                                    "total_epochs": 250
                                },
                                "message": "학습 진행 중... (50/250 에포크 완료)"
                            }
                        },
                        "success": {
                            "summary": "학습 완료",
                            "value": {
                                "status": "SUCCESS",
                                "progress": {
                                    "phase": "",
                                    "current_epoch": 0,
                                    "total_epochs": 0
                                },
                                "message": "학습이 성공적으로 완료되었습니다."
                            }
                        },
                        "fail": {
                            "summary": "학습 실패",
                            "value": {
                                "status": "FAIL",
                                "progress": {
                                    "phase": "",
                                    "current_epoch": 0,
                                    "total_epochs": 0
                                },
                                "message": "학습 실패: Some error message"
                            }
                        }
                    }
                }
            },
        }
    },
)
def get_training_status():
    """현재 학습 진행 상태를 확인합니다."""
    return training_status

@app.get(
    "/train/stream",
    summary="학습 진행률 실시간 스트림 (SSE)",
    description="""
Server-Sent Events (SSE)를 사용하여 학습 진행률을 실시간으로 스트리밍합니다.
- 폴링 없이 서버가 자동으로 상태 업데이트를 푸시합니다.
- 학습이 완료되거나 실패하면 자동으로 연결이 종료됩니다.
- EventSource API를 사용하여 연결하세요.
    """
)
async def stream_training_status():
    """학습 진행 상태를 SSE로 스트리밍합니다."""
    async def event_generator():
        previous_status = None
        while True:
            # 상태가 변경되었을 때만 전송
            current_status = training_status.copy()
            if current_status != previous_status:
                yield f"data: {json.dumps(current_status)}\n\n"
                previous_status = current_status.copy()

            # 완료 또는 실패 시 스트림 종료
            if training_status["status"] in ["SUCCESS", "FAIL"]:
                break

            await asyncio.sleep(0.5)  # 0.5초마다 체크

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.post(
    "/generate",
    response_model=MessageResponse,
    summary="이미지 생성 시작",
    responses={
        200: {
            "description": "이미지 생성이 성공적으로 시작되었을 때의 응답입니다.",
            "content": {
                "application/json": {
                    "example": {"message": "Image generation started in the background. Check /generate/status for progress."}
                }
            },
        },
        400: {
            "description": "이미 이미지 생성이 진행 중일 때의 응답입니다.",
            "content": {
                "application/json": {
                    "example": {"message": "Image generation is already in progress."}
                }
            },
        },
        404: {
            "description": "LoRA 모델을 찾을 수 없을 때의 응답입니다.",
            "content": {
                "application/json": {
                    "example": {"message": "LoRA model not found at my_lora_model. Please train the model first."}
                }
            },
        },
    },
)
async def generate_image_api(request: Request, req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    프롬프트를 기반으로 이미지 생성을 시작합니다.

    - 이미지 생성은 백그라운드에서 실행되며, 완료까지 시간이 소요될 수 있습니다.
    - 생성 진행 상태는 `/generate/status` 엔드포인트로 확인할 수 있습니다.
    - 생성된 이미지는 S3에 업로드됩니다.
    """
    # S3 URL이나 로컬 경로 중 하나는 필수
    if not req.lora_model_url and not req.lora_path:
        return JSONResponse(
            status_code=400,
            content={"message": "Either lora_model_url or lora_path must be provided."}
        )

    # 로컬 경로를 사용하는 경우에만 파일 존재 여부 확인
    if req.lora_path and not req.lora_model_url and not os.path.exists(req.lora_path):
        return JSONResponse(
            status_code=404,
            content={"message": f"LoRA model not found at {req.lora_path}. Please train the model first."}
        )

    with generation_lock:
        if generation_status["status"] == "GENERATING":
            return JSONResponse(
                status_code=400,
                content={"message": "Image generation is already in progress."}
            )
        # 생성 시작 상태로 초기화
        generation_status["status"] = "GENERATING"
        generation_status["progress"]["current_image"] = 0
        generation_status["progress"]["total_images"] = req.num_images
        generation_status["progress"]["current_step"] = 0
        generation_status["progress"]["total_steps"] = req.steps
        generation_status["message"] = "이미지 생성 준비 중..."
        generation_status["s3_keys"] = []

    base_url = str(request.base_url)
    # user_id를 GenerateRequest에서 추출 (없으면 None)
    user_id = getattr(req, 'user_id', None)
    background_tasks.add_task(run_generation_task, req, base_url, user_id)
    return {"message": "Image generation started in the background. Check /generate/status for progress."}

@app.get(
    "/generate/status",
    response_model=GenerationStatusResponse,
    summary="이미지 생성 상태 확인",
    responses={
        200: {
            "description": "현재 이미지 생성 상태에 대한 응답입니다.",
            "content": {
                "application/json": {
                    "examples": {
                        "idle": {
                            "summary": "대기 중",
                            "value": {
                                "status": "IDLE",
                                "progress": {
                                    "current_image": 0,
                                    "total_images": 0,
                                    "current_step": 0,
                                    "total_steps": 0
                                },
                                "message": "대기 중"
                            }
                        },
                        "generating": {
                            "summary": "생성 진행 중",
                            "value": {
                                "status": "GENERATING",
                                "progress": {
                                    "current_image": 1,
                                    "total_images": 3,
                                    "current_step": 20,
                                    "total_steps": 40
                                },
                                "message": "이미지 1/3 생성 중... (step 20/40)"
                            }
                        },
                        "success": {
                            "summary": "생성 완료",
                            "value": {
                                "status": "SUCCESS",
                                "progress": {
                                    "current_image": 0,
                                    "total_images": 0,
                                    "current_step": 0,
                                    "total_steps": 0
                                },
                                "message": "이미지 생성 완료 (3개)"
                            }
                        },
                        "fail": {
                            "summary": "생성 실패",
                            "value": {
                                "status": "FAIL",
                                "progress": {
                                    "current_image": 0,
                                    "total_images": 0,
                                    "current_step": 0,
                                    "total_steps": 0
                                },
                                "message": "이미지 생성 실패: Some error message"
                            }
                        }
                    }
                }
            },
        }
    },
)
def get_generation_status():
    """현재 이미지 생성 진행 상태를 확인합니다."""
    return generation_status

@app.get(
    "/generate/stream",
    summary="이미지 생성 진행률 실시간 스트림 (SSE)",
    description="""
Server-Sent Events (SSE)를 사용하여 이미지 생성 진행률을 실시간으로 스트리밍합니다.
- 폴링 없이 서버가 자동으로 상태 업데이트를 푸시합니다.
- 이미지 생성이 완료되거나 실패하면 자동으로 연결이 종료됩니다.
- EventSource API를 사용하여 연결하세요.
    """
)
async def stream_generation_status():
    """이미지 생성 진행 상태를 SSE로 스트리밍합니다."""
    async def event_generator():
        previous_status = None
        while True:
            # 상태가 변경되었을 때만 전송
            current_status = generation_status.copy()
            if current_status != previous_status:
                yield f"data: {json.dumps(current_status)}\n\n"
                previous_status = current_status.copy()

            # 완료 또는 실패 시 스트림 종료
            if generation_status["status"] in ["SUCCESS", "FAIL"]:
                break

            await asyncio.sleep(0.3)  # 0.3초마다 체크 (이미지 생성이 더 빠름)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.get(
    "/generate/images",
    summary="생성된 이미지 S3 키 목록 조회",
    description="""
생성 완료 후 Spring Boot에서 호출하여 S3 키 목록을 가져갑니다.
- 생성 완료 후에만 호출해야 합니다.
- S3 키 목록을 반환합니다 (예: ["user-123/20250118_143025_1.png", ...])
    """
)
def get_generated_image_keys():
    """생성된 이미지의 S3 키 목록을 반환합니다."""
    if generation_status["status"] != "SUCCESS":
        return JSONResponse(
            status_code=400,
            content={
                "message": "Image generation is not completed yet.",
                "status": generation_status["status"]
            }
        )

    s3_keys = generation_status.get("s3_keys", [])
    return {
        "s3_keys": s3_keys,
        "count": len(s3_keys)
    }

if __name__ == "__main__":
    import uvicorn
    print("FastAPI 서버를 시작합니다. 주소: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)