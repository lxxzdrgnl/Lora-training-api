"""
FastAPI 애플리케이션
LoRA 학습 및 이미지 생성을 위한 API
"""

import os
import json
import asyncio
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
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:5173",  # Vite 기본 포트
        "http://127.0.0.1:5173",  # localhost 대신 127.0.0.1
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# --- 정적 파일 마운트 ---
# 'outputs' 디렉토리를 '/static' 경로에 마운트하여 이미지 URL로 접근 가능하게 합니다.
os.makedirs("outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="outputs"), name="static")

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
    raw_dataset_path: str = Field("./dataset", description="원본 데이터셋 경로")
    output_dir: str = Field("my_lora_model", description="학습된 모델이 저장될 경로")
    skip_preprocessing: bool = Field(False, description="전처리 과정 스킵 여부")

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="이미지 생성을 위한 프롬프트")
    negative_prompt: Optional[str] = Field("low quality, blurry, ugly, distorted, deformed", description="이미지 생성 시 제외할 요소들에 대한 프롬프트")
    lora_path: str = Field("my_lora_model", description="사용할 LoRA 모델 경로")
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

    try:
        config = TrainingConfig(
            raw_dataset_path=req.raw_dataset_path,
            output_dir=req.output_dir
        )
        train_with_preprocessing(
            raw_dataset_path=req.raw_dataset_path,
            output_dir=req.output_dir,
            config=config,
            skip_preprocessing=req.skip_preprocessing,
            callback=update_status
        )
        update_status(
            status="SUCCESS",
            phase="",
            current_epoch=0,
            total_epochs=0,
            message="학습이 성공적으로 완료되었습니다."
        )
    except Exception as e:
        update_status(
            status="FAIL",
            phase="",
            current_epoch=0,
            total_epochs=0,
            message=f"학습 실패: {str(e)}"
        )

def run_generation_task(req: GenerateRequest, base_url: str):
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

    try:
        config = InferenceConfig(
            lora_path=req.lora_path,
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
            image_urls = [
                f"{base_url}static/{os.path.basename(path)}"
                for path in generated_files
            ]
            update_generation_status(
                status="SUCCESS",
                current_image=0,
                total_images=0,
                current_step=0,
                total_steps=0,
                message=f"이미지 생성 완료 ({len(image_urls)}개)"
            )
            # 생성된 이미지 URL 저장
            generation_status["image_urls"] = image_urls

    except Exception as e:
        update_generation_status(
            status="FAIL",
            current_image=0,
            total_images=0,
            current_step=0,
            total_steps=0,
            message=f"이미지 생성 실패: {str(e)}"
        )

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
    - 생성된 이미지는 `outputs/` 폴더에 저장됩니다.
    """
    if not os.path.exists(req.lora_path):
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
        generation_status["image_urls"] = []

    base_url = str(request.base_url)
    background_tasks.add_task(run_generation_task, req, base_url)
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

if __name__ == "__main__":
    import uvicorn
    print("FastAPI 서버를 시작합니다. 주소: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)