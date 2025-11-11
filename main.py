"""
FastAPI 애플리케이션
LoRA 학습 및 이미지 생성을 위한 API
"""

import os
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Union

from core.config import TrainingConfig, InferenceConfig
from core.train import train_with_preprocessing
from core.generate import generate_images

app = FastAPI(
    title="LoRA Training and Inference API",
    description="LoRA 모델 학습 및 이미지 생성을 위한 API 서버입니다.",
    version="1.0.0",
)

# --- 정적 파일 마운트 ---
# 'outputs' 디렉토리를 '/static' 경로에 마운트하여 이미지 URL로 접근 가능하게 합니다.
app.mount("/static", StaticFiles(directory="outputs"), name="static")

# --- 모델 및 상태 관리 ---
training_status = {"is_training": False, "message": "Not training"}

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

class TrainStatusResponse(BaseModel):
    is_training: bool
    message: str

class GenerateResponse(BaseModel):
    image_url: str

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
    training_status = {"is_training": True, "message": "Training in progress..."}
    try:
        config = TrainingConfig(
            raw_dataset_path=req.raw_dataset_path,
            output_dir=req.output_dir
        )
        train_with_preprocessing(
            raw_dataset_path=req.raw_dataset_path,
            output_dir=req.output_dir,
            config=config,
            skip_preprocessing=req.skip_preprocessing
        )
        training_status = {"is_training": False, "message": "Training completed successfully."}
    except Exception as e:
        training_status = {"is_training": False, "message": f"Training failed: {str(e)}"}

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
    if training_status["is_training"]:
        return JSONResponse(
            status_code=400,
            content={"message": "Training is already in progress."}
        )
    
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
                        "training_in_progress": {
                            "summary": "학습 진행 중",
                            "value": {"is_training": True, "message": "Training in progress..."}
                        },
                        "training_completed": {
                            "summary": "학습 완료",
                            "value": {"is_training": False, "message": "Training completed successfully."}
                        },
                        "training_failed": {
                            "summary": "학습 실패",
                            "value": {"is_training": False, "message": "Training failed: Some error message."}
                        },
                        "not_training": {
                            "summary": "학습 중 아님",
                            "value": {"is_training": False, "message": "Not training"}
                        },
                    }
                }
            },
        }
    },
)
def get_training_status():
    """현재 학습 진행 상태를 확인합니다."""
    return training_status

@app.post(
    "/generate",
    response_model=GenerateResponse,
    summary="이미지 생성",
    responses={
        200: {
            "description": "성공적으로 이미지가 생성되었을 때의 응답입니다.",
            "content": {
                "application/json": {
                    "example": {"image_url": "http://127.0.0.1:8000/static/20251111_123456_1.png"}
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
            "model": MessageResponse
        },
        422: {
            "description": "요청 본문 유효성 검사 실패 시의 응답입니다.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "prompt"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            },
            "model": ValidationErrorResponse
        },
        500: {
            "description": "이미지 생성 중 오류가 발생했을 때의 응답입니다.",
            "content": {
                "application/json": {
                    "example": {"message": "An error occurred during image generation: Some internal server error."}
                }
            },
            "model": MessageResponse
        },
    },
)
async def generate_image_api(request: Request, req: GenerateRequest):
    """
    프롬프트를 기반으로 이미지를 생성하고, 생성된 이미지의 URL을 반환합니다.
    - `lora_path`에 지정된 모델이 존재해야 합니다.
    """
    if not os.path.exists(req.lora_path):
        return JSONResponse(
            status_code=404,
            content={"message": f"LoRA model not found at {req.lora_path}. Please train the model first."}
        )

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
            config=config
        )

        if not generated_files:
            return JSONResponse(
                status_code=500,
                content={"message": "Image generation failed."}
            )
        
        image_path = generated_files[0]
        image_filename = os.path.basename(image_path)
        image_url = f"{request.base_url}static/{image_filename}"
        
        return {"image_url": image_url}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred during image generation: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    print("FastAPI 서버를 시작합니다. 주소: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)