"""
LoRA 모델 추론 모듈
"""

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import argparse
from datetime import datetime
import os
from pathlib import Path

from .config import InferenceConfig


# 파이프라인 캐시 클래스 (싱글톤 패턴)
class PipelineCache:
    """파이프라인을 메모리에 캐싱하여 재사용"""
    _instance = None
    _pipelines = {}  # {(model_id, lora_path): pipeline}

    @classmethod
    def get_pipeline(cls, model_id: str, lora_path: str, device: str):
        """캐시된 파이프라인 반환 또는 새로 로드"""
        cache_key = (model_id, lora_path)

        if cache_key in cls._pipelines:
            print(f"🚀 Using cached pipeline (model: {model_id}, lora: {lora_path})")
            return cls._pipelines[cache_key]

        print(f"📦 Loading new pipeline into cache...")
        pipe = load_pipeline(model_id, lora_path, device)
        cls._pipelines[cache_key] = pipe
        return pipe

    @classmethod
    def clear_cache(cls):
        """캐시 초기화 (메모리 정리용)"""
        cls._pipelines.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 Pipeline cache cleared")


def load_pipeline(model_id: str, lora_path: str, device: str):
    """
    Stable Diffusion 파이프라인 + LoRA 로드

    Args:
        model_id: 베이스 모델 ID 또는 로컬 경로
        lora_path: LoRA 모델 경로 (디렉토리 또는 .safetensors 파일)
        device: 디바이스 (cuda/cpu)

    Returns:
        StableDiffusionPipeline: 로드된 파이프라인
    """
    # Modal Volume 캐싱 로직
    # /cache/base_models/{model_id}/ 경로에 캐싱
    cache_base = "/cache/base_models"

    # 모델 ID를 파일시스템에 안전한 이름으로 변환 (/ -> --)
    safe_model_id = model_id.replace("/", "--")
    cached_model_path = os.path.join(cache_base, safe_model_id)

    # 캐시된 모델 확인
    if os.path.exists(cached_model_path) and os.path.exists(os.path.join(cached_model_path, "model_index.json")):
        base_model_path = cached_model_path
        print(f"\n✅ Using cached base model from volume: {base_model_path}")
    else:
        # 캐시가 없으면 HuggingFace에서 다운로드 후 캐싱
        print(f"\n📥 Downloading base model from HuggingFace: {model_id}")
        print(f"💾 Will cache to: {cached_model_path}")

        # 임시로 HuggingFace에서 로드
        temp_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )

        # Modal Volume에 저장
        os.makedirs(cache_base, exist_ok=True)
        temp_pipe.save_pretrained(cached_model_path)
        print(f"✅ Base model cached successfully!")

        base_model_path = cached_model_path

    print(f"Loading base model: {base_model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    print(f"Loading LoRA weights: {lora_path}")

    # LoRA 경로 확인 및 형식 판단
    if os.path.isdir(lora_path):
        # 디렉토리인 경우: PEFT 형식 우선 시도 (adapter_model.safetensors + adapter_config.json)
        adapter_config_path = os.path.join(lora_path, "adapter_config.json")
        adapter_model_path = os.path.join(lora_path, "adapter_model.safetensors")

        if os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path):
            # PEFT 형식 발견 - 이것이 가장 정확함 (alpha, rank 정보 포함)
            print(f"✅ Found PEFT format (adapter_model.safetensors + adapter_config.json)")
            print(f"   This preserves lora_alpha and lora_r from training")
            pipe.load_lora_weights(
                lora_path,  # 디렉토리 전달
                weight_name="adapter_model.safetensors"
            )
        else:
            # PEFT 형식이 없으면 WebUI 형식 찾기
            safetensors_files = [f for f in os.listdir(lora_path) if f.endswith('.safetensors')]
            if not safetensors_files:
                raise ValueError(f"No .safetensors file found in {lora_path}")
            lora_file = safetensors_files[0]
            print(f"Found WebUI format: {lora_file}")
            pipe.load_lora_weights(
                lora_path,
                weight_name=lora_file
            )
    elif os.path.isfile(lora_path) and lora_path.endswith('.safetensors'):
        # 단일 safetensors 파일 (Civitai 다운로드 형식)
        print(f"Found single LoRA file (WebUI format): {lora_path}")
        pipe.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name=os.path.basename(lora_path)
        )
    else:
        raise ValueError(f"Invalid LoRA path: {lora_path}")

    # GPU로 이동
    pipe.to(device)

    # xFormers 최적화 활성화
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xFormers memory efficient attention enabled")
    except Exception as e:
        print(f"⚠️ xFormers not available: {e}")
        print("   Falling back to default attention (still works, just slower)")

    # Attention slicing 활성화 (메모리 효율성)
    try:
        pipe.enable_attention_slicing(slice_size="auto")
        print("✅ Attention slicing enabled")
    except Exception as e:
        print(f"⚠️ Could not enable attention slicing: {e}")

    print("✅ LoRA loaded successfully in WebUI/Civitai format!\n")
    return pipe


def generate_images(
    lora_path: str = None,
    prompt: str = None,
    negative_prompt: str = None,
    num_images: int = None,
    steps: int = None,
    guidance_scale: float = None,
    seed: int = None,
    output_dir: str = None,
    config: InferenceConfig = None,
    callback = None
):
    """
    이미지 생성 함수 (Modal API용)

    Args:
        lora_path: LoRA 모델 경로
        prompt: 프롬프트
        negative_prompt: 네거티브 프롬프트
        num_images: 생성할 이미지 수
        steps: 추론 스텝
        guidance_scale: CFG scale
        seed: 랜덤 시드
        output_dir: 출력 폴더
        config: 추론 설정 (None이면 기본값 사용)
        callback: 진행도 업데이트 콜백 함수 (status, current_image, total_images, current_step, total_steps, message)

    Returns:
        list: 생성된 이미지 경로 리스트
    """
    if config is None:
        config = InferenceConfig()

    # 설정 오버라이드 (None이 아닐 때만)
    if prompt is not None:
        config.prompt = prompt
    if negative_prompt is not None:
        config.negative_prompt = negative_prompt
    if lora_path is not None:
        config.lora_path = lora_path
    if num_images is not None:
        config.num_images = num_images
    if steps is not None:
        config.steps = steps
    if guidance_scale is not None:
        config.guidance_scale = guidance_scale
    if seed is not None:
        config.seed = seed
    if output_dir is not None:
        config.output_dir = output_dir

    # 파이프라인 로드 (캐시 사용)
    pipe = PipelineCache.get_pipeline(config.model_id, config.lora_path, config.device)

    # Trigger word 자동 추가 (사용자 요청으로 제거됨)
    # if not config.prompt.startswith(config.trigger_word):
    #     full_prompt = f"{config.trigger_word}, {config.prompt}"
    # else:
    #     full_prompt = config.prompt
    full_prompt = config.prompt

    print("="*60)
    print(f"DEBUG: config.num_images = {config.num_images}")
    print(f"Generating {config.num_images} image(s)")
    print(f"Prompt: {full_prompt}")
    print(f"Negative: {config.negative_prompt}")
    print(f"Steps: {config.steps} | CFG Scale: {config.guidance_scale}")
    if config.seed is not None:
        print(f"Seed: {config.seed}")
    print("="*60)

    # 시드 설정
    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=config.device).manual_seed(config.seed)

    # 출력 폴더 생성
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 이미지 생성
    generated_files = []

    # 생성 시작 콜백
    if callback:
        callback(
            status="GENERATING",
            current_image=0,
            total_images=config.num_images,
            current_step=0,
            total_steps=config.steps,
            message=f"이미지 생성 시작... (0/{config.num_images})"
        )

    for i in range(config.num_images):
        print(f"\n[{i+1}/{config.num_images}] Generating...")

        # Step별 콜백 함수 정의
        def step_callback(pipe_instance, step_index, timestep, callback_kwargs):
            if callback:
                callback(
                    status="GENERATING",
                    current_image=i + 1,
                    total_images=config.num_images,
                    current_step=step_index + 1,
                    total_steps=config.steps,
                    message=f"이미지 {i+1}/{config.num_images} 생성 중... (step {step_index+1}/{config.steps})"
                )
            return callback_kwargs

        with torch.no_grad():
            image = pipe(
                prompt=full_prompt,
                negative_prompt=config.negative_prompt,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
                callback_on_step_end=step_callback if callback else None,
                callback_on_step_end_tensor_inputs=["latents"],
                cross_attention_kwargs={"scale": config.lora_scale}  # LoRA 강도 적용
            ).images[0]

        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{i+1}.png"
        output_path = output_dir / filename

        # 저장
        image.save(output_path)
        generated_files.append(str(output_path))
        print(f"✅ Saved: {output_path}")

        # 이미지 완료 콜백
        if callback:
            callback(
                status="GENERATING",
                current_image=i + 1,
                total_images=config.num_images,
                current_step=config.steps,
                total_steps=config.steps,
                message=f"이미지 {i+1}/{config.num_images} 완료"
            )

    print("\n" + "="*60)
    print(f"✅ Successfully generated {len(generated_files)} image(s)")
    print(f"📁 Output folder: {config.output_dir}")
    print("="*60)

    return generated_files


def main():
    """CLI 실행"""
    parser = argparse.ArgumentParser(
        description="학습된 LoRA 모델로 이미지 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 필수 설정
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA 모델 경로 (기본값: config 설정)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="베이스 Stable Diffusion 모델 (기본값: config 설정)"
    )

    # 프롬프트 설정
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="이미지 생성 프롬프트 (기본값: config 설정)"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="네거티브 프롬프트 (기본값: config 설정)"
    )

    # 생성 옵션
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="생성할 이미지 개수 (기본값: config 설정)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Inference steps (기본값: config 설정)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="CFG scale (기본값: config 설정)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="랜덤 시드 (재현성)"
    )

    # 출력 설정
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="생성된 이미지 저장 폴더 (기본값: config 설정)"
    )

    args = parser.parse_args()

    # 이미지 생성 (config 기본값 사용 후 CLI 인자로 오버라이드)
    config = InferenceConfig()

    # CLI 인자가 지정된 경우만 오버라이드
    if args.model_id is not None:
        config.model_id = args.model_id
    if args.lora_path is not None:
        config.lora_path = args.lora_path

    if args.prompt is not None:
        config.prompt = args.prompt
    if args.negative_prompt is not None:
        config.negative_prompt = args.negative_prompt
    if args.num_images is not None:
        config.num_images = args.num_images
    if args.steps is not None:
        config.steps = args.steps
    if args.guidance_scale is not None:
        config.guidance_scale = args.guidance_scale
    if args.seed is not None:
        config.seed = args.seed
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    # LoRA 모델 존재 확인
    if not os.path.exists(config.lora_path):
        print(f"❌ Error: LoRA model not found at {config.lora_path}")
        print(f"Please train the model first: python training.py")
        return

    generate_images(
        lora_path=config.lora_path,
        config=config
    )


if __name__ == "__main__":
    main()
