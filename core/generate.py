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


def load_pipeline(model_id: str, lora_path: str, device: str):
    """
    Stable Diffusion 파이프라인 + LoRA 로드

    Args:
        model_id: 베이스 모델 ID
        lora_path: LoRA 모델 경로
        device: 디바이스 (cuda/cpu)

    Returns:
        StableDiffusionPipeline: 로드된 파이프라인
    """
    print(f"\nLoading base model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    print(f"Loading LoRA weights: {lora_path}")
    pipe.unet = PeftModel.from_pretrained(
        pipe.unet,
        lora_path,
        torch_dtype=torch.float16
    )

    pipe.to(device)
    pipe.unet.eval()

    print("✅ Model loaded successfully!\n")
    return pipe


def generate_images(
    lora_path: str,
    prompt: str = None,
    negative_prompt: str = None,
    num_images: int = 1,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = None,
    output_dir: str = "outputs",
    config: InferenceConfig = None
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

    Returns:
        list: 생성된 이미지 경로 리스트
    """
    if config is None:
        config = InferenceConfig()

    # 설정 오버라이드
    if prompt:
        config.prompt = prompt
    if negative_prompt:
        config.negative_prompt = negative_prompt
    if lora_path:
        config.lora_path = lora_path

    config.num_images = num_images
    config.steps = steps
    config.guidance_scale = guidance_scale
    config.seed = seed
    config.output_dir = output_dir

    # 파이프라인 로드
    pipe = load_pipeline(config.model_id, config.lora_path, config.device)

    # Trigger word 자동 추가
    if not config.prompt.startswith(config.trigger_word):
        full_prompt = f"{config.trigger_word}, {config.prompt}"
    else:
        full_prompt = config.prompt

    print("="*60)
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
    for i in range(config.num_images):
        print(f"\n[{i+1}/{config.num_images}] Generating...")

        with torch.no_grad():
            image = pipe(
                prompt=full_prompt,
                negative_prompt=config.negative_prompt,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance_scale,
                generator=generator
            ).images[0]

        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{i+1}.png"
        output_path = output_dir / filename

        # 저장
        image.save(output_path)
        generated_files.append(str(output_path))
        print(f"✅ Saved: {output_path}")

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
        default="my_lora_model",
        help="LoRA 모델 경로"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="xyn-ai/anything-v4.0",
        help="베이스 Stable Diffusion 모델"
    )

    # 프롬프트 설정
    parser.add_argument(
        "--prompt",
        type=str,
        default="black hair, long hair, black and white manga style, monochrome illustration",
        help="이미지 생성 프롬프트"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="color, colorful, low quality, blurry, ugly, distorted",
        help="네거티브 프롬프트"
    )

    # 생성 옵션
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="생성할 이미지 개수"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Inference steps (20-50 권장)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="CFG scale (7-10 권장)"
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
        default="outputs",
        help="생성된 이미지 저장 폴더"
    )

    args = parser.parse_args()

    # LoRA 모델 존재 확인
    if not os.path.exists(args.lora_path):
        print(f"❌ Error: LoRA model not found at {args.lora_path}")
        print(f"Please train the model first: python training.py")
        return

    # 이미지 생성
    config = InferenceConfig(
        model_id=args.model_id,
        lora_path=args.lora_path,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_images=args.num_images,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_dir=args.output_dir
    )

    generate_images(
        lora_path=config.lora_path,
        config=config
    )


if __name__ == "__main__":
    main()
