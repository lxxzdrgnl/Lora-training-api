"""
LoRA 모델 추론 스크립트
학습된 LoRA 가중치로 이미지 생성
"""

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import argparse
from datetime import datetime
import os
from pathlib import Path


def load_pipeline(model_id, lora_path, device):
    """Stable Diffusion 파이프라인 + LoRA 로드"""
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


def generate_images(pipe, args, device):
    """이미지 생성"""
    # Trigger word 자동 추가
    if not args.prompt.startswith("sks"):
        full_prompt = f"sks girl, {args.prompt}"
    else:
        full_prompt = args.prompt

    print("="*60)
    print(f"Generating {args.num_images} image(s)")
    print(f"Prompt: {full_prompt}")
    print(f"Negative: {args.negative_prompt}")
    print(f"Steps: {args.steps} | CFG Scale: {args.guidance_scale}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print("="*60)

    # 시드 설정
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # 출력 폴더 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 이미지 생성
    generated_files = []
    for i in range(args.num_images):
        print(f"\n[{i+1}/{args.num_images}] Generating...")

        with torch.no_grad():
            image = pipe(
                prompt=full_prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator
            ).images[0]

        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{i+1}.png"
        output_path = output_dir / filename

        # 저장
        image.save(output_path)
        generated_files.append(output_path)
        print(f"✅ Saved: {output_path}")

    return generated_files


def main():
    # 환경 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 인자 파싱
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
        print(f"Please train the model first: python train.py")
        return

    # 파이프라인 로드
    pipe = load_pipeline(args.model_id, args.lora_path, device)

    # 이미지 생성
    generated_files = generate_images(pipe, args, device)

    # 완료 메시지
    print("\n" + "="*60)
    print(f"✅ Successfully generated {len(generated_files)} image(s)")
    print(f"📁 Output folder: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
