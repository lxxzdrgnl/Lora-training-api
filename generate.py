"""
이미지 생성 실행 스크립트
학습된 LoRA 모델로 이미지 생성
"""

import argparse
import os
from core.config import InferenceConfig
from core.generate import generate_images


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
        default="sks, portrait, high quality",
        help="이미지 생성 프롬프트 (sks 필수!)"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, blurry, ugly, distorted, deformed",
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
        print(f"Please train the model first: python run_train.py")
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
