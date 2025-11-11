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

    # config 기본값 사용 후 CLI 인자로 오버라이드
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
        print(f"Please train the model first: python train.py")
        return

    generate_images(
        lora_path=config.lora_path,
        config=config
    )


if __name__ == "__main__":
    main()
