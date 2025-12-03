"""
VAE Latents 사전 계산 모듈
학습 전에 모든 이미지를 VAE로 인코딩하여 디스크에 저장
"""

import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os


def precompute_latents(
    dataset_path: str,
    output_path: str,
    model_id: str = "stablediffusionapi/anything-v5",
    device: str = "cuda",
    image_size: int = 512
):
    """
    데이터셋의 모든 이미지를 VAE로 인코딩해서 latents로 저장

    Args:
        dataset_path: 전처리된 이미지 폴더 (clean dataset)
        output_path: latents 저장 폴더
        model_id: 베이스 모델 (학습에 사용할 것과 동일해야 함)
        device: cuda/cpu
        image_size: 이미지 크기 (기본 512)

    Returns:
        dict: 처리 결과 정보
    """
    print(f"\n{'='*60}")
    print("VAE Latents Precomputation")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Model: {model_id}")
    print(f"Device: {device}")

    # VAE 로드 (학습에 사용할 것과 동일한 VAE)
    print(f"\n📦 Loading VAE from {model_id}...")
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    print("✅ VAE loaded")

    # 이미지 파일들 찾기
    dataset_dir = Path(dataset_path)
    image_files = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg"))

    if len(image_files) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    print(f"\n📂 Found {len(image_files)} images")

    # 출력 디렉토리 생성
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Latents 계산 및 저장
    print(f"\n🔄 Computing and saving latents...")

    success_count = 0
    total_size = 0

    for img_file in tqdm(image_files, desc="Precomputing latents"):
        try:
            # 이미지 로드 및 전처리
            img = Image.open(img_file).convert("RGB")
            img = img.resize((image_size, image_size), Image.LANCZOS)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = (img_array - 0.5) / 0.5  # normalize to [-1, 1]
            pixel_values = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            pixel_values = pixel_values.to(device, dtype=torch.float16)

            # VAE encoding
            with torch.no_grad():
                latent = vae.encode(pixel_values).latent_dist.sample()
                latent = latent * vae.config.scaling_factor

            # 저장 (.pt 파일로)
            latent_file = output_dir / f"{img_file.stem}_latent.pt"
            torch.save(latent.cpu(), latent_file)

            # 통계
            file_size = os.path.getsize(latent_file)
            total_size += file_size
            success_count += 1

        except Exception as e:
            print(f"\n⚠️ Failed to process {img_file.name}: {e}")
            continue

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"✅ Latents precomputation completed!")
    print(f"   Processed: {success_count}/{len(image_files)} images")
    print(f"   Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"   Avg size per latent: {total_size / success_count / 1024:.1f} KB")
    print(f"   Saved to: {output_path}")
    print(f"{'='*60}\n")

    return {
        "total": len(image_files),
        "success": success_count,
        "total_size_mb": total_size / 1024 / 1024,
        "output_dir": str(output_dir)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Precompute VAE latents for LoRA training")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to clean dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save latents")
    parser.add_argument("--model_id", type=str, default="stablediffusionapi/anything-v5")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    precompute_latents(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        model_id=args.model_id,
        device=args.device
    )
