"""
베이스 모델 다운로드 스크립트 (최초 1회만 실행)
Modal Volume에 Stable Diffusion 베이스 모델을 다운로드합니다.

사용법:
    modal run download_models.py
"""

import modal

# Modal 앱 정의
app = modal.App("download-base-models")

# 공유 볼륨 생성 (모든 인스턴스가 공유)
volume = modal.Volume.from_name("lora-base-models", create_if_missing=True)

# GPU 이미지 정의
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
    )
)


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=3600,  # 1시간
)
def download_base_models():
    """
    베이스 모델 다운로드
    - 여러 Stable Diffusion 모델을 다운로드
    - 이미지 생성에 사용되는 기본 모델들
    """
    from diffusers import StableDiffusionPipeline
    import torch

    print("📥 베이스 모델 다운로드 시작...")

    # 다운로드할 모델 리스트
    models = [
        {
            "id": "stablediffusionapi/anything-v5",
            "name": "anything-v5"
        },
        {
            "id": "Lykon/AnyLoRA",
            "name": "anylora"
        }
    ]

    results = []

    for model_info in models:
        model_id = model_info["id"]
        model_name = model_info["name"]
        model_path = f"/models/{model_name}"

        print(f"\n{'='*60}")
        print(f"📦 모델 ID: {model_id}")
        print(f"💾 저장 경로: {model_path}")

        try:
            # 모델 다운로드 및 저장
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )

            # 볼륨에 저장
            pipeline.save_pretrained(model_path)

            print(f"✅ {model_name} 다운로드 완료!")

            results.append({
                "status": "SUCCESS",
                "model_id": model_id,
                "model_name": model_name,
                "model_path": model_path
            })

        except Exception as e:
            print(f"❌ {model_name} 다운로드 실패: {e}")
            results.append({
                "status": "FAIL",
                "model_id": model_id,
                "model_name": model_name,
                "error": str(e)
            })

    # 볼륨 커밋 (변경사항 저장)
    volume.commit()

    print(f"\n{'='*60}")
    print(f"✅ 모든 모델 다운로드 작업 완료!")
    print(f"성공: {sum(1 for r in results if r['status'] == 'SUCCESS')}/{len(results)}")

    return results


@app.function(
    image=image,
    volumes={"/models": volume},
)
def list_models():
    """볼륨에 저장된 모델 목록 확인"""
    import os

    print("📂 저장된 모델 목록:")
    if os.path.exists("/models"):
        for item in os.listdir("/models"):
            item_path = os.path.join("/models", item)
            if os.path.isdir(item_path):
                size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(item_path)
                    for filename in filenames
                )
                print(f"  - {item} ({size / 1024 / 1024 / 1024:.2f} GB)")
            else:
                size = os.path.getsize(item_path)
                print(f"  - {item} ({size / 1024 / 1024:.2f} MB)")
    else:
        print("  (비어 있음)")


@app.local_entrypoint()
def main():
    """로컬에서 실행"""
    print("=" * 60)
    print("🚀 베이스 모델 다운로드 시작")
    print("=" * 60)

    # 모델 다운로드
    result = download_base_models.remote()
    print(f"\n결과: {result}")

    # 모델 목록 확인
    print("\n" + "=" * 60)
    print("📋 현재 저장된 모델")
    print("=" * 60)
    list_models.remote()

    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)
    print("\n💡 이제 FastAPI 앱에서 /models 경로로 모델에 접근할 수 있습니다.")
