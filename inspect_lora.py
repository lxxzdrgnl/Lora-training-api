"""
LoRA 파일 검사 스크립트
safetensors 파일 내부의 키(key) 이름과 형식을 확인합니다.
"""

import sys
from safetensors.torch import load_file
from pathlib import Path


def inspect_lora_file(file_path):
    """
    LoRA safetensors 파일의 내용을 검사합니다.

    Args:
        file_path: .safetensors 파일 경로
    """
    print(f"\n{'='*80}")
    print(f"LoRA 파일 검사: {file_path}")
    print(f"{'='*80}\n")

    # 파일 존재 확인
    if not Path(file_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    # safetensors 로드
    try:
        state_dict = load_file(file_path)
        print(f"✅ 파일 로드 성공!")
        print(f"   총 키 개수: {len(state_dict)}")

        # 파일 크기
        file_size = Path(file_path).stat().st_size
        print(f"   파일 크기: {file_size / (1024*1024):.2f} MB")

        # 키 형식 분석
        print(f"\n📋 키(Key) 형식 분석:\n")

        # 샘플 키 출력 (처음 10개)
        print("처음 10개 키:")
        for i, (key, tensor) in enumerate(state_dict.items()):
            if i >= 10:
                break
            print(f"  [{i+1}] {key}")
            print(f"      Shape: {tensor.shape}, Dtype: {tensor.dtype}")

        # 키 형식 판별
        print(f"\n🔍 형식 판별:")

        sample_keys = list(state_dict.keys())[:5]

        if any('lora_unet_' in k for k in sample_keys):
            print("  ✅ WebUI/Civitai 형식")
            print("     (예: lora_unet_down_blocks_0_...)")
        elif any('base_model.model.' in k for k in sample_keys):
            print("  ✅ PEFT 형식")
            print("     (예: base_model.model.down_blocks.0...)")
        elif any('unet.' in k or 'text_encoder.' in k for k in sample_keys):
            print("  ✅ Diffusers 형식")
            print("     (예: unet.down_blocks.0...)")
        else:
            print("  ⚠️  알 수 없는 형식")

        # lora_down/lora_up 확인
        has_lora_down = any('lora_down' in k for k in sample_keys)
        has_lora_up = any('lora_up' in k for k in sample_keys)
        has_lora_A = any('lora_A' in k for k in sample_keys)
        has_lora_B = any('lora_B' in k for k in sample_keys)

        print(f"\n  LoRA 레이어 타입:")
        if has_lora_down and has_lora_up:
            print(f"    ✅ lora_down, lora_up (WebUI 스타일)")
        if has_lora_A and has_lora_B:
            print(f"    ✅ lora_A, lora_B (PEFT 스타일)")

        print(f"\n{'='*80}\n")

    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python inspect_lora.py <safetensors_파일_경로>")
        print("\n예시:")
        print("  python inspect_lora.py my_lora_model/checkpoint-250/lora_weights.safetensors")
        print("  python inspect_lora.py downloaded_civitai_model.safetensors")
        sys.exit(1)

    file_path = sys.argv[1]
    inspect_lora_file(file_path)
