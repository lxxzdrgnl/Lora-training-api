"""
LoRA 변환 유틸리티 (PEFT ↔ WebUI/Civitai 형식)
"""

import torch
from safetensors.torch import save_file, load_file
import os


def convert_peft_to_webui(peft_state_dict):
    """
    PEFT 형식의 LoRA 가중치를 WebUI/Civitai 형식으로 변환

    Args:
        peft_state_dict: PEFT 형식의 state dict

    Returns:
        dict: WebUI 형식의 state dict
    """
    webui_state_dict = {}

    for key, value in peft_state_dict.items():
        # PEFT 키 형식: base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight
        # WebUI 키 형식: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight

        if 'lora' not in key.lower():
            continue

        # base_model.model. 제거
        if key.startswith('base_model.model.'):
            key = key.replace('base_model.model.', '', 1)

        # lora_A/lora_B와 weight/bias 분리
        if '.lora_A.weight' in key:
            key_base = key.replace('.lora_A.weight', '')
            lora_type = 'lora_down'
            weight_type = 'weight'
        elif '.lora_A.bias' in key:
            key_base = key.replace('.lora_A.bias', '')
            lora_type = 'lora_down'
            weight_type = 'bias'
        elif '.lora_B.weight' in key:
            key_base = key.replace('.lora_B.weight', '')
            lora_type = 'lora_up'
            weight_type = 'weight'
        elif '.lora_B.bias' in key:
            key_base = key.replace('.lora_B.bias', '')
            lora_type = 'lora_up'
            weight_type = 'bias'
        else:
            # lora_A, lora_B가 아닌 경우 스킵
            continue

        # 점(.)을 밑줄(_)로 변경
        key_base = key_base.replace('.', '_')

        # WebUI 형식으로 조합
        webui_key = f"lora_unet_{key_base}.{lora_type}.{weight_type}"
        webui_state_dict[webui_key] = value.cpu()

    return webui_state_dict


def save_lora_as_webui(unet_model, save_path, lora_alpha=64, lora_rank=32):
    """
    PEFT UNet 모델에서 LoRA 가중치를 WebUI 형식으로 저장

    Args:
        unet_model: PEFT가 적용된 UNet 모델
        save_path: 저장할 .safetensors 파일 경로
        lora_alpha: LoRA alpha 값 (스케일링)
        lora_rank: LoRA rank 값
    """
    # LoRA 가중치만 추출
    peft_state_dict = {k: v.cpu() for k, v in unet_model.state_dict().items() if 'lora' in k}

    # WebUI 형식으로 변환
    webui_state_dict = convert_peft_to_webui(peft_state_dict)

    # 메타데이터에 alpha 값 추가 (각 LoRA 레이어마다)
    # Civitai/WebUI 형식은 이런 식으로 alpha를 저장함
    metadata = {}
    for key in webui_state_dict.keys():
        if '.lora_up.weight' in key or '.lora_down.weight' in key:
            # 레이어 이름 추출
            base_key = key.rsplit('.', 2)[0]  # ".lora_up.weight" 또는 ".lora_down.weight" 제거
            alpha_key = f"{base_key}.alpha"
            if alpha_key not in webui_state_dict:
                # alpha 값을 tensor로 저장
                webui_state_dict[alpha_key] = torch.tensor(lora_alpha, dtype=torch.float32)

    # safetensors로 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_file(webui_state_dict, save_path)

    print(f"✅ Saved {len(webui_state_dict)} LoRA weights in WebUI format")
    print(f"   Alpha: {lora_alpha}, Rank: {lora_rank}")
    print(f"   File: {save_path}")

    return save_path
