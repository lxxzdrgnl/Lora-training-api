"""
학습 설정 파일
"""

import torch
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """학습 설정"""
    # 환경
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_id: str = "stablediffusionapi/anything-v5"
    raw_dataset_path: str = "./dataset"
    clean_dataset_path: str = "./dataset_clean"
    image_size: int = 512

    # LoRA
    lora_r: int = 32  # LoRA rank (표현력)
    lora_alpha: int = None  # rank 값과 동일하게 설정
    lora_dropout: float = 0.0  # 작은 데이터셋은 dropout 없음
    target_modules: list = None  # ["to_q", "to_v", "to_k", "to_out.0"]

    # 학습
    num_epochs: int = 250
    learning_rate: float = 2e-5  # 학습률
    weight_decay: float = 1e-2
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Diffusion
    noise_offset: float = 0.1  # 어두운 이미지 개선
    snr_gamma: float = 5.0  # Min-SNR weighting

    # 출력
    output_dir: str = "my_lora_model"

    # 프롬프트 (BLIP 자동 캡셔닝 사용, None이면 트리거 워드 없이 캡션만 사용)
    trigger_word: str = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["to_q", "to_v", "to_k", "to_out.0"]
        if self.lora_alpha is None:
            self.lora_alpha = self.lora_r


@dataclass
class InferenceConfig:
    """추론 설정"""
    model_id: str = "stablediffusionapi/anything-v5"
    lora_path: str = "my_lora_model/checkpoint-200"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 생성 옵션
    prompt: str = "portrait, high quality"  # sks가 자동으로 추가됨
    negative_prompt: str = "low quality, blurry, ugly, distorted, deformed"
    num_images: int = 3
    steps: int = 40
    guidance_scale: float = 6.5
    lora_scale: float = 1.0  # LoRA 강도 (0.0~2.0)
    seed: int = None

    # 출력
    output_dir: str = "outputs"

    # Trigger word
    trigger_word: str = "sks"


@dataclass
class PreprocessConfig:
    """전처리 설정"""
    input_dir: str = "./dataset"
    output_dir: str = "./dataset_clean"
    image_size: int = 512

    # 전처리 옵션
    enable_text_removal: bool = True  # 텍스트 제거 활성화
    bbox_expand_ratio: float = 0.3  # bbox 확장 비율 (전신 포함)

    # 시각화
    visualize: bool = False  # 디버깅용 시각화
