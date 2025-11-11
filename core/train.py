"""
LoRA 학습 모듈
"""

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .config import TrainingConfig
from .preprocess import preprocess_dataset


def load_models(config: TrainingConfig):
    """
    Stable Diffusion 모델 + LoRA 설정

    Args:
        config: 학습 설정

    Returns:
        tuple: (vae, unet, text_encoder, tokenizer, noise_scheduler)
    """
    print(f"\nLoading models from: {config.model_id}")

    # VAE, UNet, Text Encoder 로드
    vae = AutoencoderKL.from_pretrained(
        config.model_id, subfolder="vae", torch_dtype=torch.float16
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.model_id, subfolder="unet", torch_dtype=torch.float16
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.model_id, subfolder="text_encoder", torch_dtype=torch.float16
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        config.model_id, subfolder="tokenizer"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.model_id, subfolder="scheduler"
    )

    # VAE와 Text Encoder는 freeze
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Gradient checkpointing
    unet.enable_gradient_checkpointing()

    # Device로 이동
    vae.to(config.device)
    text_encoder.to(config.device)
    unet.to(config.device)

    # LoRA 설정
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)
    unet.train()

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    return vae, unet, text_encoder, tokenizer, noise_scheduler


def load_images_with_captions(dataset_path: str, trigger_word: str = "sks"):
    """이미지 파일 + 캡션 로드"""
    path = Path(dataset_path)
    image_files = list(path.glob("*.png")) + list(path.glob("*.jpg"))

    if len(image_files) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    # 이미지와 캡션 매핑
    image_caption_pairs = []
    for img_file in image_files:
        # 캡션 파일 찾기
        caption_file = img_file.with_suffix('.txt')

        if caption_file.exists():
            # 캡션 파일이 있으면 읽기
            with open(caption_file, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        else:
            # 없으면 기본 trigger word만 사용
            caption = trigger_word

        image_caption_pairs.append((img_file, caption))

    print(f"Loaded {len(image_caption_pairs)} images with captions")

    # 샘플 캡션 출력
    if len(image_caption_pairs) > 0:
        print(f"\nSample captions from training data:")
        for i in range(min(3, len(image_caption_pairs))):
            img_file, caption = image_caption_pairs[i]
            print(f"  {img_file.name}: {caption}")

    return image_caption_pairs


def load_and_preprocess_image(img_path: str, device: str, size: int = 512):
    """이미지를 tensor로 변환"""
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5  # normalize to [-1, 1]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(device, dtype=torch.float16)


def encode_prompt(text_encoder, tokenizer, prompt_text: str, device: str):
    """텍스트 프롬프트 인코딩"""
    text_input = tokenizer(
        prompt_text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    return text_embeddings


def compute_snr(timesteps, noise_scheduler):
    """Min-SNR weighting을 위한 SNR 계산"""
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    snr = (alpha / sigma) ** 2
    return snr




def train_lora(
    dataset_path: str,
    output_dir: str,
    config: TrainingConfig = None
):
    """
    LoRA 학습 함수 (Modal API용)

    Args:
        dataset_path: 전처리된 데이터셋 경로
        output_dir: 모델 저장 경로
        config: 학습 설정 (None이면 기본값 사용)

    Returns:
        dict: 학습 결과 정보
    """
    if config is None:
        config = TrainingConfig()

    config.output_dir = output_dir

    # 모델 로드
    vae, unet, text_encoder, tokenizer, noise_scheduler = load_models(config)

    # 데이터 로드 (이미지 + 캡션)
    image_caption_pairs = load_images_with_captions(dataset_path, config.trigger_word)

    # Optimizer & Scheduler
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = len(image_caption_pairs) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = total_steps // 10
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 학습 시작
    print(f"\nStarting training:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Total steps: {total_steps}")

    loss_history = []
    global_step = 0

    for epoch in range(config.num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(image_caption_pairs, desc=f"Epoch {epoch+1}/{config.num_epochs}")

        for batch_idx, (img_path, caption) in enumerate(progress_bar):
            # 이미지 로드
            pixel_values = load_and_preprocess_image(img_path, config.device, config.image_size)

            # VAE latent 변환
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 프롬프트 인코딩 (각 이미지의 캡션 사용)
            encoder_hidden_states = encode_prompt(
                text_encoder, tokenizer, caption, config.device
            )

            # Noise 추가
            noise = torch.randn_like(latents)
            if config.noise_offset > 0:
                noise += config.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1),
                    device=latents.device
                )

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=config.device
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # UNet으로 noise 예측
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Loss 계산
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
            loss = loss.mean([1, 2, 3])

            # Min-SNR weighting
            if config.snr_gamma is not None:
                snr = compute_snr(timesteps, noise_scheduler)
                mse_loss_weights = torch.stack(
                    [snr, config.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                mse_loss_weights = mse_loss_weights / snr
                loss = loss * mse_loss_weights

            loss = loss.mean() / config.gradient_accumulation_steps

            # Backward
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Loss 기록
            actual_loss = loss.item() * config.gradient_accumulation_steps
            epoch_loss += actual_loss
            loss_history.append(actual_loss)
            global_step += 1

            # Progress bar 업데이트
            current_lr = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                "loss": f"{actual_loss:.4f}",
                "lr": f"{current_lr:.2e}"
            })

            # 메모리 정리
            if global_step % 10 == 0:
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(image_caption_pairs)
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")

    # 모델 저장
    print(f"\nSaving model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    unet.save_pretrained(output_dir)

    # 통계 출력
    print(f"\nTraining Statistics:")
    print(f"  Total steps: {len(loss_history)}")
    print(f"  Initial loss: {loss_history[0]:.4f}")
    print(f"  Final loss: {loss_history[-1]:.4f}")
    print(f"  Average loss: {np.mean(loss_history):.4f}")
    print(f"  Min loss: {np.min(loss_history):.4f}")

    return {
        "total_steps": len(loss_history),
        "final_loss": loss_history[-1],
        "avg_loss": np.mean(loss_history),
        "model_path": output_dir
    }


def train_with_preprocessing(
    raw_dataset_path: str,
    output_dir: str,
    config: TrainingConfig = None,
    skip_preprocessing: bool = False
):
    """
    전처리 + 학습 전체 파이프라인 (Modal API용)

    Args:
        raw_dataset_path: 원본 데이터셋 경로
        output_dir: 모델 저장 경로
        config: 학습 설정
        skip_preprocessing: 전처리 스킵 여부

    Returns:
        dict: 학습 결과
    """
    if config is None:
        config = TrainingConfig()

    # 1. 전처리
    clean_dataset_path = config.clean_dataset_path

    if not skip_preprocessing:
        print("\n" + "="*60)
        print("STEP 1: Dataset Preprocessing")
        print("="*60)

        preprocess_result = preprocess_dataset(
            input_dir=raw_dataset_path,
            output_dir=clean_dataset_path
        )
        print(f"Preprocessing result: {preprocess_result}")
    else:
        print(f"Skipping preprocessing, using: {clean_dataset_path}")

    # 2. 학습
    print("\n" + "="*60)
    print("STEP 2: Training")
    print("="*60)

    train_result = train_lora(
        dataset_path=clean_dataset_path,
        output_dir=output_dir,
        config=config
    )

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"Model saved to: {output_dir}")
    print(f"To generate images, run:")
    print(f"  python inference.py --lora_path {output_dir}")

    return train_result


if __name__ == "__main__":
    # 테스트 실행
    config = TrainingConfig()
    result = train_with_preprocessing(
        raw_dataset_path="./dataset",
        output_dir="my_lora_model",
        config=config
    )
    print(f"\nResult: {result}")
