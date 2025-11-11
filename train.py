"""
LoRA 모델 학습 스크립트
데이터셋 전처리 + LoRA 파인튜닝
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
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess_dataset import CharacterCropper


# =============================================================================
# 설정
# =============================================================================

class Config:
    """학습 설정"""
    # 환경
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "xyn-ai/anything-v4.0"

    # 데이터
    raw_dataset_path = "./dataset"
    clean_dataset_path = "./dataset_clean"
    image_size = 512

    # LoRA
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.0
    target_modules = ["to_q", "to_v", "to_k", "to_out.0"]

    # 학습
    num_epochs = 100
    learning_rate = 5e-5
    weight_decay = 1e-2
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0

    # Diffusion
    noise_offset = 0.1
    snr_gamma = 5.0

    # 출력
    output_dir = "my_lora_model"
    save_every_n_epochs = 50


# =============================================================================
# 전처리
# =============================================================================

def preprocess_dataset(config):
    """데이터셋 자동 전처리"""
    print("\n" + "="*60)
    print("STEP 1: Dataset Preprocessing")
    print("="*60)

    raw_path = Path(config.raw_dataset_path)
    clean_path = Path(config.clean_dataset_path)

    # 이미 전처리된 데이터가 있는지 확인
    if clean_path.exists():
        existing_files = list(clean_path.glob("*.png"))
        if len(existing_files) > 0:
            print(f"\n✅ Found {len(existing_files)} preprocessed images in {clean_path}")
            user_input = input("Skip preprocessing? (y/n): ").strip().lower()
            if user_input == 'y':
                return str(clean_path)

    # 전처리 실행
    clean_path.mkdir(exist_ok=True)

    # 이미지 파일 찾기
    image_files = (
        list(raw_path.glob("*.png")) +
        list(raw_path.glob("*.jpg")) +
        list(raw_path.glob("*.jpeg")) +
        list(raw_path.glob("*.webp"))
    )

    if len(image_files) == 0:
        raise ValueError(f"No images found in {raw_path}")

    print(f"\nFound {len(image_files)} raw images")
    print("Processing...")

    # Cropper 초기화
    cropper = CharacterCropper()

    # 처리
    success_count = 0
    for img_file in tqdm(image_files, desc="Preprocessing"):
        try:
            result = cropper.process_image(img_file, visualize=False)

            if result is not None:
                import cv2
                output_file = clean_path / f"{img_file.stem}_clean.png"
                cv2.imwrite(str(output_file), result)
                success_count += 1
        except Exception as e:
            print(f"\n⚠ Error processing {img_file.name}: {e}")
            continue

    print(f"\n✅ Preprocessing completed: {success_count}/{len(image_files)} successful")

    if success_count == 0:
        raise ValueError("No images were successfully preprocessed!")

    return str(clean_path)


# =============================================================================
# 데이터 로딩
# =============================================================================

def load_images(dataset_path, image_size=512):
    """이미지 파일 로드"""
    path = Path(dataset_path)
    image_files = list(path.glob("*.png")) + list(path.glob("*.jpg"))

    if len(image_files) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    return image_files


def load_and_preprocess_image(img_path, size=512):
    """이미지를 VAE latent로 변환"""
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5  # normalize to [-1, 1]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(Config.device, dtype=torch.float16)


def encode_prompt(text_encoder, tokenizer, prompt_text):
    """텍스트 프롬프트를 인코딩"""
    text_input = tokenizer(
        prompt_text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(Config.device))[0]
    return text_embeddings


def generate_prompt():
    """고정 프롬프트 (Trigger word 방식)"""
    return "sks girl, black hair, long hair, black and white manga style, monochrome illustration"


# =============================================================================
# 모델 로딩
# =============================================================================

def load_models(config):
    """Stable Diffusion 컴포넌트 + LoRA 설정"""
    print("\n" + "="*60)
    print("STEP 2: Model Loading")
    print("="*60)

    print(f"Loading from: {config.model_id}")

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
    print("\nSetting up LoRA...")
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
    print("✅ Models loaded successfully!")

    return vae, unet, text_encoder, tokenizer, noise_scheduler


# =============================================================================
# 학습 유틸
# =============================================================================

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


# =============================================================================
# 학습 루프
# =============================================================================

def train(config):
    """메인 학습 함수"""
    # 1. 전처리
    dataset_path = preprocess_dataset(config)

    # 2. 모델 로드
    vae, unet, text_encoder, tokenizer, noise_scheduler = load_models(config)

    # 3. 데이터 로드
    print("\n" + "="*60)
    print("STEP 3: Data Loading")
    print("="*60)

    image_files = load_images(dataset_path, config.image_size)
    print(f"Found {len(image_files)} preprocessed images")

    # 4. Optimizer & Scheduler 설정
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = len(image_files) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = total_steps // 10
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 5. 학습 시작
    print("\n" + "="*60)
    print("STEP 4: Training")
    print("="*60)
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size per image: {config.gradient_accumulation_steps}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print("="*60 + "\n")

    loss_history = []
    global_step = 0

    for epoch in range(config.num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(image_files, desc=f"Epoch {epoch+1}/{config.num_epochs}")

        for batch_idx, img_path in enumerate(progress_bar):
            # 이미지 로드
            pixel_values = load_and_preprocess_image(img_path, config.image_size)

            # VAE latent 변환
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 프롬프트 인코딩
            prompt = generate_prompt()
            encoder_hidden_states = encode_prompt(text_encoder, tokenizer, prompt)

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

        avg_loss = epoch_loss / len(image_files)
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")

        # 중간 저장
        if (epoch + 1) % config.save_every_n_epochs == 0:
            checkpoint_dir = f"{config.output_dir}_epoch{epoch+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            unet.save_pretrained(checkpoint_dir)
            print(f"  💾 Checkpoint saved: {checkpoint_dir}")

    # 6. 최종 저장
    print("\n" + "="*60)
    print("STEP 5: Saving Model")
    print("="*60)

    os.makedirs(config.output_dir, exist_ok=True)
    unet.save_pretrained(config.output_dir)
    print(f"✅ Model saved to: {config.output_dir}")

    # Loss 그래프 저장
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history, alpha=0.3, label='Raw Loss', color='blue')

    # Moving average
    window_size = 20
    if len(loss_history) >= window_size:
        moving_avg = np.convolve(
            loss_history,
            np.ones(window_size)/window_size,
            mode='valid'
        )
        plt.plot(
            range(window_size-1, len(loss_history)),
            moving_avg,
            label=f'Moving Average (window={window_size})',
            color='red',
            linewidth=2
        )

    plt.title('Training Loss Over Time')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=150)
    print("📊 Loss graph saved: training_loss.png")

    # 통계 출력
    print(f"\nTraining Statistics:")
    print(f"  Total steps: {len(loss_history)}")
    print(f"  Initial loss: {loss_history[0]:.4f}")
    print(f"  Final loss: {loss_history[-1]:.4f}")
    print(f"  Average loss: {np.mean(loss_history):.4f}")
    print(f"  Min loss: {np.min(loss_history):.4f}")
    print(f"  Max loss: {np.max(loss_history):.4f}")

    print("\n" + "="*60)
    print("✅ Training completed successfully!")
    print("="*60)
    print(f"\nTo generate images, run:")
    print(f"  python inference.py --lora_path {config.output_dir}")
    print("="*60)


# =============================================================================
# 메인
# =============================================================================

if __name__ == "__main__":
    config = Config()
    train(config)
