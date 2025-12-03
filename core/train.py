"""
LoRA 학습 모듈
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import multiprocessing

from .config import TrainingConfig
from .preprocess import preprocess_dataset

# CUDA + multiprocessing 호환성을 위해 spawn 방식 사용
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 이미 설정된 경우 무시


class LoRADataset(Dataset):
    """LoRA 학습용 Dataset 클래스 (DataLoader 병렬 로딩용)"""

    def __init__(
        self,
        image_caption_pairs,
        image_size=512,
        text_embeddings_cache=None,
        use_cached_latents=False,
        latents_dir=None
    ):
        """
        Args:
            image_caption_pairs: [(image_path, caption), ...] 형식의 리스트
            image_size: 이미지 크기 (기본 512)
            text_embeddings_cache: 사전 계산된 text embeddings 딕셔너리 (optional)
            use_cached_latents: True면 이미지 대신 사전 계산된 latents 로드
            latents_dir: 사전 계산된 latents 디렉토리 경로
        """
        self.data = image_caption_pairs
        self.image_size = image_size
        self.text_embeddings_cache = text_embeddings_cache
        self.use_cached_latents = use_cached_latents
        self.latents_dir = Path(latents_dir) if latents_dir else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        단일 이미지/latent와 캡션/embedding 로드

        Returns:
            tuple: (image_tensor or latent_tensor, text_embedding or caption)
        """
        img_path, caption = self.data[idx]

        # 이미지 또는 Latent 로드
        if self.use_cached_latents and self.latents_dir:
            # 사전 계산된 latent 로드 (초고속!)
            latent_file = self.latents_dir / f"{Path(img_path).stem}_latent.pt"
            if latent_file.exists():
                latent = torch.load(latent_file, map_location='cpu')  # CPU로 로드
                image_data = latent.squeeze(0)  # (1, C, H, W) → (C, H, W)
            else:
                # Latent 파일이 없으면 이미지 로드 (폴백)
                print(f"⚠️ Latent not found for {img_path.name}, loading image instead")
                img = Image.open(img_path).convert("RGB").resize(
                    (self.image_size, self.image_size), Image.LANCZOS
                )
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = (img_array - 0.5) / 0.5
                image_data = torch.from_numpy(img_array).permute(2, 0, 1)
        else:
            # 이미지 로드 및 전처리 (기존 방식)
            img = Image.open(img_path).convert("RGB").resize(
                (self.image_size, self.image_size), Image.LANCZOS
            )
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = (img_array - 0.5) / 0.5  # normalize to [-1, 1]
            image_data = torch.from_numpy(img_array).permute(2, 0, 1)

        # Text embedding
        if self.text_embeddings_cache is not None:
            # 캐시에서 가져오기 (초고속!)
            text_data = self.text_embeddings_cache[caption]
        else:
            # 캡션 텍스트 그대로 반환 (나중에 인코딩)
            text_data = caption

        return image_data, text_data


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


def load_images_with_captions(dataset_path: str, trigger_word: str = None):
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
            # 없으면 기본 trigger word 사용 (None이면 빈 문자열)
            caption = trigger_word if trigger_word else ""

        image_caption_pairs.append((img_file, caption))

    print(f"Loaded {len(image_caption_pairs)} images with captions")

    # 샘플 캡션 출력
    if len(image_caption_pairs) > 0:
        print(f"\nSample captions from training data:")
        for i in range(min(3, len(image_caption_pairs))):
            img_file, caption = image_caption_pairs[i]
            print(f"  {img_file.name}: {caption}")

    return image_caption_pairs


def load_and_preprocess_image(img_paths: list[str], device: str, size: int = 512):
    """이미지를 tensor로 변환"""
    images = []
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB").resize((size, size), Image.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5  # normalize to [-1, 1]
        images.append(torch.from_numpy(img_array).permute(2, 0, 1))
    return torch.stack(images).to(device, dtype=torch.float16)


def encode_prompt(text_encoder, tokenizer, prompt_texts: list[str], device: str):
    """텍스트 프롬프트 인코딩"""
    text_input = tokenizer(
        prompt_texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    return text_embeddings


def precompute_text_embeddings(
    text_encoder,
    tokenizer,
    captions_list: list[str],
    device: str
) -> dict:
    """
    유니크 캡션들의 embedding을 미리 계산하여 메모리에 캐싱

    Args:
        text_encoder: CLIP Text Encoder
        tokenizer: CLIP Tokenizer
        captions_list: 모든 캡션 리스트
        device: cuda/cpu

    Returns:
        dict: {caption: embedding_tensor} 딕셔너리
    """
    # 중복 제거
    unique_captions = list(set(captions_list))

    print(f"\n📝 Precomputing text embeddings for {len(unique_captions)} unique captions...")
    print(f"   (Total captions: {len(captions_list)}, Duplicates removed: {len(captions_list) - len(unique_captions)})")

    embeddings_cache = {}

    for caption in tqdm(unique_captions, desc="Computing embeddings"):
        # Tokenize
        text_input = tokenizer(
            [caption],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Encode
        with torch.no_grad():
            embedding = text_encoder(text_input.input_ids.to(device))[0]

        # CPU로 이동하여 캐싱 (pin_memory 호환)
        embeddings_cache[caption] = embedding.cpu()

    # 메모리 사용량 계산
    embedding_size = next(iter(embeddings_cache.values())).element_size() * \
                     next(iter(embeddings_cache.values())).nelement()
    total_memory = embedding_size * len(embeddings_cache) / 1024 / 1024  # MB

    print(f"✅ Text embeddings cached in memory")
    print(f"   Memory usage: {total_memory:.2f} MB ({embedding_size/1024:.1f} KB per embedding)")

    return embeddings_cache


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
    config: TrainingConfig = None,
    callback = None,
    use_cached_latents: bool = False,
    latents_dir: str = None
):
    """
    LoRA 학습 함수 (Modal API용)

    Args:
        dataset_path: 전처리된 데이터셋 경로
        output_dir: 모델 저장 경로
        config: 학습 설정 (None이면 기본값 사용)
        callback: 진행도 업데이트 콜백 함수 (status, phase, current_epoch, total_epochs, message)
        use_cached_latents: True면 사전 계산된 latents 사용
        latents_dir: 사전 계산된 latents 디렉토리 경로

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

    # Text Embeddings 사전 계산 (메모리 캐싱)
    all_captions = [caption for _, caption in image_caption_pairs]
    text_embeddings_cache = precompute_text_embeddings(
        text_encoder, tokenizer, all_captions, config.device
    )

    # Dataset 및 DataLoader 생성 (병렬 로딩 + Text Embeddings 캐싱 + VAE Latents 캐싱)
    if use_cached_latents:
        print(f"✅ Using cached VAE latents from: {latents_dir}")

    train_dataset = LoRADataset(
        image_caption_pairs,
        config.image_size,
        text_embeddings_cache=text_embeddings_cache,  # Text embeddings 캐시
        use_cached_latents=use_cached_latents,  # VAE latents 캐싱 여부
        latents_dir=latents_dir  # VAE latents 디렉토리
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # 에포크마다 셔플
        num_workers=10,  # 병렬 워커 수 증가 (GPU 사용률 향상)
        prefetch_factor=3,  # 워커당 3개 배치 미리 준비
        pin_memory=True,  # GPU 직접 전송 (빠름)
        drop_last=False,  # 마지막 배치도 사용
        persistent_workers=True  # 워커 재사용으로 시작 오버헤드 감소
    )
    print(f"✅ DataLoader created with 10 workers + prefetch (optimized for GPU utilization)")

    # Optimizer & Scheduler
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = (len(image_caption_pairs) // config.batch_size) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = total_steps // 10
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Mixed Precision Training 설정
    scaler = torch.cuda.amp.GradScaler()
    print("✅ Mixed Precision Training (AMP) enabled")

    # 학습 시작
    print(f"\nStarting training:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Total steps: {total_steps}")

    loss_history = []
    global_step = 0

    # 학습 시작 콜백
    if callback:
        callback(
            status="TRAINING",
            phase="training",
            current_epoch=0,
            total_epochs=config.num_epochs,
            message=f"학습 시작... (0/{config.num_epochs} 에포크)"
        )

    for epoch in range(config.num_epochs):
        epoch_loss = 0

        # DataLoader 사용 (병렬 로딩)
        total_batches = len(train_dataloader)
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

        # 에포크 시작 시 콜백 호출
        if callback:
            callback(
                status="TRAINING",
                phase="training",
                current_epoch=epoch + 1,
                total_epochs=config.num_epochs,
                message=f"Training {epoch + 1}/{config.num_epochs}"
            )

        for batch_idx, (data, text_data) in enumerate(progress_bar):
            # 데이터를 GPU로 이동
            if use_cached_latents:
                # 이미 latent! (VAE encoding 생략)
                latents = data.to(config.device, dtype=torch.float16)
            else:
                # 이미지 → latent 변환 필요
                pixel_values = data.to(config.device, dtype=torch.float16)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

            # Text embeddings (이미 캐시에서 가져온 상태!)
            # text_data는 이미 embedding tensor (배치로 스택됨)
            # DataLoader가 자동으로 배치를 만들어주므로 squeeze(1) 필요
            encoder_hidden_states = torch.stack([t.squeeze(0) for t in text_data]).to(config.device)

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

            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast():
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

            # Mixed Precision Backward Pass
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                # Gradient clipping (unscale first for correct norm calculation)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)

                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()

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

            # 메모리 정리 (너무 자주 호출하면 오히려 느려짐)
            if global_step % 30 == 0:
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / total_batches
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")

        # 50 에포크마다 체크포인트 저장
        if (epoch + 1) % 50 == 0 or (epoch + 1) == config.num_epochs:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch + 1}")
            print(f"\nSaving checkpoint to: {checkpoint_dir}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # WebUI/Civitai 형식으로 저장 (단일 .safetensors 파일)
            from .lora_utils import save_lora_as_webui

            safetensors_path = os.path.join(checkpoint_dir, "lora_weights.safetensors")
            save_lora_as_webui(
                unet,
                safetensors_path,
                lora_alpha=config.lora_alpha,
                lora_rank=config.lora_r
            )

    # 최종 모델 저장 메시지 (이제 체크포인트로 저장되므로 주석 처리 또는 수정)
    # print(f"\nSaving model to: {output_dir}")
    # os.makedirs(output_dir, exist_ok=True)
    # unet.save_pretrained(output_dir)

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
    skip_preprocessing: bool = False,
    callback = None
):
    """
    전처리 + 학습 전체 파이프라인 (Modal API용)

    Args:
        raw_dataset_path: 원본 데이터셋 경로
        output_dir: 모델 저장 경로
        config: 학습 설정
        skip_preprocessing: 전처리 스킵 여부
        callback: 진행도 업데이트 콜백 함수

    Returns:
        dict: 학습 결과
    """
    if config is None:
        config = TrainingConfig()

    # 1. 전처리 및 캡셔닝
    clean_dataset_path = config.clean_dataset_path

    if not skip_preprocessing:
        # 전처리 + 캡셔닝 (전체 파이프라인)
        # 전처리 시작 콜백
        if callback:
            callback(
                status="PREPROCESSING",
                phase="preprocessing",
                current_epoch=0,
                total_epochs=0,
                message="데이터셋 전처리 및 캡셔닝 중..."
            )

        print("\n" + "="*60)
        print("STEP 1: Dataset Preprocessing + Captioning")
        print("="*60)

        preprocess_result = preprocess_dataset(
            input_dir=raw_dataset_path,
            output_dir=clean_dataset_path,
            trigger_word=config.trigger_word  # trigger_word 전달
        )
        print(f"Preprocessing result: {preprocess_result}")

        # 전처리 완료 콜백
        if callback:
            callback(
                status="PREPROCESSING",
                phase="preprocessing",
                current_epoch=0,
                total_epochs=0,
                message="데이터셋 전처리 및 캡셔닝 완료"
            )
    else:
        # 전처리 스킵, 캡셔닝만 수행 (필수)
        from .preprocess import caption_only_dataset

        # 캡셔닝 시작 콜백
        if callback:
            callback(
                status="PREPROCESSING",
                phase="preprocessing",
                current_epoch=0,
                total_epochs=0,
                message="이미지 캡셔닝 중..."
            )

        print("\n" + "="*60)
        print("STEP 1: Captioning Only (Preprocessing Skipped)")
        print("="*60)

        caption_result = caption_only_dataset(
            input_dir=raw_dataset_path,
            trigger_word=config.trigger_word
        )
        print(f"Captioning result: {caption_result}")

        # 전처리 스킵하므로 원본 이미지 디렉토리를 학습에 사용
        clean_dataset_path = raw_dataset_path

        # 캡셔닝 완료 콜백
        if callback:
            callback(
                status="PREPROCESSING",
                phase="preprocessing",
                current_epoch=0,
                total_epochs=0,
                message="이미지 캡셔닝 완료"
            )

    # 2. VAE Latents 사전 계산 (학습 속도 30-40% 향상)
    print("\n" + "="*60)
    print("STEP 2: Precomputing VAE Latents (Speed Optimization)")
    print("="*60)

    latents_dir = os.path.join(output_dir, "cached_latents")

    if callback:
        callback(
            status="PREPROCESSING",
            phase="preprocessing",
            current_epoch=0,
            total_epochs=0,
            message="VAE latents 사전 계산 중... (학습 속도 향상을 위해)"
        )

    from .precompute_latents import precompute_latents
    precompute_latents(
        dataset_path=clean_dataset_path,
        output_path=latents_dir,
        model_id=config.model_id,
        image_size=config.image_size
    )
    print(f"✅ VAE latents cached to: {latents_dir}")

    # 3. 학습
    print("\n" + "="*60)
    print("STEP 3: Training with Cached Latents")
    print("="*60)

    train_result = train_lora(
        dataset_path=clean_dataset_path,
        output_dir=output_dir,
        config=config,
        callback=callback,
        use_cached_latents=True,
        latents_dir=latents_dir
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
