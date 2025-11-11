"""
학습 실행 스크립트
전처리 + LoRA 학습을 한번에 실행
"""

from core.config import TrainingConfig
from core.train import train_with_preprocessing


if __name__ == "__main__":
    # 설정
    config = TrainingConfig()

    print("="*60)
    print("LoRA Training Pipeline")
    print("="*60)
    print(f"Dataset: {config.raw_dataset_path}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print("="*60)

    # 학습 실행 (전처리 + 학습)
    result = train_with_preprocessing(
        raw_dataset_path=config.raw_dataset_path,
        output_dir=config.output_dir,
        config=config,
        skip_preprocessing=False  # False: 전처리 실행, True: 스킵
    )

    print(f"\n✅ Training completed!")
    print(f"Final loss: {result['final_loss']:.4f}")
    print(f"Model saved to: {result['model_path']}")
