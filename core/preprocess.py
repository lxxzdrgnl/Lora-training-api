"""
데이터셋 전처리 모듈
- 캐릭터 감지 및 크롭
- 텍스트 제거 (Inpainting)
- 이미지 리사이즈
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import easyocr
from tqdm import tqdm
import torch
from rembg import remove
from transformers import BlipProcessor, BlipForConditionalGeneration

from .config import PreprocessConfig


class ImagePreprocessor:
    """이미지 전처리 클래스"""

    def __init__(self, config: PreprocessConfig = None, enable_captioning: bool = True, trigger_word: str = None):
        self.config = config or PreprocessConfig()
        self.enable_captioning = enable_captioning
        self.trigger_word = trigger_word

        # OCR 초기화 (Modal Volume 캐싱)
        print("Initializing OCR...")

        # Modal Volume 캐시 경로 지정
        import os
        cache_dir = "/cache/easyocr_model"
        os.makedirs(cache_dir, exist_ok=True)

        self.reader = easyocr.Reader(
            ['ko', 'en'],
            gpu=torch.cuda.is_available(),
            model_storage_directory=cache_dir,  # 캐시 디렉토리 지정
            download_enabled=True  # 캐시 없으면 다운로드
        )
        print("✅ EasyOCR initialized (models cached)")

        # BLIP 캡셔닝 모델 초기화 (Modal Volume 캐싱)
        if self.enable_captioning:
            print("Initializing BLIP captioning model...")

            # Modal Volume 캐시 경로
            cache_dir = "/cache/blip_model"

            # 캐시된 모델이 있는지 확인
            import os
            if os.path.exists(cache_dir) and os.path.exists(os.path.join(cache_dir, "config.json")):
                print(f"✅ Using cached BLIP model from: {cache_dir}")
                self.caption_processor = BlipProcessor.from_pretrained(cache_dir)
                self.caption_model = BlipForConditionalGeneration.from_pretrained(
                    cache_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            else:
                # 캐시가 없으면 다운로드 후 캐싱
                print(f"📥 Downloading BLIP model (first time only)...")
                self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.caption_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )

                # Modal Volume에 저장
                os.makedirs(cache_dir, exist_ok=True)
                self.caption_processor.save_pretrained(cache_dir)
                self.caption_model.save_pretrained(cache_dir)
                print(f"✅ BLIP model cached to: {cache_dir}")

            if torch.cuda.is_available():
                self.caption_model.to("cuda")
            self.caption_model.eval()
            print("BLIP model loaded!")

    def detect_text_regions(self, image):
        """텍스트 영역 감지"""
        if isinstance(image, Image.Image):
            image = np.array(image)

        results = self.reader.readtext(image)

        text_boxes = []
        for (bbox, _, _) in results:
            points = np.array(bbox).astype(np.int32)
            x_min = points[:, 0].min()
            y_min = points[:, 1].min()
            x_max = points[:, 0].max()
            y_max = points[:, 1].max()
            text_boxes.append([x_min, y_min, x_max, y_max])

        return text_boxes

    def inpaint_text_regions(self, image, text_boxes):
        """텍스트 영역을 inpainting으로 제거"""
        if len(text_boxes) == 0:
            return image

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for box in text_boxes:
            x1, y1, x2, y2 = box
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return inpainted

    def detect_character_bbox(self, image):
        """캐릭터 전신 감지 (배경 제거 기반)"""
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image

        output = remove(image_pil, only_mask=True)
        mask = np.array(output)

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        return [x, y, x + w, y + h]

    def expand_bbox(self, char_bbox, img_shape, expand_ratio=None):
        """캐릭터 bbox 확장 (전신 포함)"""
        if expand_ratio is None:
            expand_ratio = self.config.bbox_expand_ratio

        height, width = img_shape[:2]
        x1, y1, x2, y2 = char_bbox

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        expand_w = int(bbox_w * expand_ratio)
        expand_h = int(bbox_h * expand_ratio)

        new_x1 = max(0, x1 - expand_w)
        new_y1 = max(0, y1 - expand_h)
        new_x2 = min(width, x2 + expand_w)
        new_y2 = min(height, y2 + expand_h)

        return [new_x1, new_y1, new_x2, new_y2]

    def generate_caption(self, image_pil):
        """BLIP으로 이미지 캡션 생성"""
        if not self.enable_captioning:
            # 캡셔닝 비활성화 시 trigger_word만 반환 (None이면 빈 문자열)
            return self.trigger_word if self.trigger_word else ""

        with torch.no_grad():
            inputs = self.caption_processor(image_pil, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            out = self.caption_model.generate(
                **inputs,
                max_length=50,
                num_beams=3
            )

            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)

        # Trigger word 추가 (None이 아닐 경우에만)
        if self.trigger_word:
            full_caption = f"{self.trigger_word}, {caption}"
        else:
            full_caption = caption

        return full_caption

    def crop_and_resize(self, image, bbox, target_size=None):
        """크롭 및 리사이즈 (정사각형 + 패딩)"""
        if target_size is None:
            target_size = self.config.image_size

        if isinstance(image, Image.Image):
            image = np.array(image)

        x1, y1, x2, y2 = bbox
        cropped = image[y1:y2, x1:x2]

        h, w = cropped.shape[:2]

        # 정사각형 패딩
        if h > w:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            cropped = cv2.copyMakeBorder(
                cropped, 0, 0, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
        elif w > h:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            cropped = cv2.copyMakeBorder(
                cropped, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )

        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

        return resized

    def process_single_image(self, image_path: str, output_path: str = None):
        """단일 이미지 전처리 + 캡셔닝"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # 1. 텍스트 감지 및 제거
        if self.config.enable_text_removal:
            text_boxes = self.detect_text_regions(image)
            if len(text_boxes) > 0:
                image = self.inpaint_text_regions(image, text_boxes)

        # 2. 캐릭터 감지
        char_bbox = self.detect_character_bbox(image)
        if char_bbox is None:
            raise ValueError(f"No character detected in {image_path}")

        # 3. Bbox 확장
        expanded_bbox = self.expand_bbox(char_bbox, image.shape)

        # 4. 크롭 및 리사이즈
        result = self.crop_and_resize(image, expanded_bbox)

        # 5. 캡션 생성 (크롭된 이미지로)
        caption = None
        if self.enable_captioning:
            result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            caption = self.generate_caption(result_pil)

        # 6. 저장
        if output_path:
            cv2.imwrite(str(output_path), result)

            # 캡션을 .txt 파일로 저장
            if caption:
                caption_path = Path(output_path).with_suffix('.txt')
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)

        return result, caption

    def process_batch(self, input_dir: str = None, output_dir: str = None):
        """배치 전처리"""
        input_dir = Path(input_dir or self.config.input_dir)
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(exist_ok=True)

        # 이미지 파일 찾기
        image_files = (
            list(input_dir.glob("*.png")) +
            list(input_dir.glob("*.jpg")) +
            list(input_dir.glob("*.jpeg")) +
            list(input_dir.glob("*.webp"))
        )

        if len(image_files) == 0:
            raise ValueError(f"No images found in {input_dir}")

        print(f"Found {len(image_files)} images")

        success_count = 0
        failed_files = []
        sample_captions = []

        for img_file in tqdm(image_files, desc="Preprocessing + Captioning"):
            try:
                output_file = output_dir / f"{img_file.stem}_clean.png"
                _, caption = self.process_single_image(img_file, output_file)
                success_count += 1

                # 처음 3개 캡션 샘플 저장
                if len(sample_captions) < 3 and caption:
                    sample_captions.append((img_file.name, caption))

            except Exception as e:
                print(f"\nError processing {img_file.name}: {e}")
                failed_files.append(img_file.name)
                continue

        print(f"\nPreprocessing completed: {success_count}/{len(image_files)} successful")

        # 샘플 캡션 출력
        if sample_captions:
            print(f"\nSample captions:")
            for filename, caption in sample_captions:
                print(f"  {filename}: {caption}")

        if failed_files:
            print(f"Failed files: {', '.join(failed_files)}")

        return {
            "total": len(image_files),
            "success": success_count,
            "failed": len(failed_files),
            "output_dir": str(output_dir)
        }


def caption_only_dataset(
    input_dir: str,
    trigger_word: str = None
):
    """
    캡셔닝만 수행 (전처리 없이 원본 이미지에 캡션 생성)

    Args:
        input_dir: 원본 이미지 폴더
        trigger_word: 트리거 워드 (None이면 캡션에 트리거 워드 추가 안 함)

    Returns:
        dict: 캡셔닝 결과 정보
    """
    from pathlib import Path
    from tqdm import tqdm
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image

    input_path = Path(input_dir)

    # 이미지 파일 찾기
    image_files = (
        list(input_path.glob("*.png")) +
        list(input_path.glob("*.jpg")) +
        list(input_path.glob("*.jpeg")) +
        list(input_path.glob("*.webp"))
    )

    if len(image_files) == 0:
        raise ValueError(f"No images found in {input_dir}")

    print(f"Found {len(image_files)} images for captioning")

    # BLIP 캡셔닝 모델 초기화 (Modal Volume 캐싱)
    print("Initializing BLIP captioning model...")

    # Modal Volume 캐시 경로
    import os
    cache_dir = "/cache/blip_model"

    # 캐시된 모델이 있는지 확인
    if os.path.exists(cache_dir) and os.path.exists(os.path.join(cache_dir, "config.json")):
        print(f"✅ Using cached BLIP model from: {cache_dir}")
        caption_processor = BlipProcessor.from_pretrained(cache_dir)
        caption_model = BlipForConditionalGeneration.from_pretrained(
            cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    else:
        # 캐시가 없으면 다운로드 후 캐싱
        print(f"📥 Downloading BLIP model (first time only)...")
        caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Modal Volume에 저장
        os.makedirs(cache_dir, exist_ok=True)
        caption_processor.save_pretrained(cache_dir)
        caption_model.save_pretrained(cache_dir)
        print(f"✅ BLIP model cached to: {cache_dir}")

    if torch.cuda.is_available():
        caption_model.to("cuda")
    caption_model.eval()
    print("BLIP model loaded!")

    success_count = 0
    failed_files = []
    sample_captions = []

    for img_file in tqdm(image_files, desc="Captioning images"):
        try:
            # 이미지 로드
            image_pil = Image.open(img_file).convert("RGB")

            # 캡션 생성
            with torch.no_grad():
                inputs = caption_processor(image_pil, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                out = caption_model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=3
                )

                caption = caption_processor.decode(out[0], skip_special_tokens=True)

            # Trigger word 추가 (None이 아닐 경우에만)
            if trigger_word:
                full_caption = f"{trigger_word}, {caption}"
            else:
                full_caption = caption

            # 캡션을 .txt 파일로 저장
            caption_path = img_file.with_suffix('.txt')
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(full_caption)

            success_count += 1

            # 처음 3개 캡션 샘플 저장
            if len(sample_captions) < 3:
                sample_captions.append((img_file.name, full_caption))

        except Exception as e:
            print(f"\nError captioning {img_file.name}: {e}")
            failed_files.append(img_file.name)
            continue

    print(f"\nCaptioning completed: {success_count}/{len(image_files)} successful")

    # 샘플 캡션 출력
    if sample_captions:
        print(f"\nSample captions:")
        for filename, caption in sample_captions:
            print(f"  {filename}: {caption}")

    if failed_files:
        print(f"Failed files: {', '.join(failed_files)}")

    return {
        "total": len(image_files),
        "success": success_count,
        "failed": len(failed_files),
        "output_dir": str(input_dir)
    }


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    enable_text_removal: bool = True,
    enable_captioning: bool = True,
    image_size: int = 512,
    trigger_word: str = None
):
    """
    데이터셋 전처리 함수 (Modal API용)

    Args:
        input_dir: 원본 이미지 폴더
        output_dir: 출력 폴더
        enable_text_removal: 텍스트 제거 활성화
        enable_captioning: BLIP 자동 캡셔닝 활성화
        image_size: 출력 이미지 크기
        trigger_word: 트리거 워드 (None이면 캡션에 트리거 워드 추가 안 함)

    Returns:
        dict: 전처리 결과 정보
    """
    config = PreprocessConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        enable_text_removal=enable_text_removal,
        image_size=image_size
    )

    preprocessor = ImagePreprocessor(config, enable_captioning=enable_captioning, trigger_word=trigger_word)
    return preprocessor.process_batch()


if __name__ == "__main__":
    # 테스트 실행
    result = preprocess_dataset(
        input_dir="./dataset",
        output_dir="./dataset_clean",
        enable_text_removal=True,
        enable_captioning=True
    )
    print(f"\nResult: {result}")
