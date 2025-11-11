"""
자동 데이터셋 전처리 스크립트
- 캐릭터 전신 감지 및 크롭
- 텍스트/말풍선 영역 회피
- 서비스화를 위한 자동 파이프라인
"""

import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import easyocr
from tqdm import tqdm
import torch
from rembg import remove  # 캐릭터 세그멘테이션용

class CharacterCropper:
    def __init__(self):
        # OCR 초기화 (한글, 영어 지원)
        print("Initializing OCR...")
        self.reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())

    def detect_text_regions(self, image):
        """텍스트 영역 감지"""
        # PIL Image를 numpy array로 변환
        if isinstance(image, Image.Image):
            image = np.array(image)

        # OCR로 텍스트 영역 감지
        results = self.reader.readtext(image)

        text_boxes = []
        for (bbox, text, prob) in results:
            # bbox는 4개의 좌표 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
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

        # 마스크 생성 (텍스트 영역)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for box in text_boxes:
            x1, y1, x2, y2 = box
            # 텍스트 영역을 약간 확장해서 마스크
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Inpainting으로 텍스트 제거
        inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return inpainted

    def detect_character_bbox(self, image):
        """캐릭터 전신 감지 (배경 제거 기반)"""
        # PIL Image로 변환
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image

        # 배경 제거로 캐릭터 마스크 얻기
        print("  Detecting character...")
        output = remove(image_pil, only_mask=True)
        mask = np.array(output)

        # 이진화
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Morphology 연산으로 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Contour 찾기
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 가장 큰 contour 선택 (주인공 캐릭터)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        return [x, y, x + w, y + h]

    def expand_bbox(self, char_bbox, img_shape, expand_ratio=0.3):
        """
        캐릭터 bbox를 확장 (전신 포함)
        expand_ratio: 확장 비율 (기본 30% - 전신 크롭용)
        """
        height, width = img_shape[:2]
        x1, y1, x2, y2 = char_bbox

        # 현재 bbox 크기
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # 확장량 계산
        expand_w = int(bbox_w * expand_ratio)
        expand_h = int(bbox_h * expand_ratio)

        # 확장된 bbox
        new_x1 = max(0, x1 - expand_w)
        new_y1 = max(0, y1 - expand_h)
        new_x2 = min(width, x2 + expand_w)
        new_y2 = min(height, y2 + expand_h)

        return [new_x1, new_y1, new_x2, new_y2]

    def crop_and_resize(self, image, bbox, target_size=512):
        """
        캐릭터를 크롭하고 512x512로 리사이즈
        종횡비를 유지하면서 padding 추가
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        x1, y1, x2, y2 = bbox
        cropped = image[y1:y2, x1:x2]

        # 종횡비 계산
        h, w = cropped.shape[:2]

        # 정사각형 패딩
        if h > w:
            # 세로가 더 긴 경우
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            cropped = cv2.copyMakeBorder(
                cropped, 0, 0, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
        elif w > h:
            # 가로가 더 긴 경우
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            cropped = cv2.copyMakeBorder(
                cropped, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )

        # 512x512 리사이즈
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

        return resized

    def process_image(self, image_path, visualize=False):
        """이미지 전처리 전체 파이프라인"""
        print(f"\nProcessing: {os.path.basename(image_path)}")

        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Failed to load image")
            return None

        # 1. 텍스트 영역 감지
        print("  Detecting text regions...")
        text_boxes = self.detect_text_regions(image)
        print(f"  Found {len(text_boxes)} text regions")

        # 2. 텍스트 제거 (inpainting)
        if len(text_boxes) > 0:
            print("  Removing text with inpainting...")
            image = self.inpaint_text_regions(image, text_boxes)

        # 3. 캐릭터 전신 감지
        char_bbox = self.detect_character_bbox(image)
        if char_bbox is None:
            print("  No character detected")
            return None

        print(f"  Character bbox: {char_bbox}")

        # 4. Bbox 확장 (전신 포함)
        expanded_bbox = self.expand_bbox(
            char_bbox, image.shape
        )
        print(f"  Expanded bbox: {expanded_bbox}")

        # 4. 크롭 및 리사이즈
        result = self.crop_and_resize(image, expanded_bbox)

        # 시각화 (디버깅용)
        if visualize:
            # 텍스트 제거 전 원본 로드
            original = cv2.imread(str(image_path))

            # 텍스트 제거 후 이미지
            vis_image = image.copy()

            # 캐릭터 bbox (파랑)
            cv2.rectangle(vis_image, (char_bbox[0], char_bbox[1]),
                         (char_bbox[2], char_bbox[3]), (255, 0, 0), 2)

            # 확장된 bbox (초록)
            cv2.rectangle(vis_image, (expanded_bbox[0], expanded_bbox[1]),
                         (expanded_bbox[2], expanded_bbox[3]), (0, 255, 0), 3)

            # 결과 표시
            cv2.imshow('Original', cv2.resize(original, (600, 600)))
            cv2.imshow('Text Removed', cv2.resize(vis_image, (600, 600)))
            cv2.imshow('Final Result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result


def process_dataset(input_dir, output_dir, visualize=False):
    """데이터셋 배치 처리"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 이미지 파일 찾기
    image_files = list(input_path.glob("*.png")) + \
                  list(input_path.glob("*.jpg")) + \
                  list(input_path.glob("*.jpeg")) + \
                  list(input_path.glob("*.webp"))

    print(f"Found {len(image_files)} images")

    # Cropper 초기화
    cropper = CharacterCropper()

    # 처리
    success_count = 0
    for img_file in tqdm(image_files, desc="Processing dataset"):
        try:
            result = cropper.process_image(img_file, visualize=visualize)

            if result is not None:
                # 저장
                output_file = output_path / f"{img_file.stem}_clean.png"
                cv2.imwrite(str(output_file), result)
                success_count += 1
        except Exception as e:
            print(f"\nError processing {img_file.name}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"Success: {success_count}/{len(image_files)}")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # 설정
    INPUT_DIR = "./dataset"
    OUTPUT_DIR = "./dataset_clean"
    VISUALIZE = False  # True로 설정하면 각 단계별 시각화

    # 실행
    process_dataset(INPUT_DIR, OUTPUT_DIR, visualize=VISUALIZE)
