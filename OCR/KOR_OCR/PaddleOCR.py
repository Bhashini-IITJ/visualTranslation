import os
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageOps
import json
import re
import Levenshtein

# PaddleOCR 초기화
ocr = PaddleOCR(lang='korean', use_angle_cls=True, use_gpu=True)

# 데이터 경로 설정
IMAGE_PATH = r"E:\OCR\image"
GROUND_TRUTH_PATH = r"E:\OCR\correct"

# OCR 정답에서 "xxx" 제거
def clean_ground_truth(text):
    return ' '.join([word for word in text.split() if word != "xxx"]).strip()

# 정답 파일에서 텍스트 로드
def load_ground_truth(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            annotations = data.get('annotations', [])
            texts = [annotation.get('text', '') for annotation in annotations]
            raw_text = ' '.join(texts).strip()
            return clean_ground_truth(raw_text)
    except Exception as e:
        print(f"[❌ 에러] 정답 파일 로드 실패 - {json_path}: {str(e)}")
        return None

# 텍스트 전처리 (공백 및 특수문자 제거)
def clean_text(text):
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', text).strip().lower()

# 이미지에서 가장 큰 문자 영역 추출 (ROI 선택)
def extract_largest_text_region(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("L")  # 흑백 변환
            img = ImageOps.invert(img)  # 색상 반전 (텍스트 강조)
            img_array = np.array(img)

            # 이진화
            threshold = 128
            binary_img = np.where(img_array > threshold, 255, 0).astype(np.uint8)

            # 윤곽선 검출
            non_zero_coords = np.column_stack(np.where(binary_img > 0))
            if non_zero_coords.shape[0] == 0:
                print("[⚠️ 경고] 텍스트 영역을 찾지 못함")
                return img  # 원본 이미지 반환

            # 바운딩 박스 계산
            x_min, y_min = non_zero_coords.min(axis=0)
            x_max, y_max = non_zero_coords.max(axis=0)

            # ROI 추출 (문자 영역만 자르기)
            cropped_img = img.crop((y_min, x_min, y_max, x_max))
            return cropped_img

    except Exception as e:
        print(f"[❌ 오류] 이미지 처리 중 에러 발생: {image_path} - {str(e)}")
        return None

# 평가 함수 (OCR 성능 측정)
def token_based_accuracy(pred_text, gt_text):
    pred_tokens = set(pred_text.split())
    gt_tokens = set(gt_text.split())
    correct = len(pred_tokens & gt_tokens)
    total = len(gt_tokens)
    return correct, total

def substring_matching_accuracy(pred_text, gt_text):
    correct = sum(1 for token in gt_text.split() if token in pred_text)
    total = len(gt_text.split())
    return correct, total

def edit_distance_accuracy(pred_text, gt_text):
    distance = Levenshtein.distance(pred_text, gt_text)
    max_len = max(len(pred_text), len(gt_text))
    return max_len - distance, max_len

# 평가 실행
def evaluate_paddle_ocr(image_path, ground_truth_path):
    processed_images = 0
    skipped_files = []
    total_token_correct = 0
    total_token_total = 0
    total_substring_correct = 0
    total_substring_total = 0
    total_edit_distance_score = 0
    total_edit_distance_total = 0

    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

    if not os.path.exists(image_path) or len(os.listdir(image_path)) == 0:
        print("❌ 처리할 이미지가 없습니다. 경로를 확인하세요.")
        return

    for file in os.listdir(image_path):
        if file.lower().endswith(SUPPORTED_EXTENSIONS):
            try:
                processed_images += 1
                print(f"\n📌 현재 처리 중인 파일: {file} (총 {processed_images}개 이미지 처리됨)")

                image_file_path = os.path.join(image_path, file)
                json_file_path = os.path.join(ground_truth_path, os.path.splitext(file)[0] + '.json')

                if os.path.exists(json_file_path):
                    ground_truth = load_ground_truth(json_file_path)
                    if ground_truth is None:
                        skipped_files.append(file)
                        continue
                else:
                    print(f"⚠️ 정답 파일 없음: {json_file_path}")
                    continue

                # 가장 큰 문자 영역 추출
                cropped_image = extract_largest_text_region(image_file_path)
                if cropped_image is None:
                    skipped_files.append(file)
                    continue

                # OCR 인식
                result = ocr.ocr(np.array(cropped_image), cls=True)

                # OCR 결과 확인
                if not result or not result[0]:
                    print(f"⚠️ OCR 결과가 비어 있음 - 파일: {file}")
                    continue

                paddle_text = ' '.join([line[1][0] for line in result[0]]).strip()
                print(f"🔍 OCR 결과: {paddle_text}")
                print(f"✅ OCR 정답: {ground_truth}")

                # 정확도 평가
                token_correct, token_total = token_based_accuracy(paddle_text, ground_truth)
                substring_correct, substring_total = substring_matching_accuracy(paddle_text, ground_truth)
                edit_distance_score, edit_distance_total = edit_distance_accuracy(paddle_text, ground_truth)

                # 개별 이미지 정확도 출력
                if token_total > 0 or substring_total > 0 or edit_distance_total > 0:
                    token_accuracy = (token_correct / token_total) * 100 if token_total > 0 else 0
                    substring_accuracy = (substring_correct / substring_total) * 100 if substring_total > 0 else 0
                    edit_distance_accuracy_result = (edit_distance_score / edit_distance_total) * 100 if edit_distance_total > 0 else 0

                    print(f"📊 {file} 이미지 정확도")
                    print(f"   🔹 토큰 단위 정확도: {token_accuracy:.2f}%")
                    print(f"   🔹 부분 문자열 정확도: {substring_accuracy:.2f}%")
                    print(f"   🔹 편집 거리 기반 정확도: {edit_distance_accuracy_result:.2f}%")

                # 결과 누적
                total_token_correct += token_correct
                total_token_total += token_total
                total_substring_correct += substring_correct
                total_substring_total += substring_total
                total_edit_distance_score += edit_distance_score
                total_edit_distance_total += edit_distance_total

            except Exception as e:
                print(f"❌ 파일 처리 중 오류 발생: {file} - {str(e)}")
                skipped_files.append(file)

    # 최종 평균 정확도 출력
    final_token_accuracy = (total_token_correct / total_token_total) * 100 if total_token_total > 0 else 0
    final_substring_accuracy = (total_substring_correct / total_substring_total) * 100 if total_substring_total > 0 else 0
    final_edit_distance_accuracy = (total_edit_distance_score / total_edit_distance_total) * 100 if total_edit_distance_total > 0 else 0

    print("\n📊 최종 평균 정확도 (전체 이미지 평가)")
    print(f"   🔹 토큰 단위 평균 정확도: {final_token_accuracy:.2f}%")
    print(f"   🔹 부분 문자열 평균 정확도: {final_substring_accuracy:.2f}%")
    print(f"   🔹 편집 거리 기반 평균 정확도: {final_edit_distance_accuracy:.2f}%")

# 평가 실행
evaluate_paddle_ocr(IMAGE_PATH, GROUND_TRUTH_PATH)
