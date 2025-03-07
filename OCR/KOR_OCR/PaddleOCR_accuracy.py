import os
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageOps
import json
import re
import Levenshtein
from kor_ocr import save_all_ocr_results_to_txt

# PaddleOCR 초기화
ocr = PaddleOCR(lang='korean', use_angle_cls=True, use_gpu=True)

# 데이터 경로 설정
IMAGE_PATH = r"E:\OCR\image"
GROUND_TRUTH_PATH = r"E:\OCR\correct"
OUTPUT_OCR_TXT_PATH = r"C:\Users\osh\visualTranslation\OCR\KOR_OCR\all_ocr_result.txt"
OUTPUT_CORRECT_TXT_PATH = r"C:\Users\osh\visualTranslation\OCR\KOR_OCR\all_correct_results.txt"

# OCR 정답에서 "xxx" 제거
def clean_ground_truth(text):
    return ' '.join([word for word in text.split() if word != "xxx"]).strip()

# 정답 파일에서 텍스트 로드 (모든 annotation의 텍스트를 결합)
def load_ground_truth_combined(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            annotations = data.get("annotations", [])
            # "xxx"인 항목은 제외하고 나머지 텍스트들을 결합합니다.
            texts = [ann.get("text", "") for ann in annotations if ann.get("text", "").lower() != "xxx"]
            combined_text = ' '.join(texts).strip()
            return clean_ground_truth(combined_text)
    except Exception:
        return None

# 이미지에서 가장 큰 문자 영역 추출 (ROI 선택)
def extract_largest_text_region(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("L")            # 흑백 변환
            img = ImageOps.invert(img)         # 색상 반전 (텍스트 강조)
            img_array = np.array(img)
            # 이진화
            threshold = 128
            binary_img = np.where(img_array > threshold, 255, 0).astype(np.uint8)
            # 윤곽선 검출
            non_zero_coords = np.column_stack(np.where(binary_img > 0))
            if non_zero_coords.shape[0] == 0:
                return img  # 원본 이미지 반환
            # 바운딩 박스 계산
            x_min, y_min = non_zero_coords.min(axis=0)
            x_max, y_max = non_zero_coords.max(axis=0)
            # ROI 추출 (문자 영역만 자르기)
            return img.crop((y_min, x_min, y_max, y_max))
    except Exception:
        return None

# 정확도 평가 함수들
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

# 평가 실행 (OCR 결과 추출 및 저장)
def evaluate_paddle_ocr(image_path, ground_truth_path):
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    all_results = {}  # OCR 결과 저장 딕셔너리

    total_images = 0   # 전체 이미지 개수
    ocr_images = 0     # OCR 결과가 나온 이미지 개수

    # 정확도 누적 변수
    total_token_correct = 0
    total_token_total = 0
    total_substring_correct = 0
    total_substring_total = 0
    total_edit_correct = 0
    total_edit_total = 0
    total_highest = 0  # 각 이미지의 최고 정확도 누적

    if not os.path.exists(image_path) or len(os.listdir(image_path)) == 0:
        print("❌ 처리할 이미지가 없습니다. 경로를 확인하세요.")
        return all_results

    for file in os.listdir(image_path):
        if file.lower().endswith(SUPPORTED_EXTENSIONS):
            total_images += 1
            try:
                image_file_path = os.path.join(image_path, file)
                json_file_path = os.path.join(ground_truth_path, os.path.splitext(file)[0] + '.json')

                # OCR 정답(ground truth)은 평가 시 combined 형태로 사용
                if os.path.exists(json_file_path):
                    ground_truth = load_ground_truth_combined(json_file_path)
                    if ground_truth is None:
                        continue
                else:
                    continue

                cropped_image = extract_largest_text_region(image_file_path)
                if cropped_image is None:
                    continue

                result = ocr.ocr(np.array(cropped_image), cls=True)
                if not result or not result[0]:
                    error_message = "OCR 결과가 비어 있습니다. (인식 실패 또는 이미지 품질 문제)"
                    print(f"📌 파일: {file}")
                    print(f"❌ [출력 오류 및 원인] : {error_message}\n")
                    continue

                ocr_images += 1
                paddle_text = ' '.join([line[1][0] for line in result[0]]).strip()

                print(f"📌 파일: {file}")
                print(f"🔍 OCR 결과: {paddle_text}")
                print(f"✅ OCR 정답: {ground_truth}")

                token_correct, token_total = token_based_accuracy(paddle_text, ground_truth)
                substring_correct, substring_total = substring_matching_accuracy(paddle_text, ground_truth)
                edit_correct, edit_total = edit_distance_accuracy(paddle_text, ground_truth)

                token_accuracy = (token_correct / token_total * 100) if token_total > 0 else 0
                substring_accuracy = (substring_correct / substring_total * 100) if substring_total > 0 else 0
                edit_accuracy = (edit_correct / edit_total * 100) if edit_total > 0 else 0

                highest_accuracy = max(token_accuracy, substring_accuracy, edit_accuracy)
                if highest_accuracy == token_accuracy:
                    highest_metric = "토큰 단위"
                elif highest_accuracy == substring_accuracy:
                    highest_metric = "부분 문자열"
                else:
                    highest_metric = "편집 거리 기반"
                total_highest += highest_accuracy

                print(f" : 토큰 단위 정확도: {token_accuracy:.2f}%")
                print(f" : 부분 문자열 정확도: {substring_accuracy:.2f}%")
                print(f" : 편집 거리 기반 정확도: {edit_accuracy:.2f}%")
                print(f"[가장 높은 정확도 기준 : '{highest_metric}' - {highest_accuracy:.2f}%]\n")

                total_token_correct += token_correct
                total_token_total += token_total
                total_substring_correct += substring_correct
                total_substring_total += substring_total
                total_edit_correct += edit_correct
                total_edit_total += edit_total

                # OCR 결과 저장
                for idx, detection in enumerate(result[0]):
                    try:
                        bbox_points = detection[0]
                        xs = [point[0] for point in bbox_points]
                        ys = [point[1] for point in bbox_points]
                        flat_bbox = [min(xs), min(ys), max(xs), max(ys)]
                        text = detection[1][0]
                        key = f"{os.path.splitext(file)[0]}_{idx}"
                        all_results[key] = {"txt": text, "bbox": flat_bbox}
                    except Exception as e:
                        print(f"검출 항목 처리 중 오류 발생 (인덱스 {idx}): {e}")
                        continue

            except Exception as e:
                print(f"오류 발생: {e}")
                continue

    overall_token_accuracy = (total_token_correct / total_token_total * 100) if total_token_total > 0 else 0
    overall_substring_accuracy = (total_substring_correct / total_substring_total * 100) if total_substring_total > 0 else 0
    overall_edit_accuracy = (total_edit_correct / total_edit_total * 100) if total_edit_total > 0 else 0
    overall_highest_average = (total_highest / ocr_images) if ocr_images > 0 else 0

    print("========================================================")
    print(f"[전체 이미지 개수: {total_images}개 / OCR 처리된 이미지 개수: {ocr_images}개]")
    print(f" : 토큰 단위 평균 정확도: {overall_token_accuracy:.2f}%")
    print(f" : 부분 문자열 평균 정확도: {overall_substring_accuracy:.2f}%")
    print(f" : 편집 거리 기반 평균 정확도: {overall_edit_accuracy:.2f}%")
    print(f"[최고 정확도 평균: {overall_highest_average:.2f}%]")
    print("========================================================")

    return all_results

# 정답을 저장하는 함수 (각 이미지에 대해 모든 annotation의 텍스트를 결합)
def save_all_correct_results_to_txt(ground_truth_path, output_file):
    correct_results = {}
    for file in os.listdir(ground_truth_path):
        if file.lower().endswith('.json'):
            filepath = os.path.join(ground_truth_path, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    annotations = data.get("annotations", [])
                    # "xxx" 항목은 제거하고 나머지 텍스트를 결합
                    texts = [ann.get("text", "") for ann in annotations if ann.get("text", "").lower() != "xxx"]
                    combined_text = ' '.join(texts).strip()
                    key = os.path.splitext(file)[0]
                    correct_results[key] = {"txt": combined_text}
            except Exception as e:
                print(f"파일 {file} 처리 중 오류 발생: {e}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(correct_results, f, ensure_ascii=False, indent=4)
    print(f"정답 파일이 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    results = evaluate_paddle_ocr(IMAGE_PATH, GROUND_TRUTH_PATH)
    if results:
        save_all_ocr_results_to_txt(results, OUTPUT_OCR_TXT_PATH)
        save_all_correct_results_to_txt(GROUND_TRUTH_PATH, OUTPUT_CORRECT_TXT_PATH)