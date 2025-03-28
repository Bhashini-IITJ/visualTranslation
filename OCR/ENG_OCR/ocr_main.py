import cv2
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ocr_model = PaddleOCR(lang='en')
# ocr_model = PaddleOCR(lang='korean')

img_folder = '/mnt/c/Users/T3Q/Desktop/Paddle OCR/PaddleOCR/data/data_photo'
img_list = [file for file in os.listdir(img_folder) if file.lower().endswith(('.jpg', '.png', '.jpeg'))]

all_results = []
font_path = os.path.join('PaddleOCR', 'doc', 'fonts', 'latin.ttf')

for img_name in tqdm(img_list, desc="🔍 이미지 처리 중"):
    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 이미지 로드 실패: {img_path}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = ocr_model.ocr(img_path)

    if result and result[0]:
        res = result[0]
        boxes = [r[0] for r in res]
        texts = [r[1][0] for r in res]
        scores = [round(float(r[1][1]), 4) for r in res]

        # 시각화
        annotated = draw_ocr(img_rgb, boxes, texts, scores)
        plt.figure(figsize=(12, 12))
        plt.imshow(annotated)
        plt.title(f"Detected Texts in {img_name}")
        plt.axis('off')
        plt.show()

        for text, score in zip(texts, scores):
            all_results.append({
                'Image': img_name,
                'Text': text,
                'Confidence': score
            })
    else:
        all_results.append({
            'Image': img_name,
            'Text': '[No text found]',
            'Confidence': 0.0
        })

df = pd.DataFrame(all_results)
excel_path = 'ocr_results.xlsx'
df.to_excel(excel_path, index=False)

print(f"\n✅ OCR 결과가 '{excel_path}' 파일로 저장되었습니다!")