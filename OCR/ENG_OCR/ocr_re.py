import pandas as pd
from difflib import SequenceMatcher

# [1] 파일 경로
ocr_path = '/mnt/c/Users/T3Q/Desktop/Paddle OCR/PaddleOCR/ocr_results.xlsx'
answer_path = '/mnt/c/Users/T3Q/Desktop/Paddle OCR/PaddleOCR/data/data_answer/answer_texts.xlsx'

# [2] 파일 불러오기
ocr_df = pd.read_excel(ocr_path)
answer_df = pd.read_excel(answer_path)

# [3] 이미지별로 텍스트 합치기
ocr_grouped = ocr_df.groupby('Image')['Text'].apply(lambda x: ' '.join(str(t).strip() for t in x)).reset_index()
answer_grouped = answer_df.groupby('Image')['Text'].apply(lambda x: ' '.join(str(t).strip() for t in x)).reset_index()

# [4] 병합 (공통 이미지 기준)
merged = pd.merge(ocr_grouped, answer_grouped, on='Image', how='outer', suffixes=('_OCR', '_GT'))

# [5] 유사도 계산
def compute_similarity(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return round(SequenceMatcher(None, a, b).ratio() * 100, 2)

merged['Similarity (%)'] = merged.apply(lambda row: compute_similarity(row['Text_OCR'], row['Text_GT']), axis=1)

# [6] 결과 저장
output_path = '/mnt/c/Users/T3Q/Desktop/Paddle OCR/PaddleOCR/ocr_comparison_results.xlsx'
merged.to_excel(output_path, index=False)

# [7] 출력
print(f"\n✅ 비교 완료! 결과 저장 위치:\n{output_path}")
print(f"📊 평균 정확도: {merged['Similarity (%)'].mean():.2f}%")
