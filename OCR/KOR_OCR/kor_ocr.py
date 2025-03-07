import json

def save_all_ocr_results_to_txt(ocr_results_dict, output_txt_path):
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_results_dict, f, ensure_ascii=False, indent=4)
        print(f"OCR 결과가 {output_txt_path}에 저장되었습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")
