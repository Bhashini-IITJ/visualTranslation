from PIL import Image
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type=str, required=True)
args = parser.parse_args()

# 경로 설정
para_info_path = "tmp/para_info.json"
o_f_dir = "tmp/o_f"
source_folder = "source_eng"

# 로드
img2info = json.load(open(para_info_path, "r"))

# 결과 폴더 생성
os.makedirs(args.output_folder, exist_ok=True)

# 캐시: 원본 이미지 하나씩만 로드하고 저장하기 위함
image_cache = {}

# 루프: 모든 o_f 붙이기
for img_id, info in img2info.items():
    try:
        base_id = img_id.split("_")[0]            # ex: "178_0" → "178"
        source_img_name = f"{base_id}.jpg"
        source_img_path = os.path.join(source_folder, source_img_name)

        if base_id not in image_cache:
            # 원본 이미지 처음 로드 시 → 복사해서 결과용 캐시 만들기
            img = Image.open(source_img_path).convert("RGB")
            image_cache[base_id] = img
        else:
            img = image_cache[base_id]

        # o_f crop 불러오기
        crop_path = os.path.join(o_f_dir, f"{img_id}.png")
        img_crop = Image.open(crop_path).convert("RGB")

        # bbox 영역으로 크기 맞추기
        x1, y1, x2, y2 = info["bbox"]
        box_w, box_h = x2 - x1, y2 - y1
        img_crop_resized = img_crop.resize((box_w, box_h))

        # 붙이기
        img.paste(img_crop_resized, (x1, y1))

    except Exception as e:
        print(f"[FAILED] {img_id}: {e}")

# 모든 이미지 저장
for base_id, img in image_cache.items():
    output_path = os.path.join(args.output_folder, f"{base_id}.png")
    img.save(output_path)

print(f"<<<<< 완료 >>>>> {args.output_folder}에 최종 이미지 저장됨.")
