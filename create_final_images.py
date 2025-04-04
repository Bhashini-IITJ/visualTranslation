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
output_base_dir = "source_eng"

# 로드
img2info = json.load(open(para_info_path, "r"))
imgs = os.listdir(output_base_dir)

# 파일 이름 매핑 (ex: "100" → "100_0.png")
id2img = {}
for img in imgs:
    img_base = img.split(".")[0].split("_")[0]  # "100_0.png" → "100"
    id2img[img_base] = img

# 이미지에 붙이기
for img_id in img2info.keys():
    try:
        base_id = img_id.split("_")[0]
        img_path = os.path.join(output_base_dir, id2img[base_id])
        overlay_path = os.path.join(o_f_dir, f"{img_id}.png")

        img = Image.open(img_path).convert("RGB")
        img_crop = Image.open(overlay_path).convert("RGB")

        x1, y1, _, _ = img2info[img_id]['bbox']
        img.paste(img_crop, (x1, y1))

        img.save(img_path)

    except Exception as e:
        print(f"[FAILED] {img_id}: {e}")

# 최종 결과 옮기기
os.makedirs(args.output_folder, exist_ok=True)
os.system(f"mv {output_base_dir}/* {args.output_folder}/")
print(f"<<<<< 파일 생성 완료 >>>>> 결과 폴더: {args.output_folder}")
