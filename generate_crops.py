import cv2
import json
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
args = parser.parse_args()
file = "tmp/i_s_info.json"
folder = args.folder

i_s_info = json.load(open(file, "r"))
para_info = json.load(open("tmp/para_info.json", "r"))
img_names = os.listdir(folder)
img_info = {}
for img in img_names:
    img_info[img.split(".")[0]] = img

os.makedirs("tmp/i_s", exist_ok=True)

img_ids = para_info.keys()
for img_id in tqdm(img_ids):
    try:
        ref_id = para_info[img_id]['ref_i_s']
        x1, y1, x2, y2 = i_s_info[ref_id]['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img_name = ref_id.split("_")[0]
        img = cv2.imread(os.path.join(folder, img_info[img_name]))
        img = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join("tmp/i_s", f"{img_id}.png"), img)
    except:
        print(f"Error in {img_id}")

print("crops created.")
print(f"<<<<<파일 확인>>>>> generate_crops.py")
