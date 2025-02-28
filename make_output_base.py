from PIL import Image
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
args = parser.parse_args()

file = "tmp/i_s_info.json"
folder = args.folder
img2info = json.load(open(file,'r'))

imgs = os.listdir(folder)
img_id2img = {}
for img in imgs:
    img_id = img.split(".")[0]
    img_id2img[img_id] = img
os.system(f"cp -r {folder} tmp/output_base")
for img_id in img2info.keys():
    img_name = img_id.split("_")[0]
    img = Image.open(f"tmp/output_base/{img_id2img[img_name]}")
    bg = Image.open(f"tmp/steo/{img_id2img[img_name]}")
    bg = bg.crop(img2info[img_id]['bbox'])
    img.paste(bg, (int(img2info[img_id]['bbox'][0]), int(img2info[img_id]['bbox'][1])))
    img.save(f"tmp/output_base/{img_id2img[img_name]}")

print(f"<<<<<파일 확인>>>>> make_output_base.py")