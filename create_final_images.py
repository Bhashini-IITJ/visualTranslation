from PIL import Image
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type=str, required=True)
args = parser.parse_args()

img2info = json.load(open("tmp/para_info.json", "r"))
imgs = os.listdir("tmp/output_base")
id2img = {}
for img in imgs:
    id2img[img.split(".")[0]] = img

for img_id in img2info.keys():
    try:
        img = Image.open("tmp/output_base/"+id2img[img_id.split("_")[0]])
        img_crop = Image.open(f"tmp/o_f/{img_id}.png")
        x1, y1, _, _ = img2info[img_id]['bbox']
        img.paste(img_crop, (x1, y1))
        img.save(f"tmp/output_base/"+id2img[img_id.split("_")[0]])
    except:
        print("failed", img_id)

os.system(f"mv tmp/output_base {args.output_folder}")