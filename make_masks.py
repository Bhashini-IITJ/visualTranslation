from PIL import Image, ImageDraw
import json
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
args = parser.parse_args()

img_path = args.folder
mask_path = "tmp/masks"
json_file = "tmp/i_s_info.json"

os.makedirs(mask_path,exist_ok = True)
with open(json_file,'r') as f:
    info = json.load(f)


for img_name in tqdm(os.listdir(img_path)):
    img = Image.open(os.path.join(img_path,img_name))
    mask = Image.new('L', (img.size[0], img.size[1]), 0)
    draw = ImageDraw.Draw(mask)
    for key in info:
        if(key.split("_")[0]==img_name.split(".")[0]):
            draw.rectangle(info[key]['bbox'],fill=255)

    mask.save(os.path.join(mask_path,img_name))        

print(f"<<<<<파일 확인>>>>> make_masks.py")
