import cv2
import json
import os
img2info = json.load(open("tmp/para_info.json",'r'))
os.makedirs("tmp/bg",exist_ok=True)
imgs = os.listdir("tmp/output_base")
imgs2id = {}
for img in imgs:
    imgs2id[img.split(".")[0]] = img
for img_id in img2info:
    try:
        x1, y1, x2, y2 = img2info[img_id]['bbox']
        img = cv2.imread(f"tmp/output_base/{imgs2id[img_id.split('_')[0]]}")
        img_bg = img[y1:y2,x1:x2]
        cv2.imwrite(f"tmp/bg/{img_id}.png",img_bg)
    except:
        print(img_id)