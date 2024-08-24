import json
from PIL import Image
from tqdm import tqdm
import numpy as np

data = json.load(open("tmp/para_info.json", "r"))
for k, v in tqdm(data.items()):
    try:
        img = Image.open(f"tmp/i_s/{k}.png")
        w, h = img.size
        new_w = w/v['ratio']
        new_img = Image.new("RGB", (int(new_w), int(h)))
        for i in range(np.ceil(1/v['ratio']).astype(int)):
            new_img.paste(img, (int(i*w), 0))
        new_img.save(f"tmp/i_s/{k}.png")
    except:
        print(f"Error in {k}")

print("crops transformed.")