import json

img2info = json.load(open("tmp/para_info.json", "r"))
for img_id in img2info.keys():
    img2info[img_id]['ref_i_s'] = img_id
    img2info[img_id]['ratio'] = 1.0