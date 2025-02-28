import json
from scipy.interpolate import CubicSpline, interp1d
import numpy as np
import pandas as pd

data = json.load(open("tmp/para_info.json", "r"))

i_s = {}

img_ids = data.keys()
for img_id in img_ids:
    k = 0
    for p in range(len(data[img_id]["para"])): # 이미지에서 패러그래프의 모든 단어 리스트
        para_info = data[img_id]["para"][p]
        para_words = para_info['trans_txt'].split() 
        para_l = [len(t) for t in para_words]
        para_l = np.cumsum(para_l)/np.sum(para_l)
        trans_words_list = []
        p_l_ = para_info['l']
        p_l_ = np.cumsum(p_l_)/np.sum(p_l_) # 각 줄에 대한 누적 비율 계산
        loop_trans_words = []
        j = 0
        i = 0
        while i < len(para_words): #누적 비율 기준으로 단어들 그룹화(줄 생성)
            if para_l[i] > p_l_[j]:
                trans_words_list.append(loop_trans_words)
                loop_trans_words = []
                j += 1
            else:
                loop_trans_words.append(para_words[i])
                i += 1
        trans_words_list.append(loop_trans_words)
        for l in range(len(para_info["lines"])):
            line_info = para_info["lines"][l]
            l_ = line_info['l']
            l_ = np.cumsum(l_)/np.sum(l_)
            l_ = np.hstack([0, l_])
            xcs = CubicSpline(l_, line_info['x'])
            y1cs = CubicSpline(l_, line_info['y1'])
            y2cs = CubicSpline(l_, line_info['y2'])          
            trans_words = trans_words_list[l]
            trans_l_ = [len(t) for t in trans_words]
            trans_l = np.cumsum(trans_l_)/np.sum(trans_l_)
            trans_l = np.hstack([0, trans_l])
            new_x = xcs(trans_l)
            new_y1 = y1cs(trans_l)
            new_y2 = y2cs(trans_l)
            ref_list = line_info['word_crops']
            ref_list = list(pd.cut(trans_l[1:], l_, labels=ref_list))
            ref_l = list(pd.cut(trans_l[1:], l_, labels=para_info["lines"][l]["l"],ordered=False))
            for i in range(len(trans_words)):
                i_s[f"{img_id}_{k}"] = {
                    "ref_i_s": ref_list[i],
                    "bbox": [
                        int(new_x[i]),
                        int(new_y1[i]),
                        int(new_x[i+1]),
                        int(new_y2[i+1]),
                    ],
                    "trans_txt": trans_words[i],
                    "ratio": ref_l[i]/trans_l_[i],
                }
                k += 1
json.dump(i_s, open("tmp/para_info.json", "w"), indent=4)

print(f"<<<<<파일 확인>>>>> form_word_crops.py")