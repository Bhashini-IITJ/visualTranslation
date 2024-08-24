import os
from glob import glob
import cv2
import numpy as np

def create_final_image(txt_img, bg_img):
    h, w, _ = bg_img.shape
    txt_img = cv2.resize(txt_img, (w,h))
    gray_img = cv2.cvtColor(txt_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    corner_pixel = mask[0,0]
    if corner_pixel == 255:
        mask = cv2.bitwise_not(mask)
    radius = 1

    blurred_mask = cv2.GaussianBlur(mask, (radius*2+1, radius*2+1), 0)
    blurred_mask_float = blurred_mask.astype(np.float32) / 255.0
    txt_img_float = txt_img.astype(np.float32)
    bg_img_float = bg_img.astype(np.float32)
    blurred_mask_3c = cv2.merge([blurred_mask_float, blurred_mask_float, blurred_mask_float])
    composite = (txt_img_float * blurred_mask_3c) + (bg_img_float * (1 - blurred_mask_3c))
    composite = np.uint8(composite)
    return composite

if __name__ == '__main__':
    o_t_path = 'tmp/o_t'
    bg_path = 'tmp/bg'
    o_f_path = 'tmp/o_f'
    os.makedirs(o_f_path, exist_ok=True)
    img_ids = os.listdir(o_t_path)
    for img_id in img_ids:
        try:
            o_t_img = cv2.imread(os.path.join(o_t_path, img_id))
            bg_img = cv2.imread(os.path.join(bg_path, img_id))
            
            img = create_final_image(o_t_img, bg_img)
            save_path = os.path.join(o_f_path, img_id)
            cv2.imwrite(save_path, img)
        except:
            print(img_id)