import os
import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_loops")
args = parser.parse_args()
n = int(args.num_loops)

def process_sample(sample_dir):
    if not len(os.listdir(sample_dir)) == 13: # TODO
        return

    sample_idx = sample_dir.split('/')[-1]
    input_file_path = os.path.join(sample_dir, f't_sk_1_{sample_idx}.png')
    save_file_path = os.path.join(sample_dir, f't_sk_{sample_idx}.png')

    img = cv2.imread(input_file_path, 0)
    _, img = cv2.threshold(img, 127, 255, 0)

    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open_img)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    cv2.imwrite(save_file_path, skel)

def process_directory(parent_output_dir):
    sample_dirs = [os.path.join(parent_output_dir, subdir) for subdir in os.listdir(parent_output_dir)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_sample, sample_dirs)

if __name__ == "__main__":
    for i in tqdm(range(n)):
        parent_output_dir = f'./dataset/o{i}'
        process_directory(parent_output_dir)
