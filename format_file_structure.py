import json
import os
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--num_loops")
args = parser.parse_args()
n = int(args.num_loops)
parent_dir = './dataset/'

i_s_path = os.path.join(parent_dir, 'i_s')
i_t_path = os.path.join(parent_dir, 'i_t')
t_f_path = os.path.join(parent_dir, 't_f')
t_sk_path = os.path.join(parent_dir, 't_sk')
t_t_path = os.path.join(parent_dir, 't_t')

os.makedirs(i_s_path, exist_ok=True)
os.makedirs(i_t_path, exist_ok=True)
os.makedirs(t_f_path, exist_ok=True)
os.makedirs(t_sk_path, exist_ok=True)
os.makedirs(t_t_path, exist_ok=True)

global_sample_count = 0 
for i in range(n):
    split_dir = os.path.join(parent_dir, f'o{i}')
    file_dirs = [os.path.join(split_dir, file_dir) for file_dir in os.listdir(split_dir)]
    num_samples = len(file_dirs)
    for file_dir in tqdm(file_dirs, desc=f'moving o{i} files'):
        if len(os.listdir(file_dir)) != 14:
            continue
        idx = file_dir.split('/')[-1]
        os.system(f"mv {os.path.join(file_dir, f'i_s_{idx}.png')} {os.path.join(i_s_path, f'{global_sample_count}.png')}")
        os.system(f"mv {os.path.join(file_dir, f'i_t_{idx}.png')} {os.path.join(i_t_path, f'{global_sample_count}.png')}")
        os.system(f"mv {os.path.join(file_dir, f't_f_{idx}.png')} {os.path.join(t_f_path, f'{global_sample_count}.png')}")
        os.system(f"mv {os.path.join(file_dir, f't_sk_{idx}.png')} {os.path.join(t_sk_path, f'{global_sample_count}.png')}")
        os.system(f"mv {os.path.join(file_dir, f't_t_{idx}.png')} {os.path.join(t_t_path, f'{global_sample_count}.png')}")
        global_sample_count += 1
