import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cfg
from model_o_t_gen import Generator
import os
from skimage import io
import numpy as np
from datagen import To_tensor

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
G = Generator(in_channels = 3).to(device)

G.eval()
G.load_state_dict(torch.load(cfg.checkpoint)['generator'])

size = (128, 64)
def infer(i_s, i_t, size, model, path):
    tmfr = To_tensor()
    i_s = io.imread(i_s)
    if len(i_s.shape) == 2 or i_s.shape[2] == 1:
        i_s = np.repeat(i_s[:, :, np.newaxis], 3, axis=2)
    else:
        i_s = i_s[:, :, :3]
    orig_i_s_size = i_s.shape
    orig_i_s_size = (orig_i_s_size[1], orig_i_s_size[0])
    i_s = cv2.resize(i_s, size)

    i_t = io.imread(i_t)
    i_t = cv2.resize(i_t, size)
      
    i_t, i_s = tmfr([i_t, i_s])

    i_t = i_t.unsqueeze(0).to(device)
    i_s = i_s.unsqueeze(0).to(device)

    with torch.no_grad():
      o_sk, o_t, o_f = model(i_t, i_s, (i_t.shape[2], i_t.shape[3]))

    o_f = o_f.squeeze(0).detach().to('cpu').permute(1, 2, 0).numpy()
    o_f = 127.5*o_f + 127.5
    o_f = o_f.astype('uint8')
    o_f = cv2.resize(o_f, orig_i_s_size)

    o_t = o_t.squeeze(0).detach().to('cpu').permute(1, 2, 0).numpy()
    o_t = 127.5*o_t + 127.5
    o_t = o_t.astype('uint8')
    o_t = cv2.resize(o_t, orig_i_s_size)
    o_t = cv2.cvtColor(o_t, cv2.COLOR_BGR2GRAY) 
    _, o_t = cv2.threshold(o_t, 125, 255, cv2.THRESH_BINARY) 
    
    o_sk = o_sk.squeeze(0).detach().to('cpu').permute(1, 2, 0).numpy()
    o_sk = 255.0*o_sk
    o_sk = o_sk.astype('uint8')
    o_sk = cv2.resize(o_sk, orig_i_s_size)

    plt.imsave(path, o_f)

if __name__ == "__main__":
  save_dir = 'tmp/o_t'
  os.makedirs(save_dir, exist_ok=True)
  for img_name in tqdm(os.listdir('tmp/i_s'), desc="samples processed"):
    idx = img_name.split('.')[0]
    i_s_path = os.path.join('tmp/i_s', f'{idx}.png')
    i_t_path = os.path.join('tmp/i_t', f'{idx}.png')
    out_path_1 = os.path.join(save_dir, f'{idx}.png')
    try:
      infer(i_s_path, i_t_path, (128, 64), G, out_path_1)
    except:
      print(idx, "failed to generate")
      
  print("completed!")