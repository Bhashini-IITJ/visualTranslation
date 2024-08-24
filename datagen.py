import os
from skimage import io
import cfg
import glob
import torch
from torch.utils.data import Dataset
import cv2

class datagen_srnet(Dataset):
    def __init__(self, cfg, torp = 'train'):
        if(torp == 'train'):
            self.data_dir = cfg.data_dir
            self.t_b_dir = cfg.t_b_dir
            self.batch_size = cfg.batch_size
            self.data_shape = cfg.data_shape
            self.name_list = os.listdir(os.path.join(self.data_dir, self.t_b_dir))
      
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        
        img_name = self.name_list[idx]

        i_t = cv2.imread(os.path.join(cfg.data_dir, cfg.i_t_dir, img_name))
        i_t = cv2.cvtColor(i_t, cv2.COLOR_BGR2RGB)
        i_t = cv2.resize(i_t, (128, 64))

        i_s = cv2.imread(os.path.join(cfg.data_dir, cfg.i_s_dir, img_name))
        i_s = cv2.cvtColor(i_s, cv2.COLOR_BGR2RGB)
        i_s = cv2.resize(i_s, (128, 64))

        t_sk = cv2.imread(os.path.join(cfg.data_dir, cfg.t_sk_dir, img_name))
        t_sk = cv2.cvtColor(t_sk, cv2.COLOR_BGR2GRAY)
        t_sk = cv2.resize(t_sk, (128, 64))

        t_t = cv2.imread(os.path.join(cfg.data_dir, cfg.t_t_dir, img_name))
        t_t = cv2.cvtColor(t_t, cv2.COLOR_BGR2RGB)
        t_t = cv2.resize(t_t, (128, 64))

        t_f = cv2.imread(os.path.join(cfg.data_dir, cfg.t_f_dir, img_name))
        t_f = cv2.cvtColor(t_f, cv2.COLOR_BGR2RGB)
        t_f = cv2.resize(t_f, (128, 64))
        
        return [i_t, i_s, t_sk, t_t, t_f]

class To_tensor(object):
    def __call__(self, sample):
        
        i_t, i_s = sample

        i_t = i_t.transpose((2, 0, 1)) /127.5 -1
        i_s = i_s.transpose((2, 0, 1)) /127.5 -1

        i_t = torch.from_numpy(i_t)
        i_s = torch.from_numpy(i_s)

        return (i_t.float(), i_s.float())
