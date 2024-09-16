import torch.nn as nn
import torch
import torch
from modules.network import get_network
from torchvision import transforms
import numpy as np
from tqdm import tqdm 
import os
import argparse
import pickle

from PIL import Image
import sys



class CONTRIQUE_model(nn.Module):
    # resnet50 architecture with projector
    def __init__(self, encoder, n_features, \
                 patch_dim = (2,2), normalize = True, projection_dim = 128):
        super(CONTRIQUE_model, self).__init__()

        self.normalize = normalize
        self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.n_features = n_features
        self.patch_dim = patch_dim

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool_patch = nn.AdaptiveAvgPool2d(patch_dim)

        # MLP for projector
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.BatchNorm1d(self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
        )

    def forward(self, x_i, x_j):
        # global features
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        # local features
        h_i_patch = self.avgpool_patch(h_i)
        h_j_patch = self.avgpool_patch(h_j)

        h_i_patch = h_i_patch.reshape(-1,self.n_features,\
                                      self.patch_dim[0]*self.patch_dim[1])

        h_j_patch = h_j_patch.reshape(-1,self.n_features,\
                                      self.patch_dim[0]*self.patch_dim[1])

        h_i_patch = torch.transpose(h_i_patch,2,1)
        h_i_patch = h_i_patch.reshape(-1, self.n_features)

        h_j_patch = torch.transpose(h_j_patch,2,1)
        h_j_patch = h_j_patch.reshape(-1, self.n_features)

        h_i = self.avgpool(h_i)
        h_j = self.avgpool(h_j)

        h_i = h_i.view(-1, self.n_features)
        h_j = h_j.view(-1, self.n_features)

        if self.normalize:
            h_i = nn.functional.normalize(h_i, dim=1)
            h_j = nn.functional.normalize(h_j, dim=1)

            h_i_patch = nn.functional.normalize(h_i_patch, dim=1)
            h_j_patch = nn.functional.normalize(h_j_patch, dim=1)

        # global projections
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        # local projections
        z_i_patch = self.projector(h_i_patch)
        z_j_patch = self.projector(h_j_patch)

        return z_i, z_j, z_i_patch, z_j_patch, h_i, h_j, h_i_patch, h_j_patch

def main(path):
    # load image
    model_path = "models/CONTRIQUE_checkpoint25.tar"
    device = torch.device('cuda')
    image = Image.open(path)

    # downscale image by 2
    sz = image.size
    image_2 = image.resize((sz[0] // 2, sz[1] // 2))

    # transform to tensor
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image_2 = transforms.ToTensor()(image_2).unsqueeze(0).to(device)

    # load CONTRIQUE Model
    encoder = get_network('resnet50', pretrained=False)
    model = CONTRIQUE_model( encoder, 2048)
    model.load_state_dict(torch.load("models/CONTRIQUE_checkpoint25.tar", map_location=device.type))
    model = model.to(device)

    # extract features
    model.eval()
    with torch.no_grad():
        _,_, _, _, model_feat, model_feat_2, _, _ = model(image, image_2)
    feat = np.hstack((model_feat.detach().cpu().numpy(),\
                                model_feat_2.detach().cpu().numpy()))

    # load regressor model
    regressor = pickle.load(open('models/CLIVE.save', 'rb'))
    score = regressor.predict(feat)[0]
    return score

import warnings
warnings.filterwarnings('ignore')

dir = sys.argv[1]


score = 0
count = 0
for path in tqdm(os.listdir(dir)):
    if('.jpg' in path or '.png' in path):
        score+=main(os.path.join(dir,path))
        count+=1

print(f"Average contrique of dir {dir} is {score/count}")



