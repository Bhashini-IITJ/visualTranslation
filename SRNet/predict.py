import os
import argparse
import torch
from tqdm import tqdm
from model import Generator, Discriminator, Vgg19
from utils import *
from datagen import datagen_srnet, example_dataset, To_tensor
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

def run_srnet(input_dir, save_dir, checkpoint_path, learning_rate, beta1, beta2, file_mapping, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_log('Model compiling start.', content_color=PrintColor['yellow'])

    G = Generator(in_channels=3).to(device)
    D1 = Discriminator(in_channels=6).to(device)
    D2 = Discriminator(in_channels=6).to(device)
    vgg_features = Vgg19().to(device)

    G_solver = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, beta2))
    D1_solver = torch.optim.Adam(D1.parameters(), lr=learning_rate, betas=(beta1, beta2))
    D2_solver = torch.optim.Adam(D2.parameters(), lr=learning_rate, betas=(beta1, beta2))

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    G.load_state_dict(checkpoint['generator'])
    D1.load_state_dict(checkpoint['discriminator1'])
    D2.load_state_dict(checkpoint['discriminator2'])
    G_solver.load_state_dict(checkpoint['g_optimizer'])
    D1_solver.load_state_dict(checkpoint['d1_optimizer'])
    D2_solver.load_state_dict(checkpoint['d2_optimizer'])

    trfms = To_tensor()
    example_data = example_dataset(data_dir=input_dir, transform=trfms)
    example_loader = DataLoader(dataset=example_data, batch_size=1, shuffle=False)
    example_iter = iter(example_loader)

    print_log('Model compiled.', content_color=PrintColor['yellow'])
    print_log('Predicting', content_color=PrintColor['yellow'])

    G.eval()
    D1.eval()
    D2.eval()

    with torch.no_grad():
        for step in tqdm(range(len(example_data))):
            try:
                inp = next(example_iter)
            except StopIteration:
                example_iter = iter(example_loader)
                inp = next(example_iter)

            i_t = inp[0].to(device)
            i_s = inp[1].to(device)
            name = str(inp[2][0])

            # Extract numeric part of name and lookup original name
            try:
                numeric_name = int(name.split('_')[0])
            except ValueError:
                numeric_name = None

            original_name = file_mapping.get(numeric_name, name)

            o_sk, o_t, o_b, o_f = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))

            o_sk = o_sk.squeeze(0).detach().to('cpu')
            o_t = o_t.squeeze(0).detach().to('cpu')
            o_b = o_b.squeeze(0).detach().to('cpu')
            o_f = o_f.squeeze(0).detach().to('cpu')

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            o_sk = F.to_pil_image(o_sk)
            o_t = F.to_pil_image((o_t + 1) / 2)
            o_b = F.to_pil_image((o_b + 1) / 2)
            o_f = F.to_pil_image((o_f + 1) / 2)

            input_size = (i_s.shape[2], i_s.shape[3])
            o_f = F.resize(o_f, input_size)

            # Save output using the original name
            o_f.save(os.path.join(save_dir, f"{original_name}.png"))

    print_log('Predicting finished.', content_color=PrintColor['yellow'])
