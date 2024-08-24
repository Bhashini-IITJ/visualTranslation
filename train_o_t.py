
import numpy as np
import os
import torch
from utils import *
import cfg
from tqdm import tqdm
from model_o_t_gen import Generator, Discriminator, Vgg19
from loss import build_generator_loss, build_discriminator_loss
from datagen import datagen_srnet
from torch.utils.data import DataLoader


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def custom_collate(batch):
    
    i_t_batch, i_s_batch = [], []
    t_sk_batch, t_t_batch, t_b_batch, t_f_batch = [], [], [], []
    
    for items in batch:
        i_t, i_s, t_sk, t_t, t_b, t_f = items

        t_sk = np.expand_dims(t_sk, axis = -1) 
        
        i_t = i_t.transpose((2, 0, 1))
        i_s = i_s.transpose((2, 0, 1))
        t_sk = t_sk.transpose((2, 0, 1))
        t_t = t_t.transpose((2, 0, 1))
        t_b = t_b.transpose((2, 0, 1))
        t_f = t_f.transpose((2, 0, 1))
         
        i_t_batch.append(i_t) 
        i_s_batch.append(i_s)
        t_sk_batch.append(t_sk)
        t_t_batch.append(t_t) 
        t_b_batch.append(t_b) 
        t_f_batch.append(t_f)

    i_t_batch = np.stack(i_t_batch)
    i_s_batch = np.stack(i_s_batch)
    t_sk_batch = np.stack(t_sk_batch)
    t_t_batch = np.stack(t_t_batch)
    t_b_batch = np.stack(t_b_batch)
    t_f_batch = np.stack(t_f_batch)
    
    i_t_batch = torch.from_numpy(i_t_batch.astype(np.float32) / 127.5 - 1.) 
    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.) 
    t_sk_batch = torch.from_numpy(t_sk_batch.astype(np.float32) / 255.) 
    t_t_batch = torch.from_numpy(t_t_batch.astype(np.float32) / 127.5 - 1.) 
    t_b_batch = torch.from_numpy(t_b_batch.astype(np.float32) / 127.5 - 1.) 
    t_f_batch = torch.from_numpy(t_f_batch.astype(np.float32) / 127.5 - 1.) 
    
    return [i_t_batch, i_s_batch, t_sk_batch, t_t_batch, t_b_batch, t_f_batch]

def clip_grad(model):
    for h in model.parameters():
        h.data.clamp_(-0.01, 0.01)
 
def main():
    device = 'cuda'
    
    os.makedirs(cfg.checkpoint_savedir, exist_ok=True)
    
    print_log('Initializing SRNET', content_color = PrintColor['yellow'])
    
    train_data = datagen_srnet(cfg)    
    train_data = DataLoader(
                            dataset = train_data, 
                            batch_size = cfg.batch_size, 
                            shuffle = True, 
                            collate_fn=custom_collate, 
                            pin_memory = True, 
                            num_workers=2
                            )
    
    print_log('training start.', content_color = PrintColor['yellow'])
        
    G = Generator(in_channels = 3).to(device)
    
    D = Discriminator(in_channels = 6).to(device)
        
    vgg_features = Vgg19().to(device)
        
    G_solver = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    D_solver = torch.optim.Adam(D.parameters(), lr=cfg.learning_rate, betas = (cfg.beta1, cfg.beta2))
    
    try:
      checkpoint = torch.load(cfg.ckpt_path)
      G.load_state_dict(checkpoint['generator'])
      D.load_state_dict(checkpoint['discriminator'])
      G_solver.load_state_dict(checkpoint['g_optimizer'])
      D_solver.load_state_dict(checkpoint['d_optimizer'])
      
      print('Resuming after loading...')
    except FileNotFoundError:

      print('checkpoint not found')
      pass 
    

    requires_grad(G, False)
    requires_grad(D, True)
    
    trainiter = iter(train_data)
    K = torch.nn.ZeroPad2d((0, 1, 1, 0))

    for step in tqdm(range(cfg.max_iter)):
        D_solver.zero_grad()
        
        if ((step+1) % cfg.save_ckpt_interval == 0):
            
            torch.save(
                {
                    'generator': G.state_dict(),
                    'discriminator': D.state_dict(),
                    'g_optimizer': G_solver.state_dict(),
                    'd_optimizer': D_solver.state_dict(),
                },
                cfg.checkpoint_savedir+f'train_step-{step+1}.model',
            )
        
        try:

          i_t, i_s, t_sk, t_t, t_f = next(trainiter)

        except StopIteration:

          trainiter = iter(train_data)
          i_t, i_s, t_sk, t_t, t_f = next(trainiter)
                
        i_t = i_t.to(device)
        i_s = i_s.to(device)
        t_sk = t_sk.to(device)
        t_t = t_t.to(device)
        t_f = t_t.to(device)
        labels = [t_sk, t_t, t_f]
        o_sk, o_t, o_f = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
        o_sk = K(o_sk)
        o_t = K(o_t)
        o_f = K(o_f)
        
        i_df_true = torch.cat((t_f, i_t), dim = 1)
        i_df_pred = torch.cat((o_f, i_t), dim = 1)
        
        o_df_true = D(i_df_true)
        o_df_pred = D(i_df_pred)
        
        i_vgg = torch.cat((t_f, o_f), dim = 0)
        
        out_vgg = vgg_features(i_vgg)
        
        df_loss = build_discriminator_loss(o_df_true, o_df_pred)

        df_loss.backward()
        D_solver.step()
        
        clip_grad(D)
        
        
        if ((step+1) % 2 == 0):
            
            requires_grad(G, True)

            requires_grad(D, False)
            
            G_solver.zero_grad()
            
            o_sk, o_t, o_f = G(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
            
            o_sk = K(o_sk)
            o_t = K(o_t)
            o_f = K(o_f)

            i_df_true = torch.cat((t_f, i_t), dim = 1)
            i_df_pred = torch.cat((o_f, i_t), dim = 1)

            o_df_pred = D(i_df_pred)

            i_vgg = torch.cat((t_f, o_f), dim = 0)

            out_vgg = vgg_features(i_vgg)
            
            out_g = [o_sk, o_t, o_f]

            out_d = [o_df_pred]
        
            g_loss, _ = build_generator_loss(out_g, out_d, out_vgg, labels)    
                
            g_loss.backward()
            
            G_solver.step()
                        
            requires_grad(G, False)

            requires_grad(D, True)
            
        if ((step+1) % cfg.write_log_interval == 0):
            
            print('Iter: {}/{} | Gen: {} | D_fus: {}'.format(step+1, cfg.max_iter, g_loss.item(), df_loss.item()))

if __name__ == '__main__':
    import warnings
    warnings.simplefilter("ignore")
    main()
