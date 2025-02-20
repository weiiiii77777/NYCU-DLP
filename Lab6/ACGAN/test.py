'''
python test_ACGAN.py --resume ckpt_ACGAN_bs128_ngf300_ndf64_BCE100_update_dis_4 --test_file new_test.json
python test_ACGAN.py --resume ckpt_ACGAN_bs128_ngf300_ndf64_BCE100_update_dis_4
'''
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch
import ACGAN
from dataset import iclevrDataset
from tqdm import tqdm
import random
from evaluator import evaluation_model

def norm_img(imgs):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    new_imgs = normalize(imgs)
    return new_imgs


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument('--ngf', type=int, default=300, help="feature channels of generator")
    parser.add_argument('--ndf', type=int, default=64, help="feature channels of discriminator")
    parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--nc", type=int, default=300, help="number of condition embedding dim")
    parser.add_argument('--out_path', default='./result/4/', help='folder to output images and model checkpoints')
    parser.add_argument('--g_ckpt_path', default='./save_model/4_aux100_pro2/G/Generator_298.pth', help='path to resume model weight')
    parser.add_argument('--d_ckpt_path', default='./save_model/4_aux100_pro2/D/Discriminator_298.pth', help='path to resume model weight')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    device = 'cuda:2'

    # load generator and discriminator weights
    generator = ACGAN.Generator(args).to(device)
    discriminator = ACGAN.Discriminator(args).to(device)
    generator.load_state_dict(torch.load(args.g_ckpt_path))
    discriminator.load_state_dict(torch.load(args.d_ckpt_path))

    test_dataloader = DataLoader(
        iclevrDataset(mode='test', root='../dataset', test_file='test.json'),
        batch_size=args.batch_size,
        shuffle=False
    )
    new_test_dataloader = DataLoader(
        iclevrDataset(mode='test', root='../dataset', test_file='new_test.json'),
        batch_size=args.batch_size,
        shuffle=False
    )
    # Initilaize evaluator
    evaluator = evaluation_model()
    discriminator.eval()
    generator.eval()
    with torch.no_grad():
        for i, cond in enumerate(test_dataloader):
            cond = cond.to(device)
            batch_size = cond.size(0)
            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_image = generator(noise, cond)
            dis_output, aux_output = discriminator(fake_image)
            transform=transforms.Compose([
                transforms.Normalize((0, 0, 0), (1/0.5, 1/0.5, 1/0.5)),
                transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
            ])
            fake_image = transform(fake_image)                
            output = norm_img(fake_image)
            acc = evaluator.eval(output, cond)
            acc = round(acc, 3)
        save_image(fake_image.detach(), os.path.join(args.out_path + 'test/', f'{round(acc*1000)}.png'), normalize=False)                
        print(f'GAN test.json Accuracy: {acc}')

    with torch.no_grad():
        for i, cond in enumerate(new_test_dataloader):
            cond = cond.to(device)
            batch_size = cond.size(0)
            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_image = generator(noise, cond)
            dis_output, aux_output = discriminator(fake_image)
            transform=transforms.Compose([
                transforms.Normalize((0, 0, 0), (1/0.5, 1/0.5, 1/0.5)),
                transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
            ])
            fake_image = transform(fake_image)                
            fake_image = norm_img(fake_image)
            acc = evaluator.eval(output, cond)
            acc = round(acc, 3)
        save_image(fake_image.detach(), os.path.join(args.out_path + 'new_test/', f'{round(acc*1000)}.png'), normalize=False)                
        print(f'GAN new_test.json Accuracy: {acc}')