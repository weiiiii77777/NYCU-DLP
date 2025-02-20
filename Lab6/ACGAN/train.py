import argparse
import os
import numpy as np
import math
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch

import ACGAN
from dataset import iclevrDataset
from evaluator import evaluation_model
import torchvision.models as models

class classification_model():
    def __init__(self):
        #modify the path to your own path
        checkpoint = torch.load('./checkpoint.pth', map_location={'cuda:0':'cuda:1'})
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512,24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.to('cuda:0')
        self.resnet18.eval()
        self.classnum = 24

    def eval(self, images):
        with torch.no_grad():
            #your image shape should be (batch, 3, 64, 64)
            out = self.resnet18(images)
            
            return out


# custom weights initialization called on Generator and Discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--epoch", type=int, default=300, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument('--ngf', type=int, default=300, help="feature channels of generator")
    parser.add_argument('--ndf', type=int, default=64, help="feature channels of discriminator")
    parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--nc", type=int, default=300, help="number of condition embedding dim")
    parser.add_argument('--proportion', type=int, default=4)
    parser.add_argument('--path', default='./save_model/6_aux10_pro4', help='folder to output images and model checkpoints')
    parser.add_argument('--aux_weight', type=int, default=10, help='path to resume model weight')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    device = args.device
    print('ACGAN')

    # Loss functions
    dis_criterion = nn.BCELoss().to(device)
    aux_criterion = nn.BCELoss().to(device) 
    class_criterion = nn.BCELoss().to(device)

    # Initialize generator and discriminator
    G = ACGAN.Generator(args).to(device)
    D = ACGAN.Discriminator(args).to(device)

    G.apply(weights_init) 
    D.apply(weights_init)

    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Dataloader
    train_loader = DataLoader(
        iclevrDataset(mode='train', root='../dataset'),
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        iclevrDataset(mode='test', root='../dataset'),
        batch_size=args.batch_size,
        shuffle=False
    )

    # Initilaize evaluator
    evaluator = evaluation_model()
    class_evaluator = classification_model()
    loss_g = []
    loss_d = []
    test_acc = []
    best_acc = 0
    for epoch in range(1, args.epoch + 1):
        epoch_loss_d = 0
        epoch_loss_g = 0
        D.train()
        G.train()
        for i, (image, cond) in enumerate(train_loader):
            ###################################################################
            # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z))) #
            ###################################################################
            optimizer_D.zero_grad()
            # train with real
            real_image = image.to(device)
            cond = cond.to(device)
            batch_size = image.size(0)
            # Use soft and noisy labels [0.7, 1.0]
            real_label = ((1.0 - 0.7) * torch.rand(batch_size) + 0.7).to(device)
            aux_label = cond
            # train with fake
            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_image = G(noise, aux_label)
            # Use soft and noisy labels [0.0, 0.3]
            fake_label = ((0.3 - 0.0) * torch.rand(batch_size) + 0.0).to(device)
            # occasionally flip the labels when training the discriminator
            if random.random() < 0.1:
                real_label, fake_label = fake_label, real_label
            dis_output, aux_output = D(real_image)
            dis_errD_real = dis_criterion(dis_output, real_label)
            aux_errD_real = aux_criterion(aux_output, aux_label)
            errD_real = dis_errD_real + args.aux_weight * aux_errD_real
            errD_real.backward()
            D_x = dis_output.mean().item()
            dis_output, aux_output = D(fake_image.detach())
            dis_errD_fake = dis_criterion(dis_output, fake_label)
            aux_errD_fake = aux_criterion(aux_output, aux_label)
            errD_fake = dis_errD_fake + args.aux_weight * aux_errD_fake
            errD_fake.backward()
            D_G_z1 = dis_output.mean().item()
            errD = errD_real + errD_fake
            optimizer_D.step()
            epoch_loss_d += errD.item()
            ###############################################
            # (2) Update Generator: maximize log(D(G(z))) #
            ###############################################
            for _ in range(args.dis_iters):
                optimizer_G.zero_grad()
                noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
                fake_image = G(noise, aux_label)
                # fake labels are real for generator cost
                generator_label = torch.ones(batch_size).to(device)  
                dis_output, aux_output = D(fake_image)
                dis_errG = dis_criterion(dis_output, generator_label)
                aux_errG = aux_criterion(aux_output, aux_label)
                errG = dis_errG + args.aux_weight * aux_errG
                errG.backward()
                D_G_z2 = dis_output.mean().item()
                optimizer_G.step()
                epoch_loss_g += errG.item()

        epoch_loss_d /= len(train_loader)
        epoch_loss_g /= len(train_loader)
        epoch_loss_d = round(epoch_loss_d, 5)
        epoch_loss_g = round(epoch_loss_g, 5)
        loss_d.append(epoch_loss_d)
        loss_g.append(epoch_loss_g)
        D.eval()
        G.eval()
        with torch.no_grad():
            for cond in tqdm(test_loader): 
                cond = cond.to(device)
                batch_size = cond.size(0)
                noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
                fake_image = G(noise, cond)
         
                acc = evaluator.eval(fake_image, cond)
                acc = round(acc, 5)
                test_acc.append(acc)

                if (acc > best_acc) or (acc >= 0.7):
                    if acc > best_acc:
                        best_acc = acc
                    torch.save(G.state_dict(), args.path + '/G/Generator_' + str(epoch) + '.pth')
                    torch.save(D.state_dict(), args.path + '/D/Discriminator_' + str(epoch) + '.pth')

        print('Epoch: '+str(epoch)+' Loss_D: '+ str(epoch_loss_d)+' Loss_G: '+str(epoch_loss_g)+' Accuracy: '+str(acc))

    f = open(os.path.join(args.path, f"record.txt"), 'w')
    print('D Loss', file=f)
    for i in range(args.epoch):
        print(loss_d[i],',', end=' ', file=f)
        if i%10 == 9:
            print(file=f)
    print(file=f)
    print('G Loss', file=f)
    for i in range(args.epoch):
        print(loss_g[i],',', end=' ', file=f)
        if i%10 == 9:
            print(file=f)
    print(file=f)
    print('Accuracy', file=f) 
    for i in range(args.epoch):
        print(test_acc[i],',', end=' ', file=f)
        if i%10 == 9:
            print(file=f)
    print(file=f)
    f.close()