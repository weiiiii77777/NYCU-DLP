import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        super(kl_annealing, self).__init__()
        self.args = args
        self.cur_epoch = current_epoch
    def get_beta(self, epoch):
        # TODO
        if self.args.kl_anneal_type == 'No': 
            return 1
        else:
            beta = self.frange_cycle_linear(self.args.num_epoch, 0.0, 1.0,  self.args.kl_anneal_cycle, self.args.kl_anneal_ratio)
            return beta[epoch]
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=0.5):
        # TODO
        # https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
        if self.args.kl_anneal_type == 'Cyclical':
            pass
        elif self.args.kl_anneal_type == 'Monotonic':
            ratio = 0.2
            n_cycle = 1
        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        if ratio == 1 :
            ratio = (period - 1)/float(period)
        step = (stop-start)/(period*ratio)
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L 

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        train_loss = []
        valid_loss = []
        valid_psnr = []
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            epoch_loss = 0
            beta = self.kl_annealing.get_beta(self.current_epoch)
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, beta, adapt_TeacherForcing)
                epoch_loss += loss
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {:.2f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {:.2f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            epoch_loss /= len(train_loader)
            train_loss.append(epoch_loss.item())
            loss, psnr, frame_psnr, final_img = self.eval()
            valid_loss.append(loss.item())
            valid_psnr.append(psnr)
            print('Epoch : {}, Train_loss : {:.5f}, Valid_loss : {:.5f}, Valid_PSNR : {:.2f}'.format(self.current_epoch,epoch_loss,loss,psnr))
            if psnr > 20.0 :
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
        # self.plot_valid_loss_curve(valid_loss)
        # self.plot_psnr_curve(valid_psnr)  
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr, frame_psnr, final_img = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

        return loss, psnr, frame_psnr, final_img
    
    def training_one_step(self, img, label, beta, adapt_TeacherForcing):
        # TODO
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        loss = 0
        if adapt_TeacherForcing :
            for i in range(1, img.shape[0]) :
                frame = self.frame_transformation(img[i])
                pose = self.label_transformation(label[i])
                z, mu, logvar = self.Gaussian_Predictor(frame, pose)
                pic = self.frame_transformation(img[i-1]) 
                gen_pic = self.Decoder_Fusion(pic, pose, z)
                gen_pic = self.Generator(gen_pic) 
                loss += beta * kl_criterion(mu, logvar, img.shape[1])
                loss += self.mse_criterion(gen_pic, img[i])
            loss.backward()
            self.optimizer_step()  
            loss /= (img.shape[0] - 1) 
        else :
            pre_frame = img[0]
            for i in range(1, img.shape[0]) :
                frame = self.frame_transformation(img[i])
                pose = self.label_transformation(label[i])
                z, mu, logvar = self.Gaussian_Predictor(frame, pose)
                pic = self.frame_transformation(pre_frame) 
                gen_pic = self.Decoder_Fusion(pic, pose, z)
                gen_pic = self.Generator(gen_pic)
                pre_frame = gen_pic
                loss += beta * kl_criterion(mu, logvar, img.shape[1])
                loss += self.mse_criterion(gen_pic, img[i])
            loss.backward()
            self.optimizer_step() 
            loss /= (img.shape[0] - 1) 
        return loss
        
    def val_one_step(self, img, label):
        # TODO
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        loss = 0
        psnr = 0
        frame_psnr = []
        final_img = []
        pre_frame = img[0]
        for i in range(1, 630) :
            frame = self.frame_transformation(img[i])
            pose = self.label_transformation(label[i])
            z_temp, mu, logvar = self.Gaussian_Predictor(frame, pose)
            z = torch.randn_like(z_temp)
            feature = self.frame_transformation(pre_frame)
            gen_pic = self.Decoder_Fusion(feature, pose, z)
            gen_pic = self.Generator(gen_pic) 
            pre_frame = gen_pic
            loss += self.mse_criterion(gen_pic, img[i])
            x = Generate_PSNR(gen_pic, img[i]).item()
            psnr += x
            frame_psnr.append(x)
            if self.args.test:
                final_img.append(gen_pic[0]) 
        loss /= (img.shape[0] - 1)
        psnr /= (img.shape[0] - 1)
        return loss, psnr, frame_psnr, final_img
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        min = 0.0
        n = 5
        r = 10 % n
        if self.current_epoch >= self.tfr_sde and self.tfr > min and self.current_epoch % n == r:
            self.tfr -= self.tfr_d_step
        self.tfr = round(self.tfr, 2)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

    def plot_loss_curve(self, train_loss, valid_loss):
        idx = []
        for i in range(1, self.args.num_epoch+1) :  idx.append(i)
        label_1 = 'train_loss'
        label_2 = 'valid_loss'
        plt.figure(figsize=(6,5))
        plt.plot(idx, train_loss, label = label_1)    
        plt.plot(idx, valid_loss, label = label_2)
        plt.legend([label_1, label_2])
        plt.xlabel('Epoch')    
        plt.ylabel('Loss')
        plt.title(str(self.args.kl_anneal_type) + ' KL annealing Loss Curve')
        plt.savefig('./pic/6/loss_curve.jpg')

    def plot_valid_loss_curve(self, valid_loss):
        idx = []
        for i in range(1, self.args.num_epoch+1) :  idx.append(i)
        label_1 = 'valid_loss'
        plt.figure(figsize=(6,5)) 
        plt.plot(idx, valid_loss, label = label_1)
        plt.legend([label_1])
        plt.xlabel('Epoch')    
        plt.ylabel('Loss')
        plt.title(str(self.args.kl_anneal_type) + ' KL annealing Loss Curve')
        plt.savefig('./pic/13/loss_curve.jpg')
        path = './pic/13/valid_loss.txt'
        f = open(path, 'w')
        for i in range(self.args.num_epoch):
            print(round(valid_loss[i], 5),', ', end=' ', file=f)
            if i % 10 == 9 :
                print(file=f)
        f.close()

    def plot_psnr_curve(self, valid_psnr):
        idx = []
        for i in range(1, self.args.num_epoch+1) :  idx.append(i)
        label_1 = 'valid_psnr'
        plt.figure(figsize=(6,5))
        plt.plot(idx, valid_psnr, label = label_1)    
        plt.legend([label_1])
        plt.xlabel('Epoch')    
        plt.ylabel('PSNR')
        plt.title('PSNR Curve')
        plt.savefig('./pic/13/PSNR_curve.jpg')
        path = './pic/13/valid_psnr.txt'
        f = open(path, 'w')
        for i in range(self.args.num_epoch):
            print(round(valid_psnr[i], 4),', ', end=' ', file=f)
            if i % 10 == 9 :
                print(file=f)
        f.close()

    def plot_frame_psnr(self, frame_psnr):
        idx = []
        total = 0.0
        for i in range(1, 630) :  
            idx.append(i)
            total += frame_psnr[i-1]
        label_1 = 'Avg_PSNR : ' + str(round(total/629.0, 3))
        plt.figure(figsize=(6,5))
        plt.plot(idx, frame_psnr, label = label_1)    
        plt.legend([label_1])
        plt.xlabel('Frame index')    
        plt.ylabel('PSNR')
        plt.title('Per frame Quality (PSNR)')
        plt.savefig('./result/demo/frame_PSNR.jpg')

def main(args):
    print('GPU Fan 2')
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        loss, psnr, frame_psnr, final_img = model.eval()
        model.plot_frame_psnr(frame_psnr)
        model.make_gif(final_img, './result/demo/pred_val.gif')
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=3)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda:3")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=100,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.5,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
