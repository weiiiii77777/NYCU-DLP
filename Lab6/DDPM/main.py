from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from matplotlib import pyplot as plt
import torch
import numpy as np
from dataloader import iclevrDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import torch.nn as nn
from tqdm.auto import tqdm
import argparse
from Unet import Unet
from torch.utils.tensorboard import SummaryWriter
from evaluator import evaluation_model
import torchvision.transforms as transforms
import os
from diffusers.optimization import get_cosine_schedule_with_warmup

class ConditionlDDPM():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.epoch = args.epoch
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_train_timestamps = args.num_train_timestamps
        self.svae_root = args.save_root
        self.label_embeding_size = args.label_embeding_size
        
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timestamps, beta_schedule="squaredcos_cap_v2")
        self.noise_predicter = Unet(labels_num=24, embedding_label_size=self.label_embeding_size).to(self.device)
        self.eval_model = evaluation_model()
        
        self.train_dataset = iclevrDataset(root="../dataset/iclevr", mode="train")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.optimizer = torch.optim.Adam(self.noise_predicter.parameters(), lr=self.lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=len(self.train_loader) * self.epoch,
            num_cycles=50
        )
        
    def train(self):
        loss_criterion = nn.MSELoss()
        # training 
        train_loss = []
        test_acc = []
        bestacc = 0
        for epoch in range(1, self.epoch+1):
            epoch_loss = 0
            for x, y in tqdm(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                noise = torch.randn_like(x)
                timestamp = torch.randint(0, self.num_train_timestamps - 1, (x.shape[0], ), device=self.device).long()
                noise_x = self.noise_scheduler.add_noise(x, noise, timestamp)
                perd_noise = self.noise_predicter(noise_x, timestamp, y)
                
                loss = loss_criterion(perd_noise, noise)
                loss.backward()
                nn.utils.clip_grad_value_(self.noise_predicter.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.lr = self.lr_scheduler.get_last_lr()[0]
                epoch_loss += loss.item()
            epoch_loss /= len(self.train_loader)
            train_loss.append(epoch_loss)
            print('Epoch ' + str(epoch) + ' Loss: ' + str(epoch_loss))
            acc = self.evaluate(epoch, test_what="test")
            test_acc.append(acc)
            print('Epoch ' + str(epoch) + ' Accuracy: ' + str(acc))
            if acc > bestacc or acc > 0.7 :
                if acc > bestacc :
                    bestacc = acc
                self.save(os.path.join(self.args.ckpt_root, f"epoch={epoch}.ckpt"), epoch)

        f = open(os.path.join(self.args.ckpt_root, f"record.txt"), 'w')
        print('Loss', file=f)
        for i in range(args.epoch):
            print(train_loss[i],',', end=' ', file=f)
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

    def evaluate(self, test_what="test"):
        test_dataset = iclevrDataset(mode=f"{test_what}")
        test_dataloader = DataLoader(test_dataset, batch_size=32)
        for y in test_dataloader:
            y = y.to(self.device)
            x = torch.randn(32, 3, 64, 64).to(self.device)
            for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                with torch.no_grad():
                    pred_noise = self.noise_predicter(x, t, y)
                x = self.noise_scheduler.step(pred_noise, t, x).prev_sample
            acc = self.eval_model.eval(images=x.detach(), labels=y)
            acc = round(acc, 3)
            denormalized_x = (x.detach() / 2 + 0.5).clamp(0, 1)
            generated_grid_imgs = make_grid(denormalized_x)
            save_image(generated_grid_imgs, "./result/1/"+test_what+"/result_"+str(acc*1000)+".jpg")
        return acc
    
    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.noise_predicter = checkpoint["noise_predicter"]
            self.noise_scheduler = checkpoint["noise_scheduler"]
            self.optimizer = checkpoint["optimizer"]
            self.lr = checkpoint["lr"]
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint['last_epoch']
    
    def save(self, path, epoch):
        torch.save({
            "noise_predicter": self.noise_predicter,
            "noise_scheduler": self.noise_scheduler,
            "optimizer": self.optimizer,
            "lr"        : self.lr,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "last_epoch": epoch
        }, path)
        print(f"save ckpt to {path}")

    def progressive_generate_image(self):
        # label_one_hot = [0] * 24
        # label_one_hot[2] = 1
        # label_one_hot[19] = 1
        # label_one_hot[3] = 1
        # label_one_hot = torch.tensor(label_one_hot).to(self.device)
        # label_one_hot = torch.unsqueeze(label_one_hot, 0)
        # # breakpoint()
        # x = torch.randn(1, 3, 64, 64).to(self.device)
        # img_list = []
        # for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
        #     with torch.no_grad():
        #         pred_noise = self.noise_predicter(x, t, label_one_hot)
        #     x = self.noise_scheduler.step(pred_noise, t, x).prev_sample
        #     if(t % 100 == 0):
        #         denormalized_x = (x.detach() / 2 + 0.5).clamp(0, 1)
        #         save_image(denormalized_x, "./result/1/" + str(t) + ".jpg")
        #         img_list.append(denormalized_x)
        # grid_img = make_grid(torch.cat(img_list, dim=0), nrow=10)
        # save_image(grid_img, "./result/1/progressive_genrate_image.jpg")

        test_dataset = iclevrDataset(mode=f"denoise")
        test_dataloader = DataLoader(test_dataset, batch_size=1)
        for y in test_dataloader:
            y = y.to(self.device)
            x = torch.randn(1, 3, 64, 64).to(self.device)
            img_list = []
            for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                with torch.no_grad():
                    pred_noise = self.noise_predicter(x, t, y)
                x = self.noise_scheduler.step(pred_noise, t, x).prev_sample
                if(t % 100 == 0):
                    denormalized_x = (x.detach() / 2 + 0.5).clamp(0, 1)
                    save_image(denormalized_x, "./result/1/" + str(t) + ".jpg")
                    img_list.append(denormalized_x)
            grid_img = make_grid(torch.cat(img_list, dim=0), nrow=10)
            save_image(grid_img, "./result/1/progressive_genrate_image.jpg")

            
def main(args):
    conditionlDDPM = ConditionlDDPM(args)
    if args.test:
        conditionlDDPM.load_checkpoint()
        # print('DDPM test.json Accuracy : ', conditionlDDPM.evaluate(test_what="test"))
        # print('DDPM new_test.json Accuracy : ', conditionlDDPM.evaluate(test_what="new_test"))
        conditionlDDPM.progressive_generate_image()
    else:
        conditionlDDPM.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--num_train_timestamps', type=int, default=1000)
    parser.add_argument('--lr_warmup_steps', default=0, type=int)
    parser.add_argument('--save_root', type=str, default="eval")
    parser.add_argument('--label_embeding_size', type=int, default=4)
    # ckpt
    parser.add_argument('--ckpt_root', type=str, default="./save_model/1")
    parser.add_argument('--ckpt_path', type=str, default="./save_model/1/epoch=97.ckpt")
    
    args = parser.parse_args()
    main(args)