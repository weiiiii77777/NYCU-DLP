import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch}")
        for i, img in enumerate(pbar):
            self.optim.zero_grad()
            img = img.to(args.device)
            logits, gt = self.model(img)
            # logits (batch_size, h*w, 1025)
            # gt (batch_size, h*w)
            logits = logits.reshape(-1, logits.shape[-1])
            # logits (batch_size*h*w, 1025)
            gt = gt.reshape(-1)
            # gt (batch_size * h*w)
            loss = F.cross_entropy(logits, gt)
            epoch_loss += loss
            loss.backward()
            self.optim.step(self.scheduler)
            pbar.set_postfix_str(f'Loss: {loss:.3f}')
        epoch_loss /= len(train_loader)
        return epoch_loss

    def eval_one_epoch(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            epoch_loss = 0
            pbar = tqdm(val_loader, total=len(val_loader))
            pbar.set_description(f"Validation ")
            for i, img in enumerate(pbar):
                img = img.to(args.device)
                logits, gt = self.model(img)
                logits = logits.reshape(-1, logits.size(-1))
                gt = gt.reshape(-1)
                loss = F.cross_entropy(logits, gt)
                epoch_loss += loss
                pbar.set_postfix_str(f'Loss: {loss:.3f}')
                
        epoch_loss /= len(val_loader)
        return epoch_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=args.learning_rate)
        scheduler = None
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./save_model/9/', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(default: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    print("start training")
    train_loss = []
    valid_loss = []
    for epoch in range(args.start_from_epoch, args.epochs+1):
        epoch_train_loss = train_transformer.train_one_epoch(train_loader)
        epoch_valid_loss = train_transformer.eval_one_epoch(val_loader)

        train_loss.append(epoch_train_loss.item())
        valid_loss.append(epoch_valid_loss.item())

        print(f"Epoch {epoch} , Train loss : {epoch_train_loss.item()}, Valid loss : {epoch_valid_loss.item()}")
        torch.save(train_transformer.model.transformer.state_dict(), os.path.join(args.checkpoint_path, f"epoch_{epoch}.pt"))
    
    # f = open(os.path.join(args.checkpoint_path, f"loss.txt"), 'w')
    # print('Train Loss', file=f)
    # for i in range(args.epochs):
    #     print(round(train_loss[i], 5),', ', end=' ', file=f)
    # print(file=f)
    # print('Valid Loss', file=f)
    # for i in range(args.epochs):
    #     print(round(valid_loss[i], 5),', ', end=' ', file=f)
    # print(file=f)
    # f.close()