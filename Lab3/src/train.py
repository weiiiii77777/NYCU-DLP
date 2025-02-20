import argparse
import oxford_pet
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from models.unet import UNet
from models.resnet34_unet import Res34UNet
import utils
import evaluate

def train(args):
    # implement the training function here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.network == 'unet' :  
        model = UNet().to(device)
    elif args.network == 'res34unet' :
        model = Res34UNet().to(device)
    train_data = oxford_pet.load_dataset(args.data_path, 'train')
    valid_data = oxford_pet.load_dataset(args.data_path, 'valid')
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    print("> Found %d images..." % (len(train_data))) 
    print("> Found %d images..." % (len(valid_data)))  
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    best_dice = 0
    train_dice_list = []
    valid_dice_list = []
    for epoch in range(args.epochs) :
        model.train()
        dice = 0
        for i, data in enumerate(train_loader):
            img = data['image'].to(device).to(torch.float)
            gt = data['mask'].to(device)
            outputs = model(img)
            loss = criterion(outputs, gt)
            loss += (1 - float(utils.dice_score(outputs, gt)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dice += float(utils.dice_score(outputs, gt))
        print('Epoch:', epoch+1, ', Training dice score:', round(dice/len(train_loader), 4), end = ' ')
        train_dice_list.append(round(dice/len(train_loader), 4))
        val_dice = evaluate.evaluate(epoch, model, valid_loader, device)
        valid_dice_list.append(round(val_dice, 4))
        if val_dice > best_dice :
            best_dice = val_dice
            best_epoch = epoch+1
            torch.save(model.state_dict(), '../saved_models/'+args.network+'_best_model_'+str(epoch+1)+'.pth')

    utils.plot_curve(args.network, train_dice_list, valid_dice_list)
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet/",help='path of the input data')
    parser.add_argument('--network', type=str, help='res34unet or unet')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.0001, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)