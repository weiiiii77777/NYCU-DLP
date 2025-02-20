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

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--network', type=str, help='res34unet or unet')
    parser.add_argument('--model', help='saved model name')
    parser.add_argument('--data_path', type=str, default='../dataset/oxford-iiit-pet/',help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = oxford_pet.load_dataset(args.data_path, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)
    print("> Found %d images..." % (len(test_data))) 
    if args.network == 'res34unet' :
        model = Res34UNet().to(device)
    elif args.network == 'unet' :
        model = UNet().to(device)
    model.load_state_dict(torch.load('../saved_models/' + args.network + '/' + args.model))
    # model.load_state_dict(torch.load('../saved_models/' + args.model))
    val_dice = 0
    model.eval()
    with torch.no_grad() :  
        for i, data in enumerate(test_loader):
            img = data['image'].to(device).to(torch.float)
            mask = data['mask'].to(device)
            outputs = model(img)
            val_dice += float(utils.dice_score(outputs, mask))
            if i == 0 :
                utils.show_predict(args.network, data['image'][0], data['mask'][0], outputs[0])
    print(args.network, ' test dice score : ', round(val_dice/len(test_loader), 4))
