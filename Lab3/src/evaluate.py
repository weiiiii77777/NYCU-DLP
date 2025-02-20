import torch
import torch.nn as nn
import numpy as np
import utils

def evaluate(epoch, net, datas, device):
    # implement the evaluation function here
    val_dice = 0
    net.eval()
    with torch.no_grad() :  
        for data in datas:
            img = data['image'].to(device).to(torch.float)
            outputs = net(img)
            val_dice += float(utils.dice_score(outputs, data['mask'].to(device)))
    print(', Valid dice score:', round(val_dice/len(datas), 4))

    return val_dice / len(datas)