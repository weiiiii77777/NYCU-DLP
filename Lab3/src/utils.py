import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    # https://www.twblogs.net/a/5d4092aabd9eee51fbf9a608
    
    num = len(gt_mask)
    smooth = 1
    m1 = torch.round(torch.sigmoid(pred_mask))
    m1 = m1.view(num, -1)  
    m2 = gt_mask.view(num, -1)  
    intersection = m1 * m2
    score = (2 * intersection.sum() + smooth) / (m1.sum() + m2.sum() + smooth)
    return score

def show_predict(network, img, mask, outputs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_1 = mask.to(device)
    dice = round(float(dice_score(outputs, mask_1)), 4)
    plt.figure(figsize=(10,10)) 
    img = np.moveaxis(img.numpy(), 0, -1)
    img = Image.fromarray(img)
    plt.subplot(131)  
    plt.title(network, loc='left')
    plt.title('Score : ' + str(dice), loc='right')              
    plt.imshow(img)   
    mask = np.moveaxis(mask.numpy(), 0, -1)
    mask = np.reshape(mask,(256,256))
    plt.subplot(132) 
    plt.title('GT mask')                
    plt.imshow(mask) 
    outputs = torch.round(torch.sigmoid(outputs))
    outputs = outputs.to('cpu').numpy()
    outputs = np.moveaxis(outputs, 0, -1)
    outputs = np.reshape(outputs,(256,256))
    plt.subplot(133)
    plt.title('Predict mask')             
    plt.imshow(outputs)
    plt.savefig('./' + network + '_predict.jpg')

def plot_curve(network, train_dice, valid_dice):
    idx = []
    for i in range(1, 51) :  idx.append(i)
    label_1 = network + '_train_dice_score'
    label_2 = network + '_valid_dice_score'
    plt.figure(figsize=(6,5))
    plt.plot(idx, train_dice, label = label_1)    
    plt.plot(idx, valid_dice, label = label_2)
    plt.legend([label_1, label_2])
    plt.xlabel('Epoch')    
    plt.ylabel('Dice Score')
    plt.title('Dice Score Curve')
    plt.savefig('./' + network + '_curve.jpg')