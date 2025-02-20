import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils import data
import random

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('./dataset/train.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    elif mode == 'test':
        df = pd.read_csv('./dataset/test.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    elif mode == 'valid':
        df = pd.read_csv('./dataset/valid.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    

class BufferflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        # step 1.
        img = Image.open(str(self.root) + str(self.img_name[index]))

        # step 2.
        label = self.label[index]

        # step3. random flipping & random rotation
        if self.mode == "train" :
            flipping = random.random()
            if(flipping < 0.5) :
                if(flipping < 0.25) :
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                else :
                    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            rotation = random.random()
            if(rotation < 0.45) :
                if(rotation < 0.15) :
                    img = img.transpose(Image.Transpose.ROTATE_90)
                elif(rotation < 0.3) :
                    img = img.transpose(Image.Transpose.ROTATE_180)
                else :
                    img = img.transpose(Image.Transpose.ROTATE_270)

        # step 3. jpg -> np.array -> tensor
        img = np.array(img)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.Tensor(img)

        # step 4.
        return img, label