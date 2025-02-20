import torch
import torch.nn as nn
from torch import autograd

# https://blog.csdn.net/weixin_43977304/article/details/121497425 
# Implement your ResNet34_UNet model here

class BTNK1(nn.Module):
    def __init__(self, input_size, output_size, s):
        super(BTNK1, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_size, int(output_size/4), kernel_size=1, stride=s),
            nn.BatchNorm2d(int(output_size/4)),
            nn.ReLU(),
            nn.Conv2d(int(output_size/4), output_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(output_size),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=1, stride=s),
            nn.BatchNorm2d(output_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x) :
        fx = self.feature(x)
        x = self.shortcut(x)
        x = fx + x
        x = self.relu(x)
        return x
    
class BTNK2(nn.Module):
    def __init__(self, size):
        super(BTNK2, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(size, int(size/4), kernel_size=1, stride=1),
            nn.BatchNorm2d(int(size/4)),
            nn.ReLU(),
            nn.Conv2d(int(size/4), size, kernel_size=1, stride=1),
            nn.BatchNorm2d(size),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(size, size, kernel_size=1, stride=1),
            nn.BatchNorm2d(size),
        )
        self.relu = nn.ReLU()

    def forward(self, x) :
        fx = self.feature(x)
        x = self.shortcut(x)
        x = fx + x
        x = self.relu(x)
        return x

class convBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(convBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
        )

    def forward(self, x) :
        x = self.conv(x)
        return x

class Res34UNet(nn.Module):
    def __init__(self):
        super(Res34UNet, self).__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(
            BTNK2(64),
            BTNK2(64),
            BTNK2(64),
        )
        self.stage2 = nn.Sequential(
            BTNK1(64, 128, 2),
            BTNK2(128),
            BTNK2(128),
            BTNK2(128),
        )
        self.stage3 = nn.Sequential(
            BTNK1(128, 256, 2),
            BTNK2(256),
            BTNK2(256),
            BTNK2(256),
            BTNK2(256),
            BTNK2(256),
        )
        self.stage4 = nn.Sequential(
            BTNK1(256, 512, 2),
            BTNK2(512),
            BTNK2(512),
        )
        self.stage5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.stage6 = nn.Sequential(
            convBlock(768, 384),
            nn.ConvTranspose2d(384, 32, kernel_size=2, stride=2),
        )
        self.stage7 = nn.Sequential(
            convBlock(288, 144),
            nn.ConvTranspose2d(144, 32, kernel_size=2, stride=2),
        )
        self.stage8 = nn.Sequential(
            convBlock(160, 80),
            nn.ConvTranspose2d(80, 32, kernel_size=2, stride=2),
        )
        self.stage9 = nn.Sequential(
            convBlock(96, 48),
            nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2),
        )
        self.stage10 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        
        x = self.stage0(x) # layer_1
        x = self.stage1(x) # layer_2 - layer_7
        c1 = x
        x = self.stage2(x) # layer_8 - layer_15
        c2 = x
        x = self.stage3(x) # layer_16 - layer_27
        c3 = x
        x = self.stage4(x) # layer_28 - layer_33
        c4 = x
        x = self.stage5(x)
        x = torch.cat([c4, x], dim=1)
        x = self.stage6(x)
        x = torch.cat([c3, x], dim=1)
        x = self.stage7(x)
        x = torch.cat([c2, x], dim=1)
        x = self.stage8(x)
        x = torch.cat([c1, x], dim=1)
        x = self.stage9(x)
        x = self.stage10(x)

        return x
