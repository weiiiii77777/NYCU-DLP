import torch
import torch.nn as nn
from torch import autograd

# https://blog.csdn.net/jiangpeng59/article/details/80189889
# Implement your UNet model here

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


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.stage1 = nn.Sequential(
            convBlock(3, 64),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            convBlock(64, 128),
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            convBlock(128, 256),
        )
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            convBlock(256, 512),
        )
        self.stage5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            convBlock(512, 1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        )
        self.stage6 = nn.Sequential(
            convBlock(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        )
        self.stage7 = nn.Sequential(
            convBlock(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )
        self.stage8 = nn.Sequential(
            convBlock(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )
        self.stage9 = nn.Sequential(
            convBlock(128, 64),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        
    def forward(self, x):
        x = self.stage1(x)
        c1 = x
        x = self.stage2(x)
        c2 = x
        x = self.stage3(x)
        c3 = x
        x = self.stage4(x)
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
        return x