import torch
import torch.nn as nn

# https://zhuanlan.zhihu.com/p/353235794
# Define the ResNet model architecture

class BTNK1(nn.Module):
    def __init__(self, input_size, output_size, s):
        super(BTNK1, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_size, int(output_size/4), kernel_size=1, stride=s),
            nn.BatchNorm2d(int(output_size/4)),
            nn.ReLU(),
            nn.Conv2d(int(output_size/4), int(output_size/4), kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(int(size/4), int(size/4), kernel_size=3, stride=1, padding=1),
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


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(
            BTNK1(64, 256, 1),
            BTNK2(256),
            BTNK2(256),
        )
        self.stage2 = nn.Sequential(
            BTNK1(256, 512, 2),
            BTNK2(512),
            BTNK2(512),
            BTNK2(512),
        )
        self.stage3 = nn.Sequential(
            BTNK1(512, 1024, 2),
            BTNK2(1024),
            BTNK2(1024),
            BTNK2(1024),
            BTNK2(1024),
            BTNK2(1024),
        )
        self.stage4 = nn.Sequential(
            BTNK1(1024, 2048, 2),
            BTNK2(2048),
            BTNK2(2048),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048 * 2 * 2, 100),
        )

    def forward(self, x):
        x = self.stage0(x) # layer_1
        x = self.stage1(x) # layer_2 - layer_10
        x = self.stage2(x) # layer_11 - layer_22
        x = self.stage3(x) # layer_23 - layer_40
        x = self.stage4(x) # layer_41 - layer_49
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x) # layer_50
        return x