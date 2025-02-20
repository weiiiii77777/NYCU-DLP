import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,args):
        super(Generator, self).__init__()
        self.ngf = args.ngf # ngf: 在G中onvolution時的kernel數量，相當於feature厚度: 300
        self.nc = args.nc # nc: 將condition經過線性轉換後從n_classes維變成nc維: 100
        self.nz = args.latent_dim # nz: latent(noise)的維度: 100        
        self.n_classes = args.n_classes # 24

        # condition embedding: 將condition從n_classes(24)維線性轉換成nc(100)維
        self.label_emb = nn.Sequential(
            nn.Linear(self.n_classes, self.nc),
            nn.LeakyReLU(0.2, True)
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.nz + self.nc, self.ngf * 8, 4, 1, 0, bias=False), 
            # (input feature大小 - 1) * stride - 2 * padding + kernel_ size
            # input channel: 100 + 100 = 200, output channel: 300 * 8 = 2400, spatial: (64 - 1) * 1 - 2 * 0 + 4 = 67
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False), 
            # 2400, 1200, (67 - 1) * 2 - 2 * 1 + 4 = 134
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            # 1200, 600, (134 - 1) * 2 - 2 * 1 + 4 = 268
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            # 600, 300, (268 - 1) * 2 - 2 * 1 + 4 = 536
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            # 300, rgb: 3, (536 - 1) * 2 - 2 * 1 + 4 = 1072
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels).view(-1, self.nc, 1, 1) # batch size, 100, 1, 1
        gen_input = torch.cat((label_emb, noise), 1) # batch size, 100 + 100(nz), 1, 1
        out = self.main(gen_input)
        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.ndf = args.ndf # 64
        self.nc = args.nc # 100
        self.n_classes = args.n_classes # 24
        self.img_size = args.img_size # 64
        self.label_emb = nn.Sequential(
            nn.Linear(self.n_classes, self.img_size * self.img_size),
            nn.LeakyReLU(0.2, True)
        )

        self.main = nn.Sequential(
            
            nn.Conv2d(3 + 1, self.ndf, 4, 2, 1, bias=False), 
            # [(input + 2 * pad) - filter + 1] / stride
            # 3, 64, [(64 + 2 * 1) - 3 + 1] / 2 = 32
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5, inplace=False),
            
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            # 64, 128, [(32 + 2 * 0) - 3 + 1] / 1 = 30
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5, inplace=False),
            
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            # 128, 256, [(30 + 2 * 1) - 3 + 1] / 2 = 15
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5, inplace=False),
            
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            # 256, 512, [(15 + 2 * 0) - 3 + 1] / 1 = 13
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5, inplace=False),
           
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()

        )
        
        # discriminator fc: real or fake
        self.fc_dis = nn.Sequential(
            nn.Linear(5*5*2048, 1),
            nn.Sigmoid()
        )
        # aux-classifier fc: class
        self.fc_aux = nn.Sequential(
            nn.Linear(5*5*2048, self.n_classes),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_emb = self.label_emb(labels).view(-1, 1, self.img_size,  self.img_size) 
        dis_input = torch.cat((img, label_emb), 1) # batch size, 100 + 100(nz), 1, 1
        out = self.main(dis_input)

        return out.view(-1, 1).squeeze(1)