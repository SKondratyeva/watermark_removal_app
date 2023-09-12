from torch.utils.data import DataLoader, Dataset
#from unet_parts import *
#from vgg import Vgg16
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Image_and_Mask(Dataset):
    def __init__(self):
        self.transform_norm = transforms.Compose([transforms.ToTensor()])
        self.images_free_path = r'C:\Users\sofya\uni\diploma\CLWD\CLWD\train\Watermark_free_image'
        self.images_wm_path = r'C:\Users\sofya\uni\diploma\CLWD\CLWD\train\Watermarked_image'
        self.mask_path = r'C:\Users\sofya\uni\diploma\CLWD\CLWD\train\Mask'
        self.alpha_path = r'C:\Users\sofya\uni\diploma\CLWD\CLWD\train\Alpha'
        self.wm_path = r'C:\Users\sofya\uni\diploma\CLWD\CLWD\train\Watermark'
        self.balance_path = r'C:\Users\sofya\uni\diploma\CLWD\CLWD\train\Loss_balance'

        self.images_free = os.listdir(self.images_free_path)
        self.images_wm = os.listdir(self.images_wm_path)
        self.masks = os.listdir(self.mask_path)
        self.alphas = os.listdir(self.alpha_path)
        self.wms = os.listdir(self.wm_path)
        self.balances = os.listdir(self.balance_path)


    def __len__(self):
        # q_10 = int(len(self.images_free)*0.1)
        # return len(self.images_free[:q_10])
        return len(self.images_free)

    def __getitem__(self, i):
        image_free = Image.open(os.path.join(self.images_free_path, self.images_free[i]))
        image_wm = Image.open(os.path.join(self.images_wm_path, self.images_wm[i]))
        mask = Image.open(os.path.join(self.mask_path, self.masks[i])).convert('L')
        alpha = Image.open(os.path.join(self.alpha_path, self.alphas[i]))
        wm = Image.open(os.path.join(self.wm_path, self.wms[i]))

        image_wm = np.array(image_wm)
        image_free = self.transform_norm(np.array(image_free))
        image_wm = self.transform_norm(image_wm)
        mask = self.transform_norm(np.array(mask))
        alpha = self.transform_norm(np.array(alpha)[:, :, 0])
        wm = self.transform_norm(np.array(wm))
        return image_free, image_wm, mask, wm, alpha


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c // 2, in_c // 2, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(in_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class generator(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(generator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dilation=nn.Sequential(
          nn.Conv2d(512,512,3,1,2, dilation=2),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2),
          nn.Conv2d(512,512,3,1,4, dilation=4),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2),
          nn.Conv2d(512,512,3,1,6, dilation=6),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2),
          )
        self.outw = OutConv(64, 3)
        self.outa = OutConv(64, 1)
        self.out_mask = OutConv(64, 1)
        self.sg=nn.Sigmoid()
        self.other=OutConv(64, 64)
        self.post_process_1=nn.Sequential(
          nn.Conv2d(64+6, 64, 3, 1, 1),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(0.2),
          nn.Conv2d(64, 128, 3, 1, 1),
          )
        self.post_process_2=nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 128, 3, 1, 1),
          )
        self.post_process_3=nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 128, 3, 1, 1),
          )
        self.post_process_4=nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 128, 3, 1, 1),
          )
        self.post_process_5=nn.Sequential(
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2),
          nn.Conv2d(128, 3, 3, 1, 1),
          nn.Sigmoid(),
          )
    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dilation(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        w = self.outw(x)
        a = self.outa(x)
        other = self.other(x)
        other = self.sg(other)
        mask = self.out_mask(x)
        mask=self.sg(mask)
        a=self.sg(a)
        w=self.sg(w)
        a=mask*a
        I_watermark=(x0-a*w)/(1.0-a+1e-6)
        I_watermark=torch.clamp(I_watermark,0,1)
        xx1=self.post_process_1(torch.cat([other,I_watermark,x0],1))
        xx2=self.post_process_2(xx1)
        xx3=self.post_process_3(xx1+xx2)
        xx4=self.post_process_4(xx2+xx3)
        I_watermark2=self.post_process_5(xx4+xx3)
        I=I_watermark2*mask+(1.0-mask)*x0
        return I,mask,a,w,I_watermark