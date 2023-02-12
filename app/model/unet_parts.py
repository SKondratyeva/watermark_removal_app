
import torch
import torch.nn as nn

n_channels = 1

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


class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e0 = encoder_block(n_channels, 64)
        self.e1 = encoder_block(64, 128)
        self.e2 = encoder_block(128, 256)
        self.e3 = encoder_block(256, 512)
        self.e4 = encoder_block(512, 512)

        self.conv_block = conv_block(n_channels, 64)
        self.dilation = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 1, 4, dilation=4),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 1, 6, dilation=6),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        self.d1 = decoder_block(1024, 256)
        self.d2 = decoder_block(512, 128)
        self.d3 = decoder_block(256, 64)
        self.d4 = decoder_block(128, 64)

        self.sg = torch.nn.Sigmoid()

        self.m_outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        p1 = self.conv_block(inputs)
        #  x1, p1 = self.e0(inputs)
        x2, p2 = self.e1(p1)
        x3, p3 = self.e2(p2)
        x4, p4 = self.e3(p3)
        x5, p5 = self.e4(p4)

        b = self.dilation(p5)

        x = self.d1(b, p4)
        x = self.d2(x, p3)
        x = self.d3(x, p2)
        x = self.d4(x, p1)

        mask = self.sg(self.m_outputs(x))
        return mask