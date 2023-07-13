import torch
import torch.nn as nn
import torch.nn.functional as F


class DCMBlock(nn.Module):
    """
    Dilated convolution layer
    """
    def __init__(self, channels):
        super(DCMBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.dil1 = nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1)
        self.dil2 = nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2)
        self.dil3 = nn.Conv2d(channels, channels, kernel_size=3, dilation=3, padding=3)
        self.conv2 = nn.LazyConv2d(channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        dout1 = self.dil1(x)
        dout2 = self.dil2(x)
        dout3 = self.dil3(x)
        out_cat = torch.cat([dout1, dout2, dout3], dim=1)
        return F.relu(self.conv2(out_cat))


class EResPathBlock(nn.Module):
    def __init__(self, channels):
        super(EResPathBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        return out1 + out2


class EResUpBlock(nn.Module):
    def __init__(self, channels):
        super(EResUpBlock, self).__init__()
        self.channels = channels

        # e-res path
        self.block1 = EResPathBlock(channels)
        self.block2 = EResPathBlock(channels)
        self.block3 = EResPathBlock(channels)

        # upconv
        self.up = nn.LazyConvTranspose2d(channels, kernel_size=2, stride=2)

        # e-res layer
        self.conv1x1 = nn.LazyConv2d(channels * 3, kernel_size=1)
        self.conv1 = nn.LazyConv2d(channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, b, s):
        # e-res path
        bout1 = self.block1(*s)
        bout2 = self.block2(bout1)
        bout3 = self.block3(bout2)

        # upconv
        x = torch.cat([self.up(b), bout3], dim=1)

        # e-res layer
        out1x1 = F.relu(self.conv1x1(x))
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        out3 = F.relu(self.conv3(out2))
        out_cat = torch.cat([out1, out2, out3], dim=1)
        return out1x1 + out_cat
