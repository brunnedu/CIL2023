import torch
import torch.nn as nn


class DinkDilateBlock(nn.Module):
    def __init__(self, channel):
        super(DinkDilateBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        )

        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate_out = self.layers(x)
        out = x + dilate_out
        return out


class DinkUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DinkUpBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, b, s):
        return self.layer(b) + s
