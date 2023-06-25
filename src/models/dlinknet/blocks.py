import torch.nn as nn


class DLinkDilateBlock(nn.Module):
    def __init__(self, channels):
        super(DLinkDilateBlock, self).__init__()
        self.dil1 = nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1)
        self.dil2 = nn.Conv2d(channels, channels, kernel_size=3, dilation=2, padding=2)
        self.dil3 = nn.Conv2d(channels, channels, kernel_size=3, dilation=4, padding=4)
        self.dil4 = nn.Conv2d(channels, channels, kernel_size=3, dilation=8, padding=8)
        self.dil5 = nn.Conv2d(channels, channels, kernel_size=3, dilation=16, padding=16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out1 = self.dil1(x)
        out2 = self.dil2(out1)
        out3 = self.dil3(out2)
        out4 = self.dil4(out3)
        out5 = self.dil5(out4)

        return x + out1 + out2 + out3 + out4 + out5


class DLinkUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DLinkUpBlock, self).__init__()

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
