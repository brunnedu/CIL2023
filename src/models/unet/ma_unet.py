import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet
from .backbones import ABackbone
from .blocks import AttentionGateUpBlock
from src.models.attention_layers import ChannelAttention2D, SpatialAttention2D

class MAUNet(nn.Module):
    """ 
        Multiscale Attention UNet architecture based on this paper:
        https://arxiv.org/pdf/2012.10952.pdf

        See the documentation for the AttentionGateUpBlock for further information about the parameters
    """
    def __init__(self, backbone: ABackbone, up_mode: str='upsample', 
                 ag_batch_norm: bool = False, ag_bias_wx: bool = False) -> None:
        super().__init__()

        gated_up_block_ctor = lambda nr_channels: AttentionGateUpBlock(nr_channels=nr_channels, up_mode=up_mode, ag_batch_norm=ag_batch_norm, ag_bias_wx=ag_bias_wx)
        bottom_ctor = lambda nr_channels: nn.Sequential(
            nn.LazyConv2d(nr_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(nr_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(nr_channels, nr_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(nr_channels),
            nn.ReLU(inplace=True)
        )

        # TODO: could make this more channels than just the top level amount, sadly not specified in paper
        attention_channels = backbone.get_channels()[0]
        final = nn.Sequential(
            nn.LazyConv2d(attention_channels, kernel_size=1)
        )
        
        self.channel_attention = ChannelAttention2D()
        self.spatial_attention = SpatialAttention2D(channels=attention_channels) 

        self.unet = UNet(
            backbone=backbone,
            up_block_ctor=gated_up_block_ctor,
            bottom_ctor=bottom_ctor,
            final=final, 
            multiscale_final=True
        )

    def forward(self, x):
        B,C,H,W = x.shape

        x = self.unet(x)
        
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)

        x = torch.mean(torch.cat([ca, sa], dim=1), 1).unsqueeze(1) # sum fusion = sum over all channels
        
        # reduce to input size
        x = F.interpolate(x, (H,W), mode='bilinear').reshape(B, 1, H, W)
        return torch.sigmoid(x)


    