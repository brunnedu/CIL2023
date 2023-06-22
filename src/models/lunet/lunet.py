import torch
import typing
from torch import nn

from src.models import UNet, UpBlock, Resnet18Backbone


class LUNet(nn.Module):
    """
    LUNet (Label Uncertainty Network) is a probabilistic segmentation model that outputs a gaussian distribution for
    every pixel instead of a single value.

    LUNet consists of 2 modules:
        1. mean_net: UNet-like transform that calculates for every pixel the mean of its gaussian output distribution
        2. var_net: UNet-like transform that calculates for every pixel the variance of its gaussian output distribution

    Both modules should have an output dimension of 1 x H x W, where H and W are the height and width of the input image
    These will be concatenated in the channels dimension resulting in an output of 2 x H x W for the LUNet

    Adapted from `Weakly-Supervised Semantic Segmentation by Learning Label Uncertainty
    <https://arxiv.org/pdf/2110.05926.pdf>`
    """

    def __init__(self, net_cls: typing.Type = None, net_kwargs=None, default_unet_up_mode: str = 'upsample'):
        """
        Parameters
        ----------
        net_cls
            The class of the network to be used for both modules.
        net_kwargs
            The keyword arguments to be passed to the net_cls constructor.
        default_unet_up_mode
            If no net_cls specified, the up mode to be used for the default UNet.
        """
        super().__init__()

        if net_cls is None:
            # Use default UNet with ResNet18 backbone
            self.mean_net = UNet(backbone=Resnet18Backbone(),
                                 up_block_ctor=lambda ci: UpBlock(ci, up_mode=default_unet_up_mode))
            self.var_net = UNet(backbone=Resnet18Backbone(),
                                up_block_ctor=lambda ci: UpBlock(ci, up_mode=default_unet_up_mode))

        else:
            self.mean_net = net_cls(**net_kwargs)
            self.var_net = net_cls(**net_kwargs)

    def forward(self, x):
        # Pass input through both modules
        mean = self.mean_net(x)
        variance = self.var_net(x)

        # Concatenate mean and variance in channels dimension
        return torch.cat([mean, variance], 1)
