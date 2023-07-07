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
            # for mean use ConvTranspose without BatchNorm or Sigmoid (don't want to limit mean to [0, 1])
            self.mean_net = UNet(
                backbone=Resnet18Backbone(),
                up_block_ctor=lambda ci: UpBlock(ci, up_mode=default_unet_up_mode),
                final=nn.LazyConvTranspose2d(1, kernel_size=2, stride=2)
            )
            # for var use Softplus activation (don't want negative variance)
            # WARNING: Softplus can cause problems on macOS -> set accelerator = 'cpu' in trainer
            self.var_net = UNet(
                backbone=Resnet18Backbone(),
                up_block_ctor=lambda ci: UpBlock(ci, up_mode=default_unet_up_mode),
                final=nn.Sequential(
                    nn.LazyConvTranspose2d(1, kernel_size=2, stride=2),
                    nn.Softplus(),
                    )
            )

        else:
            self.mean_net = net_cls(**net_kwargs)
            self.var_net = net_cls(**net_kwargs)

    def probs(self, x):
        """
        Outputs the mean and variance of the gaussian distribution (probabilistic) for every pixel.
        """

        # Pass input through both modules
        mean = self.mean_net(x)
        variance = self.var_net(x)

        # Return the expected output
        return mean, variance

    def forward(self, x):
        """
        Outputs the expected value of the gaussian distribution for every pixel.
        """

        # Pass input through both modules
        mean, variance = self.probs(x)

        # Compute the expected values over the gaussian distribution when using a sigmoid activation function
        expected_values = (mean / (1 + torch.pi * variance / 8).sqrt()).sigmoid()

        # Return the expected values
        return expected_values
