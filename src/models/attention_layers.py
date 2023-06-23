import torch
import torch.nn as nn

from .lazy_modules import LazyLayerNorm

class LazyAttentionGate2D(nn.Module):
    """ 
        Implementation as proposed in https://arxiv.org/pdf/1804.03999v3.pdf 
        alpha is using additive attention as in (2) of the paper
    """
    def __init__(self, channels_int: int, batch_norm: bool = False, bias_wx = False):
        super().__init__()

        if batch_norm:
            self.Wg = nn.Sequential(
                nn.LazyConv2d(channels_int, kernel_size=1),
                nn.BatchNorm2d(channels_int)
            )
            self.Wx = nn.Sequential(
                nn.LazyConv2d(channels_int, kernel_size=1, bias=bias_wx),
                nn.BatchNorm2d(channels_int)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(channels_int, 1, kernel_size=1),
                nn.BatchNorm2d(1)
            )
        else: # as in the original paper
            self.Wg = nn.LazyConv2d(channels_int, kernel_size=1)
            self.Wx = nn.LazyConv2d(channels_int, kernel_size=1, bias=bias_wx)
            self.psi = nn.Conv2d(channels_int, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, g, x):
        """ 
        Input features x are scaled with attention coefficients (alpha) computet in AG.
            - g (batch_size * channels_g * Hg * Wg): gating signal (case UNet: collected from coarser scale) 
            - x (batch_size * channels_x * Hx * Wx): input features (case UNet: previous feature-map = upsampled from one level below)
        """
        intermediate = self.relu(self.Wg(g) + self.Wx(x))
        alpha = self.sigmoid(self.psi(intermediate)) 

        return x*alpha
    
class ChannelAttention2D(nn.Module):
    """ Implementation as proposed in https://arxiv.org/pdf/1804.03999v3.pdf """
    def __init__(self) -> None:
        super().__init__()

        self.beta = nn.Parameter(torch.zeros(1)) # beta initialized to zero
        self.softmax = nn.Softmax2d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        flat = torch.flatten(x, start_dim=2) # -> B,C,N
        
        X = torch.bmm(flat, flat.transpose(2,1)) # -> B,C,C
        X = self.softmax(X) # -> B,C,C

        A = torch.bmm(X.transpose(2,1), flat).reshape(B,C,H,W) # -> B,C,H,W

        return self.beta * A + x # -> B,C,H,W
    
class SpatialAttention2D(nn.Module):
    """ Implementation as proposed in https://arxiv.org/pdf/1804.03999v3.pdf """
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.context_model = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Softmax2d()
        )
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            LazyLayerNorm(),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.transform(self.context_model(x) * x)
