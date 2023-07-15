import torch
import torch.nn as nn
import torch.nn.functional as F

class VotenetFinal(nn.Module):
    """ 
        Final layer that outputs a mean and std mask of numerical values
    """

    def __init__(self) -> None:
        super().__init__()

        self.final = nn.Sequential(
            nn.LazyConv2d(4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2),
        )

    def forward(self, x):
        mean, std = torch.chunk(self.final(x), 2, dim=1)

        # ensure positivity & avoid numerical instability
        std = F.relu(std).clamp(0.1, 500.0)

        return torch.cat([mean, std], dim=1)