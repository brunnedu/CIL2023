import typing as t

import torch
import torch.nn as nn

class MfidFinal(nn.Module):
    """ 
        Output layer for the mask, flow, intersection, deadeand network 

        Parameters
        ----------
        - channels: how many channels should be used to combine the different masks
    """
    def __init__(self, channels: int = 8):
        super().__init__()

        self.unet_final = nn.Sequential(
            nn.LazyConvTranspose2d(4, kernel_size=2, stride=2),
            nn.BatchNorm2d(4),
            nn.Sigmoid()
        )

        self.prep_mask = nn.Sequential(
            nn.LazyConv2d(channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU()
        )
        
        self.prep_flow = nn.Sequential(
            nn.LazyConv2d(channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU()
        )

        self.prep_intersection = nn.Sequential(
            nn.LazyConv2d(channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU()
        )

        self.prep_deadend = nn.Sequential(
            nn.LazyConv2d(channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU()
        )

        self.combine = nn.Sequential(
            nn.Conv2d(4*channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channels, 1, kernel_size=2, stride=2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ux = self.unet_final(x)

        ux_mask, ux_flow, ux_intersection, ux_deadend = self.extract_mfid(ux)
        x_mask, x_flow, x_intersection, x_deadend = self.extract_mfid(x)
        
        y_mask = self.prep_mask(x_mask)
        y_flow = self.prep_flow(x_flow)
        y_intersection = self.prep_intersection(x_intersection)
        y_deadend = self.prep_deadend(x_deadend)

        y = torch.cat([y_mask, y_flow, y_intersection, y_deadend], dim=1)

        final_mask = self.combine(y)
        ux[:,0,:,:] = ((ux_mask + final_mask) * 0.5).squeeze(1)

        return torch.cat([final_mask, ux_flow, ux_intersection, ux_deadend], dim=1)

    def extract_mfid(self, x):
        return x[:,0,:,:].unsqueeze(1), x[:,1,:,:].unsqueeze(1), x[:,2,:,:].unsqueeze(1), x[:,3,:,:].unsqueeze(1)


