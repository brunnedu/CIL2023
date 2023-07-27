import torch
import torch.nn as nn

import typing as t
from src.utils import load_model


class SegmentationEnsemble(nn.Module):
    """
    Ensemble of segmentation models.
    """

    def __init__(
            self,
            final_layer: t.Optional[nn.Module] = None,
    ):
        """
        Parameters
        ----------
        final_layer
            The final layer which the channel-stacked outputs of all submodels are passed through. If None, the mean of
            all submodel outputs is taken.
        """
        super(SegmentationEnsemble, self).__init__()

        self.final_layer = final_layer

    def forward(self, x):
        if self.final_layer is not None:
            output = self.final_layer(x)
        else:
            output = torch.mean(x, dim=1, keepdim=True)

        return output
