from torch import nn
import typing as t
from src.run import predict_patches
from src.utils import init_model


class PatchPredictionModel(nn.Module):
    """
    Wrapper for segmentation models to predict full images using the predict_patches function.
    """

    def __init__(self, base_model_config: t.Dict, patches_config: t.Dict = None):
        super(PatchPredictionModel, self).__init__()
        self.base_model = init_model(base_model_config)
        self.patches_config = {'size': (224, 224), 'subdivisions': (2, 2)} if patches_config is None else patches_config

    def forward(self, x):
        prediction = predict_patches(self.base_model, x, **self.patches_config)
        return prediction
