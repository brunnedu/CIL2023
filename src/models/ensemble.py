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
            experiment_ids: t.List[str],
            freeze_submodels: t.Optional[bool] = True,
            final_layer: t.Optional[nn.Module] = None,
            last: bool = False,
    ):
        """
        Parameters
        ----------
        experiment_ids
            List of experiment ids of the submodels to be loaded. All models should have the same resolution.
        freeze_submodels
            If True, the weights of the submodels are frozen (not updated anymore).
        final_layer
            The final layer which the channel-stacked outputs of all submodels are passed through. If None, the mean of
            all submodel outputs is taken.
        last
            If True, load the last checkpoint instead of the best one.
        """
        super(SegmentationEnsemble, self).__init__()

        self.experiment_ids = experiment_ids
        self.freeze_submodels = freeze_submodels
        self.final_layer = final_layer

        # load pretrained models
        models = []
        configs = []
        for experiment_id in experiment_ids:
            model, config = load_model(experiment_id, last=last, ret_cfg=True)
            models.append(model)
            configs.append(config)

        # check model resolution and prediction mode
        if not all([cfg.MODEL_RES == configs[0].MODEL_RES for cfg in configs]):
            print("WARNING: Not all models in the ensemble use the same resolution.")
            for ei, cfg in zip(experiment_ids, configs):
                print(f"{ei}:\t{cfg.MODEL_RES}")

        if not all([cfg.PREDICT_USING_PATCHES == configs[0].PREDICT_USING_PATCHES for cfg in configs]):
            print("WARNING: Not all models in the ensemble use the same prediction mode (predict_using_patches).")
            for ei, cfg in zip(experiment_ids, configs):
                print(f"{ei}:\t{cfg.PREDICT_USING_PATCHES}")

        # un-/freeze weights as specified
        for model in models:
            model.requires_grad_(not freeze_submodels)

        self.submodels = nn.ModuleList(models)

    def forward(self, x):
        outputs = []

        # forward pass through each submodel
        for submodel in self.submodels:
            outputs.append(submodel(x))

        # stack outputs of all submodels in channel dimension
        output = torch.cat(outputs, dim=1)  # shape: (N, len(self.submodels), H, W)

        if self.final_layer is not None:
            output = self.final_layer(output)
        else:
            output = torch.mean(output, dim=1, keepdim=True)

        return output
