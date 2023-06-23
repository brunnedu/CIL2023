import logging
import os
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.utils import save_image

import pytorch_lightning as pl

from src.wrapper import PLWrapper

def run_pl_wrapper(
    experiment_id: str,
    dataset: Dataset,
    pl_wrapper: pl.LightningModule
) -> float:
    """
    Run the model on every element of a dataset, upscale to the original size and then save the resulting image
    """

    # create data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # batch_size must be 1!
    tb_logger = pl.loggers.TensorBoardLogger("tb_logs/", name=experiment_id)
    tb_logger.experiment.add_text('experiment_id', experiment_id)

    experiment_dir = os.path.join('out', experiment_id)
    checkpoint_file = [f for f in os.listdir(experiment_dir) if 'model-epoch=' in f][0]
    print(f'Loading {checkpoint_file}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TODO: make this work with pl load_from_checkpoint (issue: plwrapper has module as hyperparameter)
    pl_wrapper.load_state_dict(torch.load(os.path.join(experiment_dir, checkpoint_file))['state_dict'], device)
    pl_wrapper = pl_wrapper.to(device)
    pl_wrapper = pl_wrapper.eval()

    out_dir = f'./out/{experiment_id}/run'
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, (names,original_sizes,inputs) in enumerate(dataloader):
            name = names[0]
            size = original_sizes[0]
            outputs = pl_wrapper(inputs.to(device))
            outputs = F.interpolate(outputs, size, mode='bilinear')
            output = outputs[0].to('cpu')

            save_image(output, os.path.join(out_dir, name))

            if i % 10 == 0:
                print(i)