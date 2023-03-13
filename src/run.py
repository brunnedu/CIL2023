import logging
import os
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision

from src.utils import create_logger, load_model

def run_model(
        experiment_id: str,
        model: nn.Module,
        dataset: Dataset,
        log_frequency: int = 10,
        logger: logging.Logger = None,
        device: t.Optional[str] = None,
) -> float:
    """
    Run the model on every element of a dataset, upscale to the original size and then save the resulting image
    """

    # create logger with file and console stream handlers
    if logger is None:
        logger = create_logger(experiment_id)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # batch_size must be 1!

    # move model to GPU if available
    model = model.to(device)

    # load best model of experiment
    logger.info(f"Loading best model from ./out/{experiment_id}/best_model.pth.tar")
    model = load_model(model, experiment_id, 'best_model.pth.tar')

    model.eval()

    out_dir = f'./out/{experiment_id}/run'
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, (names,original_sizes,inputs) in enumerate(dataloader):
            name = names[0]
            size = original_sizes[0]
            outputs = model(inputs.to(device))
            outputs = F.interpolate(outputs, size, mode='bilinear')
            output = outputs[0].to('cpu')
            output = output * 255
            output = output.type(torch.ByteTensor)

            torchvision.io.write_png(output, os.path.join(out_dir, name))

            # log after every `log_frequency` batches
            if i % log_frequency == 0 or i == len(dataloader) - 1:
                logger.info(f'Run {i}/{len(dataloader)}')
