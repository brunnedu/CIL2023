import csv
import logging
import os
import random
import time
from typing import Optional, Tuple, Union, List, Dict

import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim import Optimizer
from torchvision.transforms import Normalize
from matplotlib import pyplot as plt
import cv2


def fix_all_seeds(seed: int) -> None:
    """
    Fix all the different seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_logger(experiment_id: str) -> logging.Logger:
    """
    Set up a logger for the current experiment.
    """
    # set up directory for the current experiment
    experiment_dir = os.path.join("out", experiment_id)
    log_dir = os.path.join(experiment_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # define filename for log file
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_fn = os.path.join(log_dir, f"{time_str}.log")

    # set up logger
    logging.basicConfig(filename=str(log_fn), format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # only add a stream handler if there isn't already one
    if len(logger.handlers) == 1:  # <-- file handler is the existing handler
        console = logging.StreamHandler()
        logger.addHandler(console)

    return logger


def save_plotting_data(experiment_id: str, metric: str, epoch: int, metric_val: float):
    """
    Save metrics after each epoch in a CSV file (to create plots for our report later).
    """
    fn = os.path.join("out", experiment_id, f"{metric}.csv")

    # define header if file does not exist yet
    if not os.path.isfile(fn):
        with open(fn, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", metric])

    # append new data row
    with open(fn, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, metric_val])


def save_checkpoint(
    experiment_id: str,
    next_epoch: int,
    best_acc: float,
    model: nn.Module,
    optimizer: Optimizer,
    filename: str="checkpoint.pth.tar"
    ):
    """
    Save all the necessary data to resume the training at a later point in time.
    More details: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    """
    # checkpoint states
    d = {
        "next_epoch": next_epoch,
        "best_acc": best_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    experiment_dir = os.path.join("out", experiment_id)
    torch.save(d, os.path.join(experiment_dir, filename))


def load_checkpoint(experiment_id: str, model: nn.Module, optimizer: Optimizer) -> Tuple[
    nn.Module, Optimizer, int, float]:
    """
    Load the latest checkpoint and return the updated model and optimizer, the next epoch and best accuracy so far.
    """
    # load checkpoint
    filename = os.path.join("out", experiment_id, "checkpoint.pth.tar")
    checkpoint = torch.load(filename)

    # restore model and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    next_epoch = checkpoint['next_epoch']
    best_acc = checkpoint['best_acc']

    return model, optimizer, next_epoch, best_acc


def save_model(model: nn.Module, experiment_id: str, filename: str):
    """
    Save the current model from the current experiment.
    """
    experiment_dir = os.path.join("out", experiment_id)
    torch.save(model.state_dict(), os.path.join(experiment_dir, filename))

def load_model(model: nn.Module, experiment_id: str, filename : str) -> nn.Module:
    """
    Loads the model from an experiment
    """
    experiment_dir = os.path.join("out", experiment_id)
    file = torch.load(os.path.join(experiment_dir, filename))

    # restore model
    model.load_state_dict(file)

    return model

def display_image(
        img: Union[torch.Tensor, List[torch.Tensor]],
        normalization_params: Dict = None,
        plt_title: str = None,
        nrow: int = 8,
):
    """
    Display a single or multiple torch.Tensor images.
    """
    if not (isinstance(img, torch.Tensor) and len(img.shape) == 3):
        # place images into grid
        image = torchvision.utils.make_grid(img, nrow=nrow)

    if normalization_params is not None:
        # reverse normalization according to normalization dict
        norm_mean, norm_std = np.array(normalization_params['mean']), np.array(normalization_params['std'])
        reverse_normalize = Normalize(mean=-norm_mean / norm_std, std=1 / norm_std)
        img = reverse_normalize(img)

    img_np = img.numpy()
    # shuffle the color channels correctly
    plt.imshow(np.transpose(img_np, (1, 2, 0)))

    # plot
    plt.title(plt_title)
    plt.show()


def display_sample(
        sample: Tuple[torch.Tensor, torch.Tensor],
):
    """
    Display a single dataset sample.
    """
    img, mask = sample

    img = np.transpose(img.numpy(), (1, 2, 0))
    mask = np.transpose(mask.numpy(), (1, 2, 0))

    # overlay image and red mask
    red_mask = np.copy(img)
    red_mask[np.squeeze(mask > 0)] = [1, 0, 0]
    overlay = 0.8 * img + 0.2 * red_mask

    # plot
    plt.imshow(overlay)
    plt.show()
