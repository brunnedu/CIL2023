import logging
import time
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, random_split

from src.utils import fix_all_seeds, create_logger, load_checkpoint, save_model, save_checkpoint, save_plotting_data


def train_model(
        experiment_id: str,
        model: nn.Module,
        dataset: Dataset,
        criterion: nn.Module,
        val_frac: float = 0.1,
        optimizer: Optional[Optimizer] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        num_workers: int = 4,
        resume_from_checkpoint: bool = False,
        log_frequency: int = 10,
        fix_seed: bool = True,  # training will not be reproducible if you resume from a checkpoint!
        seed: int = 0,
        logger: logging.Logger = None,
        save_models: bool = True,
        device: Optional[str] = None,
) -> float:
    """
    Training loop.
    """
    if fix_seed:
        fix_all_seeds(seed=seed)

    # create logger with file and console stream handlers
    if logger is None:
        logger = create_logger(experiment_id)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train-val split
    ds_train, ds_val = random_split(dataset, [1 - val_frac, val_frac], generator=torch.Generator().manual_seed(seed))

    # create data loaders
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # use Adam if no optimizer is specified
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    # move model and criterion to GPU if available
    model = model.to(device)
    criterion = criterion.to(device)

    # resume from latest checkpoint if required
    start_epoch = 0
    best_acc = 0.0
    if resume_from_checkpoint:
        logger.info(f"Loading checkpoint from ./out/{experiment_id}/checkpoint.pth.tar")

        if fix_seed:
            logger.info(f"Training will not be reproducible because you resumed from a checkpoint!")

        model, optimizer, start_epoch, best_acc = load_checkpoint(experiment_id, model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        train(experiment_id, model, train_loader, device, criterion, optimizer, epoch, logger, log_frequency)

        # evaluate on validation set
        acc = validate(experiment_id, model, val_loader, device, criterion, epoch, logger, log_frequency)

        # save best model so far
        if acc > best_acc:
            best_acc = acc
            if save_models:
                logger.info(f"Saving best model to ./out/{experiment_id}/best_model.pth.tar")
                save_model(model, experiment_id, "best_model.pth.tar")

        if save_models:
            # update checkpoint
            logger.info(f"Saving checkpoint to ./out/{experiment_id}/checkpoint.pth.tar")
            save_checkpoint(experiment_id, epoch + 1, best_acc, model, optimizer)

    if save_models:
        # save final model
        logger.info(f"Saving final model to ./out/{experiment_id}/final_model.pth.tar")
        save_model(model, experiment_id, "final_model.pth.tar")

    # return best accuracy (for optuna)
    return best_acc


def train(
        experiment_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        device: str,
        criterion: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        logger: logging.Logger,
        log_frequency: int,
) -> None:
    """
    Train the model for one epoch.
    """
    # keep track of batch processing time, data loading time and losses
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to training mode
    model.train()

    for i, (inputs, labels) in enumerate(train_loader):
        curr_time = time.time()
        curr_batch_size = inputs.size(0)

        # TODO: adjust inputs data type according to criterion
        # inputs = inputs.long().to(device)  # cross entropy loss function expects long type

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss and batch processing time
        losses.update(loss.item(), curr_batch_size)
        batch_time.update(time.time() - curr_time)

        # log after every `log_frequency` batches
        if i % log_frequency == 0 or i == len(train_loader) - 1:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, i, len(train_loader) - 1, batch_time=batch_time,
                speed=curr_batch_size / batch_time.val, loss=losses)
            logger.info(msg)

    # save plotting data for later use
    save_plotting_data(experiment_id, "train_loss", epoch, losses.avg)


def validate(
        experiment_id: str,
        model: nn.Module,
        val_loader: DataLoader,
        device: str,
        criterion: nn.Module,
        epoch: int,
        logger: logging.Logger,
        log_frequency: int,
) -> float:
    """
    Validate the model using the validation set and return the accuracy.
    """
    # keep track of batch processing time and losses
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            curr_time = time.time()

            # TODO: adjust inputs data type according to criterion
            # inputs = inputs.long().to(device)  # cross entropy loss function expects long type

            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels)

            # record loss and batch processing time
            losses.update(loss.item(), labels.size(0))
            batch_time.update(time.time() - curr_time)

            # log after every `log_frequency` batches
            if i % log_frequency == 0 or i == len(val_loader) - 1:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    i, len(val_loader) - 1, batch_time=batch_time,
                    loss=losses)
                logger.info(msg)

        # save plotting data for later use
        save_plotting_data(experiment_id, "valid_loss", epoch, losses.avg)

    return losses.avg


class AverageMeter(object):
    """
    Computes and stores the average and current value for some metric.
    Adopted from: https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/core/function.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0