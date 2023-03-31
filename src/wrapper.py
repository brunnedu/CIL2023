import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn

import typing as t

from .models.unet.backbones import ABackbone
from src.metrics.continuous import FocalLoss, BinaryF1Score


class PLWrapper(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module = FocalLoss(alpha=0.25, gamma=2.0, bce_reduction='none'),
            optimizer_cls: t.Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: t.Dict[str, t.Any] = None,
            lr_scheduler_cls: t.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            lr_scheduler_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
            val_metrics: t.Optional[t.Dict[str, t.Callable]] = None,
    ) -> None:
        """
        PyTorch Lightning wrapper for pytorch models.

        Parameters
        ----------
        model : nn.Module
            The model to wrap.
        loss_fn : nn.Module
            The loss function to use for training.
        optimizer_cls : t.Type[torch.optim.Optimizer]
            The optimizer class to use for training.
        optimizer_kwargs : t.Dict[str, t.Any]
            The optimizer kwargs to use for training.
        lr_scheduler_cls : t.Optional[torch.optim.lr_scheduler._LRScheduler]
            The learning rate scheduler class to use for training.
        lr_scheduler_kwargs : t.Optional[t.Dict[str, t.Any]]
            The learning rate scheduler kwargs to use for training.
        val_metrics : t.Optional[t.Dict[str, t.Callable]]
            The dictionary of validation metrics to use for training.
            Has to contain the key 'acc' with the value being an accuracy function for which larger values are better.
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {'lr': 1e-3, 'weight_decay': 0}
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.loss_fn = loss_fn

        # accuracy metric is necessary as it is needed for early stopping and checkpointing
        if 'acc' not in val_metrics:
            raise ValueError("The val_metrics must contain the key 'acc'."
                             "The provided val_metrics are: " + str(val_metrics))

        self.val_metrics = val_metrics

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        # log additional metrics next to loss
        for name, metric_fn in self.val_metrics.items():
            metric_value = metric_fn(y_hat, y)
            self.log(f'val_{name}', metric_value, on_epoch=True, on_step=False, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)

        if self.lr_scheduler_cls is not None:
            lr_scheduler_kwargs = self.lr_scheduler_kwargs
            # some LR schedulers require a metric to "monitor" which must be set separately
            lr_monitor = lr_scheduler_kwargs.pop("monitor", None)
            lr_scheduler = self.lr_scheduler_cls(optimizer, **lr_scheduler_kwargs)

            return optimizer, {
                "scheduler": lr_scheduler,
                "monitor": lr_monitor if lr_monitor is not None else "val_loss",
            }

        return optimizer


class PLUNet(pl.LightningModule):
    """
        Basic UNet architecture
        Parameters:
            - backbone: the "leftmost" nodes of the UNet, defines the depth and the amount of channels per level
            - up_block_ctor: function that is fed the amount of channels and returns an up node of the UNet
            - final: the final layers of the UNet
    """

    def __init__(
            self,
            backbone: ABackbone,
            up_block_ctor: t.Callable[[int], nn.Module],
            final: t.Optional[nn.Module] = None,
            criterion: nn.Module = FocalLoss(alpha=0.25, gamma=2.0, bce_reduction='none'),
            optimizer_cls: t.Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
            lr_scheduler_cls: t.Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            lr_scheduler_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        ):
        super().__init__()

        self.criterion = criterion
        self.optimizer_cls = optimizer_cls
        self.optimizer_kws = optimizer_kwargs if optimizer_kwargs is not None else {'lr': 1e-3, 'weight_decay': 0}
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        # down nodes / encoder
        self.backbone = backbone

        # layers
        channels = self.backbone.get_channels()
        ups = []
        for ci in channels[:-1]:
            layer = up_block_ctor(ci)
            ups.append(layer)

        self.ups = nn.ModuleList(ups)

        if final:
            self.final = final
        else:
            self.final = nn.Sequential(
                nn.LazyConvTranspose2d(1, kernel_size=2, stride=2),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

    def forward(self, x):
        outs, x = self.backbone(x)

        b = outs[-1]
        for s, up in zip(reversed(outs[:-1]), reversed(self.ups)):
            b = up(b, [s])

        return self.final(b)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(**self.optimizer_kws)

        if self.lr_scheduler_cls is not None:
            lr_scheduler_kwargs = self.lr_scheduler_kwargs
            # some LR schedulers require a metric to "monitor" which must be set separately
            lr_monitor = lr_scheduler_kwargs.pop("monitor", None)
            lr_scheduler = self.lr_scheduler_cls(optimizer, **lr_scheduler_kwargs)

            return optimizer, {
                "scheduler": lr_scheduler,
                "monitor": lr_monitor if lr_monitor is not None else "val_loss",
            }

        return optimizer
