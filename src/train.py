import os
import typing as t

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl


def train_pl_wrapper(
        experiment_id: str,
        train_dataset: Dataset,
        val_dataset: Dataset,
        pl_wrapper: pl.LightningModule,
        batch_size: int = 64,
        num_workers_dl: int = 4,
        pl_trainer_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        save_checkpoints: bool = True,
        seed: int = 0,
        resume_from_checkpoint: bool = False,
) -> pl.Trainer:
    """
    Train the pytorch lightning wrapper.
    Train loop is completely handled by pytorch lightning.
    """

    # set seed for reproducibility
    pl.seed_everything(seed, workers=True)

    # initialize loggers
    tb_logger = pl.loggers.TensorBoardLogger("tb_logs/", name=experiment_id)
    tb_logger.experiment.add_text('experiment_id', experiment_id)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_dl, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_dl, generator=torch.Generator().manual_seed(seed))

    # instantiate trainer callbacks
    pl_trainer_callbacks = []

    for callback in pl_trainer_kwargs.pop('callbacks', []):
        callback_cls, callback_kwargs = callback
        pl_trainer_callbacks.append(callback_cls(**callback_kwargs))

    if save_checkpoints:
        # add model checkpoint callback
        pl_trainer_callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join('out', experiment_id),
                filename='model-{epoch:02d}-{val_acc:.2f}',
                monitor='val_acc',
                mode='max',
                save_top_k=1,
                save_last=True,
            )
        )

    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=pl_trainer_callbacks,
        enable_checkpointing=save_checkpoints,
        **pl_trainer_kwargs
    )

    trainer.fit(
        pl_wrapper,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=os.path.join('out', experiment_id, 'last.ckpt') if resume_from_checkpoint else None
    )

    return trainer
