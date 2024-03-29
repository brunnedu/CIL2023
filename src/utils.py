import copy
import logging
import os
import time
import warnings
from typing import Tuple, Union, List, Dict

import numpy as np
import optuna
import torch
import torchvision
from pytorch_lightning import Callback, LightningModule, Trainer
from torchvision.io import ImageReadMode
from torchvision.transforms import Normalize
from matplotlib import pyplot as plt

import importlib
from src.wrapper import PLWrapper
import typing as t
from functools import reduce


def get_config(experiment_id: str):
    config_path = os.path.join("out", experiment_id, "config.py")

    if not os.path.isfile(config_path):
        print(f"WARNING: No config at {config_path}")
        return

    config = importlib.import_module('.'.join(["out", experiment_id, "config"]))

    return config


def get_ckpt_path(experiment_id: str, last: bool = False) -> str:
    experiment_dir = os.path.join('out', experiment_id)

    ckpt_name = [f for f in os.listdir(experiment_dir) if f.startswith('last' if last else 'model-epoch=')][0]

    ckpt_path = os.path.join(experiment_dir, ckpt_name)

    return ckpt_path


def init_model(model_config: t.Dict):
    if 'backbone_cls' in model_config and model_config['backbone_cls'] is not None:
        model_config['model_kwargs']['backbone'] = model_config['backbone_cls'](
            **model_config.get('backbone_kwargs', {}))
    model = model_config['model_cls'](**model_config['model_kwargs'])

    return model


def init_wrapper(train_config: t.Dict):
    model = init_model(train_config['model_config'])

    # initialize pytorch lightning wrapper for model
    pl_wrapper = PLWrapper(
        model=model,
        **train_config['pl_wrapper_kwargs'],
    )

    return pl_wrapper


def load_wrapper(experiment_id: str, last: bool = False, ret_cfg: bool = False, device: str = None) -> t.Tuple:
    """
    Load pytorch lightning wrapper from experiment_id

    Parameters
    ----------
    experiment_id : str
        The experiment_id to load the PLWrapper from.
    last : bool
        Whether to load the wrapper from the last epoch instead of the best epoch.
    ret_cfg : bool
        Whether to also return the config.
    device : str
        Which device to load the PLWrapper to.
    """

    # initialize the base wrapper
    config = get_config(experiment_id)
    pl_wrapper = init_wrapper(config.TRAIN_CONFIG)

    # load state dict
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = get_ckpt_path(experiment_id, last)
    pl_wrapper.load_state_dict(
        torch.load(ckpt_path, map_location=device)['state_dict']
    )
    pl_wrapper = pl_wrapper.to(device)

    if ret_cfg:
        return pl_wrapper, config

    return pl_wrapper


def load_model(experiment_id: str, last: bool = False, ret_cfg: bool = False, device: str = None) -> t.Tuple:
    """
    Load pytorch model from experiment_id (without wrapper)

    Parameters
    ----------
    experiment_id : str
        The experiment_id to load the PLWrapper from.
    last : bool
        Whether to load the wrapper from the last epoch instead of the best epoch.
    ret_cfg : bool
        Whether to also return the config.
    device : str
        Which device to load the PLWrapper to.
    """

    res = load_wrapper(experiment_id, last, ret_cfg, device)

    if ret_cfg:
        pl_wrapper, config = res
        return pl_wrapper.model, config

    return res.model


def prime_factors(n):
    """Returns the prime factors of a number"""
    return set(reduce(list.__add__, ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))


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
        img = torchvision.utils.make_grid(img, nrow=nrow)

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


def ensure_dir(dir_path: str):
    """
    Create a directory if it does not exist yet.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_segmentation_masks(dir_path):
    masks = []
    mask_paths = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.png')])
    for mask_path in mask_paths:
        mask = torchvision.io.read_image(mask_path, mode=ImageReadMode.GRAY) / 255.0
        masks.append(mask)
    masks = torch.stack(masks)
    return masks


################
# OPTUNA UTILS #
################


class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


def up_block_ctor_conv(ci: int):
    """
    Workaround because lambda functions aren't serializable with pickle.
    """

    # return UpBlock(ci, up_mode='upconv')


def sample_param_val(trial: optuna.Trial, kwarg: str, val: Tuple[str, float, float]) -> Union[float, int]:
    """
    Sample a parameter value from a given value range.
    """
    # type, lower bound, upper bound
    t, lo, hi = val

    if "float" in t:
        # if t is "float-log" we sample logarithmically
        return trial.suggest_float(kwarg, lo, hi, log=("log" in t))
    elif "int" in t:
        # if t is "int-log" we sample from integer powers of 2
        suggested_int = trial.suggest_int(kwarg, lo, hi, log=False)
        return 2 ** suggested_int if "log" in t else suggested_int
    else:
        raise ValueError(f"The type of parameter to search over should be 'int' or 'float'. Type provided: {t}")


def create_optuna_config(optuna_config: dict, trial: optuna.Trial) -> dict:
    """
    Instantiates every parameter to search over in an optuna configuration dictionary.
    """
    # create a copy of the config
    current_optuna_config = copy.deepcopy(optuna_config)

    def instantiate_dict(d):
        for k, v in d.items():
            if isinstance(v, tuple):
                # sample parameter value
                d[k] = sample_param_val(trial, k, v)
            elif isinstance(v, dict):
                # recurse into dictionary
                instantiate_dict(v)

    instantiate_dict(current_optuna_config)

    return current_optuna_config
