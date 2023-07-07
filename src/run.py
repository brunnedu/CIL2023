import os
import typing as t

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

import pytorch_lightning as pl

from tqdm import tqdm

from src.utils import predict_patches, get_ckpt_path, load_wrapper


def predict(
        image: torch.Tensor,
        original_size: t.Tuple[int],
        pl_wrapper: pl.LightningModule,
        device: str
) -> torch.Tensor:
    assert image.shape[0] == 1
    output = pl_wrapper(image.to(device))
    output = F.interpolate(output, original_size, mode='bilinear')
    return output


def run_pl_wrapper(
        experiment_id: str,
        dataset: Dataset,
        patches_config: t.Optional[t.Dict],
        out_dir: t.Optional[str] = None,
        use_last_ckpt: bool = False,
) -> float:
    """
    Run the model on every element of a dataset

    Parameters
    ----------
    - experiment_id: the full name experiment id (determines where to load the model from)
    - dataset: provides the satellite images
    - patches_config: specifies if prediction will be done in patches or all at once
    - out_dir (optional): where should the generated images be stored? 
        if not specified, will create a run folder inside the experiment folder
    - use_last_ckpt: if true, will use the last checkpoint instead of the best one
    """

    # create data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size must be 1!
    tb_logger = pl.loggers.TensorBoardLogger("tb_logs/", name=experiment_id)
    tb_logger.experiment.add_text('experiment_id', experiment_id)

    ckpt_path = get_ckpt_path(experiment_id, use_last_ckpt)
    print(f'Loading {ckpt_path}')

    if patches_config:
        print('Predicting Using Patches')
    else:
        print('Predicting Using Downscaled Image')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    pl_wrapper = load_wrapper(experiment_id, use_last_ckpt)
    pl_wrapper = pl_wrapper.eval()

    if out_dir is None:
        out_dir = os.path.join('out', experiment_id, 'run_last' if use_last_ckpt else 'run')
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, (names, original_sizes, images) in enumerate(tqdm(dataloader)):
            if patches_config:
                output = predict_patches(images, patches_config['size'], patches_config['subdivisions'], pl_wrapper,
                                         device)
            else:
                output = predict(images, original_sizes[0], pl_wrapper, device)

            output = output[0].to('cpu').squeeze().numpy() * 255
            cv2.imwrite(os.path.join(out_dir, names[0]), output)
