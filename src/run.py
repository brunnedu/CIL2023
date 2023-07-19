import os
import typing as t

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import cv2

import pytorch_lightning as pl

from tqdm import tqdm

from src.utils import get_ckpt_path, load_wrapper, prime_factors


def predict(
        image: torch.Tensor,
        original_size: t.Tuple[int],
        pl_wrapper: pl.LightningModule,
        device: str,
        select_channel: t.Optional[int] = None, 
) -> torch.Tensor:
    assert image.shape[0] == 1
    output = pl_wrapper(image.to(device))
    output = F.interpolate(output, original_size, mode='bilinear')

    if select_channel is not None:
        output = output[:,select_channel]

    return output


def predict_patches(
        model: nn.Module,
        image: torch.Tensor,
        patch_size: t.Tuple[int],
        subdivisions: t.Tuple[int],
        device: str = None,
        select_channel: t.Optional[int] = None, 
) -> torch.Tensor:
    """
    Predicts a large image by splitting it into patches, predicting each patch individually and finally averaging them.
    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N, C, H, W = image.shape

    ph, pw = patch_size
    sh, sw = subdivisions

    stride_h = (H - ph) // (sh - 1)
    stride_w = (W - pw) // (sw - 1)

    assert (H - ph) % (sh - 1) == 0 and (W - pw) % (sw - 1) == 0, \
        f"In order to consider all pixels in the patch prediction use one of the following " \
        f"number of subdivisions: {[f + 1 for f in sorted(prime_factors(H - ph))]} "

    # generate patches from image
    patches = image.unfold(2, size=ph, step=stride_h).unfold(3, size=pw, step=stride_w)

    # pass patches through model
    inputs = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ph, pw)
    outputs = model(inputs.to(device))

    if select_channel is not None:
        outputs = outputs[:,select_channel,:,:].unsqueeze(1)

    ch_out = outputs.shape[1]

    # overlay predicted patches and average
    outputs = outputs.reshape(N, sh, sw, ch_out, ph, pw).permute(0, 3, 4, 5, 1, 2).reshape(N, ch_out * ph * pw, -1)
    norm_cnt = torch.ones_like(outputs, dtype=torch.float).to(device)

    outputs = F.fold(outputs, output_size=(H, W), kernel_size=patch_size, stride=(stride_h, stride_w))
    norm_cnt = F.fold(norm_cnt, output_size=(H, W), kernel_size=patch_size, stride=(stride_h, stride_w))

    final_outputs = outputs / norm_cnt

    return final_outputs


def predict_patches_old(
        model: nn.Module,
        image: torch.Tensor,
        patch_size: t.Tuple[int],
        subdivisions: t.Tuple[int],
        device: str,
        select_channel: t.Optional[int] = None, 
) -> torch.Tensor:
    """
    Old version of predict_patches. This version can't handle batched inputs but is able to handle (4, 4) subdivisions.
    """
    N, C, H, W = image.shape

    assert N == 1, "This version of predict_patches can only handle batch size 1."

    py, px = patch_size
    sy, sx = subdivisions

    stride_y = (H - py) / (sy - 1)
    stride_x = (W - px) / (sx - 1)

    positions = [(round(y * stride_y), round(x * stride_x)) for y in range(sy) for x in range(sx)]

    images = [image[0, :, y:y + py, x:x + px] for y, x in positions]
    images = torch.stack(images, dim=0).to(device)
    outputs = model(images)
    
    if select_channel is not None:
        outputs = outputs[:,select_channel,:,:]

    # compute the average over all generated patches
    output_total = torch.zeros((1, 1, H, W)).to(device)
    output_cnt = torch.zeros((1, 1, H, W)).to(device)
    for i, (y, x) in enumerate(positions):
        output_total[0, :, y:y + py, x:x + py] += outputs[i]
        output_cnt[0, :, y:y + py, x:x + py] += 1.0

    output = output_total / output_cnt
    return output


def run_pl_wrapper(
        experiment_id: str,
        dataset: Dataset,
        patches_config: t.Optional[t.Dict],
        out_dir: t.Optional[str] = None,
        use_last_ckpt: bool = False,
        select_channel: t.Optional[int] = None, 
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
    - select_channel: if not none, will return only the i'th channel from the prediction 
        (e.g. if the model predicts multiple masks)
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
    pl_wrapper = load_wrapper(experiment_id, use_last_ckpt, device=device)
    pl_wrapper = pl_wrapper.eval()

    if out_dir is None:
        out_dir = os.path.join('out', experiment_id, 'run_last' if use_last_ckpt else 'run')
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, (names, original_sizes, images) in enumerate(tqdm(dataloader)):
            if patches_config:
                output = predict_patches(pl_wrapper, images, patches_config['size'], patches_config['subdivisions'],
                                         device, select_channel)
            else:
                output = predict(images, original_sizes[0], pl_wrapper, device, select_channel)

            output = output[0].to('cpu').squeeze().numpy() * 255
            cv2.imwrite(os.path.join(out_dir, names[0]), output)
