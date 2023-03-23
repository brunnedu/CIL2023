import os
from typing import Optional

import torch
from torch import nn
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode

from albumentations.core.composition import BaseCompose

from src.transforms import AUG_TRANSFORM, NORMALIZATION_PARAMS_2022, NORMALIZATION_PARAMS_EQUALIZED, RUN_TRANSFORM


class SatelliteDataset(Dataset):
    """
    Dataset for loading satellite images and corresponding roadmap mask labels
    """
    def __init__(
            self,
            data_dir: str = 'data/training',
            add_data_dir: Optional[str] = None,
            hist_equalization: bool = True,
            aug_transform: Optional[BaseCompose] = AUG_TRANSFORM,
    ):
        """
        Parameters
        ----------
        data_dir
            Directory where the original data is located. Has to contain two subdirectories "images" & "groundtruth".
            Both subdirectories should contain files with matching names.
        add_data_dir
            Directory where additional data is located. Has to contain two subdirectories "images" & "groundtruth".
            Both subdirectories should contain files with matching names.
        hist_equalization
            Whether to apply histogram equalization as pre-processing step or not.
        aug_transform
            An albumentation transform that is applied to both the satellite image and the road mask.
        """
        self.data_dir = data_dir
        self.add_data_dir = add_data_dir
        self.img_paths = [os.path.join(self.data_dir, 'images', f) for f in os.listdir(os.path.join(self.data_dir, 'images'))]
        self.mask_paths = [os.path.join(self.data_dir, 'groundtruth', os.path.split(p)[-1]) for p in self.img_paths]

        self.hist_equalization = hist_equalization
        self.aug_transform = aug_transform
        self.post_transform = Normalize(**NORMALIZATION_PARAMS_EQUALIZED) if hist_equalization else Normalize(**NORMALIZATION_PARAMS_2022)

        if self.add_data_dir:
            # add additional data
            add_img_paths = [os.path.join(self.add_data_dir, 'images', f) for f in os.listdir(os.path.join(self.add_data_dir, 'images'))]
            add_mask_paths = [os.path.join(self.add_data_dir, 'groundtruth', os.path.split(p)[-1]) for p in add_img_paths]
            self.img_paths += add_img_paths
            self.mask_paths += add_mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # read img and mask
        img = torchvision.io.read_image(img_path, mode=ImageReadMode.RGB)
        mask = torchvision.io.read_image(mask_path, mode=ImageReadMode.GRAY)/255

        if self.hist_equalization:
            img = F.equalize(img)

        if self.aug_transform:
            # augment img and mask together
            transformed = self.aug_transform(image=img.permute(1, 2, 0).numpy(), mask=mask.permute(1, 2, 0).numpy())
            img = torch.from_numpy(transformed['image']).permute(2, 0, 1)
            mask = torch.from_numpy(transformed['mask']).permute(2, 0, 1)

        # convert from uint8 to float32 and normalize img
        img = self.post_transform(img/255)

        return img, mask


class SatelliteDatasetRun(Dataset):
    """
    Dataset for loading satellite images and their names and original size (but without labels)
    """
    def __init__(
            self,
            data_dir: str = 'data/test',
            hist_equalization: bool = True,
            transform: Optional[nn.Module] = RUN_TRANSFORM,
    ):
        """
        Parameters
        ----------
        data_dir
            Directory where the test data is located. Has to contain a subdirectories named "images".
        hist_equalization
            Whether to apply histogram equalization as pre-processing step or not.
        transform
            A torchvision transform that is applied to the satellite images right after loading them.
        """
        self.data_dir = data_dir
        self.img_paths = [os.path.join(self.data_dir, 'images', f) for f in os.listdir(os.path.join(self.data_dir, 'images'))]

        self.hist_equalization = hist_equalization
        self.transform = transform
        self.post_transform = Normalize(**NORMALIZATION_PARAMS_EQUALIZED) if hist_equalization else Normalize(**NORMALIZATION_PARAMS_2022)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = torchvision.io.read_image(img_path, mode=ImageReadMode.RGB)
        original_size = (img.shape[1], img.shape[2])

        if self.hist_equalization:
            img = F.equalize(img)

        if self.transform:
            img = self.transform(img)

        img = self.post_transform(img/255)

        return img_path, original_size, img