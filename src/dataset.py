import os
from typing import Optional

import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode

from albumentations.core.composition import BaseCompose

from src.transforms import AUG_TRANSFORM, NORMALIZATION_PARAMS_2022, NORMALIZATION_PARAMS_EQUALIZED


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
        self.img_paths = sorted([os.path.join(self.data_dir, 'images', f) for f in os.listdir(os.path.join(self.data_dir, 'images'))])
        self.mask_paths = sorted([os.path.join(self.data_dir, 'groundtruth', f) for f in os.listdir(os.path.join(self.data_dir, 'groundtruth'))])

        self.hist_equalization = hist_equalization
        self.aug_transform = aug_transform
        self.post_transform = Normalize(**NORMALIZATION_PARAMS_EQUALIZED) if hist_equalization else Normalize(**NORMALIZATION_PARAMS_2022)

        if self.add_data_dir:
            # add additional data
            self.img_paths += sorted(
                [os.path.join(self.add_data_dir, 'images', f) for f in
                 os.listdir(os.path.join(self.add_data_dir, 'images'))])
            self.mask_paths += sorted(
                [os.path.join(self.add_data_dir, 'groundtruth', f) for f in
                 os.listdir(os.path.join(self.add_data_dir, 'groundtruth'))])

        assert len(self.img_paths) == len(self.mask_paths), "number of satellite images doesn't match number of road masks"

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
