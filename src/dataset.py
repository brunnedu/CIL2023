import os
from typing import Optional

import torch
from torch import nn
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
import numpy as np

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
            hist_equalization: bool = False,
            aug_transform: Optional[BaseCompose] = AUG_TRANSFORM,
            include_low_quality_mask: bool = False,
            include_fid: bool = False,
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
        include_low_quality_mask
            If specified, will load and stack low quality masks from /lowqualitymask together with the satellite image
        include_fid
            If in addition to the mask, flow, intersection and deadend masks should be loaded
        """
        self.data_dir = data_dir
        self.add_data_dir = add_data_dir
        self.include_low_quality_mask = include_low_quality_mask
        self.include_fid = include_fid
        self.img_paths = [os.path.join(self.data_dir, 'images', f) for f in os.listdir(os.path.join(self.data_dir, 'images')) if f.endswith('.png')]
        if self.include_fid:
            self.mask_paths = [os.path.join(self.data_dir, 'transformed/mask_flow_intersection_deadend', os.path.split(p)[-1].split('.')[0] + '.npy') for p in self.img_paths]
        else:
            self.mask_paths = [os.path.join(self.data_dir, 'groundtruth', os.path.split(p)[-1]) for p in self.img_paths]
        self.low_quality_mask_paths = [os.path.join(self.data_dir, 'lowqualitymask', os.path.split(p)[-1]) for p in self.img_paths]

        self.hist_equalization = hist_equalization
        self.aug_transform = aug_transform
        self.post_transform = Normalize(**NORMALIZATION_PARAMS_EQUALIZED) if hist_equalization else Normalize(**NORMALIZATION_PARAMS_2022)

        if self.add_data_dir:
            # add additional data
            add_img_paths = [os.path.join(self.add_data_dir, 'images', f) for f in os.listdir(os.path.join(self.add_data_dir, 'images')) if f.endswith('.png')]
            if self.include_fid:
                add_mask_paths = [os.path.join(self.add_data_dir, 'transformed/mask_flow_intersection_deadend', os.path.split(p)[-1].split('.')[0] + '.npy') for p in add_img_paths]
            else:
                add_mask_paths = [os.path.join(self.add_data_dir, 'groundtruth', os.path.split(p)[-1]) for p in add_img_paths]
            add_low_quality_mask_paths = [os.path.join(self.add_data_dir, 'lowqualitymask', os.path.split(p)[-1]) for p in add_img_paths]
                
            self.img_paths += add_img_paths
            self.mask_paths += add_mask_paths
            self.low_quality_mask_paths += add_low_quality_mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.include_low_quality_mask:
            return self.get_with_low_quality_mask(idx)
        
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # read img and mask
        img = torchvision.io.read_image(img_path, mode=ImageReadMode.RGB)

        if self.include_fid:
            mask = torch.Tensor(np.load(mask_path))
        else:
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
    
    def get_with_low_quality_mask(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        low_quality_mask_path = self.low_quality_mask_paths[idx]

        # read img and mask
        img = torchvision.io.read_image(img_path, mode=ImageReadMode.RGB)
        mask = torchvision.io.read_image(mask_path, mode=ImageReadMode.GRAY)/255
        low_quality_mask = torchvision.io.read_image(low_quality_mask_path, mode=ImageReadMode.GRAY)/255

        if self.hist_equalization:
            img = F.equalize(img)

        if self.aug_transform:
            # augment img and mask together
            transformed = self.aug_transform(image=img.permute(1, 2, 0).numpy(), masks=[mask.permute(1, 2, 0).numpy(), low_quality_mask.permute(1, 2, 0).numpy()])
            img = torch.from_numpy(transformed['image']).permute(2, 0, 1)
            mask = torch.from_numpy(transformed['masks'][0]).permute(2, 0, 1)
            low_quality_mask = torch.from_numpy(transformed['masks'][1]).permute(2, 0, 1)

        # convert from uint8 to float32 and normalize img
        img = self.post_transform(img/255)

        return torch.cat([img, low_quality_mask], dim=0), mask


class SatelliteDatasetRun(Dataset):
    """
    Dataset for loading satellite images and their names and original size (but without labels)
    """
    def __init__(
            self,
            data_dir: str = 'data/test',
            hist_equalization: bool = False,
            transform: Optional[nn.Module] = RUN_TRANSFORM,
            include_low_quality_mask: bool = False
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
        include_low_quality_mask
            If specified, will load and stack low quality masks from /lowqualitymask together with the satellite image
        """
        self.data_dir = data_dir
        self.include_low_quality_mask = include_low_quality_mask
        self.img_dir = os.path.join(self.data_dir, 'images')
        self.img_paths = list([p for p in os.listdir(self.img_dir) if p.endswith('.png')])
        self.low_quality_mask_paths = [os.path.join(self.data_dir, 'lowqualitymask', os.path.split(p)[-1]) for p in self.img_paths]

        self.hist_equalization = hist_equalization
        self.transform = transform
        self.post_transform = Normalize(**NORMALIZATION_PARAMS_EQUALIZED) if hist_equalization else Normalize(**NORMALIZATION_PARAMS_2022)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.include_low_quality_mask:
            return self.get_with_low_quality_mask(idx)
        
        img_name = self.img_paths[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = torchvision.io.read_image(img_path, mode=ImageReadMode.RGB)
        original_size = (img.shape[1], img.shape[2])

        if self.hist_equalization:
            img = F.equalize(img)

        if self.transform:
            img = self.transform(img)

        img = self.post_transform(img/255)

        return img_name, original_size, img
    
    def get_with_low_quality_mask(self, idx):
        img_name = self.img_paths[idx]
        img_path = os.path.join(self.img_dir, img_name)
        low_quality_mask_path = self.low_quality_mask_paths[idx]

        img = torchvision.io.read_image(img_path, mode=ImageReadMode.RGB)
        low_quality_mask = torchvision.io.read_image(low_quality_mask_path, mode=ImageReadMode.GRAY)/255
        original_size = (img.shape[1], img.shape[2])

        if self.hist_equalization:
            img = F.equalize(img)

        if self.transform:
            transformed = self.transform(image=img.permute(1, 2, 0).numpy(), masks=[low_quality_mask.permute(1, 2, 0).numpy()])
            img = torch.from_numpy(transformed['image']).permute(2, 0, 1)
            low_quality_mask = torch.from_numpy(transformed['masks'][0]).permute(2, 0, 1)

        img = self.post_transform(img/255)

        return img_name, original_size, torch.cat([img, low_quality_mask], dim=0)