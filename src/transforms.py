import albumentations as A
from torchvision.transforms import Compose, Resize, Normalize

AUG_TRANSFORM = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
])

# not used currently, completely defined in config.py
RUN_TRANSFORM = Compose([
    Resize(224),
])

# normalization parameters of the 2022 dataset
NORMALIZATION_PARAMS_2022 = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

# normalization parameters of the 2022 dataset after histogram equalization
NORMALIZATION_PARAMS_EQUALIZED = {
    'mean': [0.4931, 0.4934, 0.4928],
    'std': [0.2903, 0.2905, 0.2906],
}
