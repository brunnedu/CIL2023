import albumentations as A
from torchvision.transforms import Normalize

AUG_TRANSFORM = A.Compose([
    A.Resize(height=224, width=224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
])

# normalization parameters of the 2022 dataset
NORMALIZATION_PARAMS_2022 = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

NORMALIZATION_TRANSFORM = Normalize(**NORMALIZATION_PARAMS_2022)

