import albumentations as A
from torchvision.transforms import Normalize

AUG_TRANSFORM = A.Compose([
    A.Resize(height=224, width=224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
])

# normalization transform using the 2022 dataset means and stds
NORMALIZATION_TRANSFORM = Normalize(mean=(0.5098, 0.5205, 0.5180), std=(0.2315, 0.2096, 0.1998))
