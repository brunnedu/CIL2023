from torchvision.transforms import Compose
from torchvision.transforms import Resize

RESNET_RESIZE = Compose([
    Resize(224),
])