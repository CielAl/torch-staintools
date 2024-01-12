import torch
from torchvision.transforms.functional import convert_image_dtype
from ..eps import get_eps

_eps_val = torch.finfo(torch.float32).eps


def rgb2od(image: torch.Tensor):
    """Convert RGB to Optical Density space.

    Cedric Walker's adaptation from torchvahadane
    RGB = 255 * exp(-1*OD_RGB) --> od_rgb = -1 * log(RGB / 255)

    Args:
        image: Image RGB. Input scale does not matter.

    Returns:
        Optical density RGB image.
    """
    # to [0, 255]
    image = convert_image_dtype(image, torch.uint8)
    # device = image.device
    # eps = torch.tensor(_eps_val).to(device)
    eps = get_eps(image)
    mask = (image == 0)
    image[mask] = 1
    return torch.maximum(-1 * torch.log(image / 255), eps)


def od2rgb(OD: torch.Tensor):
    """Convert Optical Density to RGB space

    Cedric Walker's adaptation from torchvahadane
    RGB = 255 * exp(-1*OD_RGB)

    Args:
        OD: OD
    Returns:
        RGB.
    """
    assert OD.min() >= 0, "Negative optical density."
    eps = get_eps(OD)
    od_max = torch.maximum(OD, eps)
    return torch.exp(-1 * od_max)
