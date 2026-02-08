import torch
from torchvision.transforms.functional import convert_image_dtype


_eps_val = torch.finfo(torch.float32).eps


def rgb2od(image: torch.Tensor):
    """Convert RGB to Optical Density space.

    Cedric Walker's adaptation from torchvahadane
    RGB = 255 * exp(-1*OD_RGB) --> od_rgb = -1 * log(RGB / 255)

    Args:
        image: Image RGB. Either float image in [0, 1] or ubyte image in [0, 255]

    Returns:
        Optical density RGB image.
    """
    # to [0, 255]
    image = convert_image_dtype(image, torch.float32)
    # device = image.device
    # eps = torch.tensor(_eps_val).to(device)
    eps = 1. / 255.
    image = image.clamp_min(eps)
    return -torch.log(image)


def od2rgb(od: torch.Tensor):
    """Convert Optical Density to RGB space

    Cedric Walker's adaptation from torchvahadane
    RGB = 255 * exp(-1*OD_RGB)

    Args:
        od: OD
    Returns:
        RGB.
    """
    assert od.min() >= 0, "Negative optical density."
    # eps = get_eps(OD)
    # od_max = torch.maximum(OD, eps)
    # eps = torch.finfo(torch.float32).eps
    # ignore negative OD
    od = od.clamp_min(0)
    return torch.exp(-od).clamp(0, 1)
