import torch
from torchvision.transforms.functional import convert_image_dtype
from torch_staintools.functional.compile import lazy_compile


@lazy_compile(dynamic=True)
def _to_od(image: torch.Tensor) -> torch.Tensor:
    eps = 1. / 255.
    image = image.clamp_min(eps)
    return -torch.log(image)


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
    return _to_od(image)


@lazy_compile(dynamic=True)
def _to_rgb(od: torch.Tensor) -> torch.Tensor:
    od = od.clamp_min(0)
    return torch.exp(-od).clamp(0, 1)


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
    return _to_rgb(od)

