from kornia.color import rgb_to_lab as rgb_to_lab_kornia
from kornia.color import lab_to_rgb as lab_to_rgb_kornia
import torch


def rgb_to_lab(image: torch.Tensor):
    """

    Args:
        image: BxCxHxW

    Returns:
        BxCxHxW
    """
    return rgb_to_lab_kornia(image)


def lab_to_rgb(image: torch.Tensor):
    """

    Args:
        image: BxCxHxW

    Returns:
        BxCxHxW
    """
    return lab_to_rgb_kornia(image)


