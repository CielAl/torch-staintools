from kornia.color import rgb_to_lab as rgb_to_lab_kornia
from kornia.color import lab_to_rgb as lab_to_rgb_kornia
import torch


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    """A simple wrapper to the Kornia implementation of convert RGB to LAB.

    TODO: add other backends (e.g., cv2?) if necessary.

    Args:
        image: Tensor (BxCxHxW)

    Returns:
        Tensor (BxCxHxW)
    """
    return rgb_to_lab_kornia(image)


def lab_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """A simple wrapper to the Kornia implementation of convert LAB from RGB.

    Args:
        image: BxCxHxW

    Returns:
        torch.Tensor in shape of BxCxHxW
    """
    return lab_to_rgb_kornia(image)


