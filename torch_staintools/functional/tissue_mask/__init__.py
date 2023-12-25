import numpy as np
import torch
from torch_staintools.functional.conversion.lab import rgb_to_lab
from torchvision.transforms.functional import convert_image_dtype
from skimage.util import img_as_ubyte
import cv2


class TissueMaskException(Exception):
    ...


def get_tissue_mask(image: torch.Tensor, luminosity_threshold=0.8,
                    throw_error: bool = True,
                    true_when_empty: bool = False) -> torch.Tensor:
    """
    Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.

    Typically, we use to identify tissue in the image and exclude the bright white background.
    If `luminosity_threshold` is None then entire image region is considered as tissue

    Args:
        image: RGB [0, 1]. -> BCHW
        luminosity_threshold: threshold to get the tissue. If None then everywhere is considered as tissue.
        throw_error: whether to throw error
        true_when_empty: if True, then return an all-True mask if no tissue are detected. Effectively bypass the
            tissue detection. Note that in certain cases, both Vahadane and Macenko may either obtain low quality
            results or even crash if the nearly entire input image is background.
    Returns:
        mask (B1HW)
    """
    image = convert_image_dtype(image, torch.float32)

    img_lab: torch.Tensor = rgb_to_lab(image)
    img_lab = torch.atleast_3d(img_lab)
    # enforce the NCHW convention
    if img_lab.ndimension() <= 3:
        img_lab = img_lab.unsqueeze(0)

    L = img_lab[:, 0:1, :, :] / 100

    if luminosity_threshold is None:
        return torch.ones_like(L, dtype=torch.bool)

    mask = (L < luminosity_threshold) & (
                L > 0)  # fix bug in original stain tools code where black background is not ignored.
    # Check it's not empty
    if throw_error and mask.sum() == 0:
        raise TissueMaskException("Empty tissue mask computed")
    return mask


# todo refactor later
def get_tissue_mask_np(I, luminosity_threshold=0.8,  throw_error: bool = True):
    """
    Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
    Typically we use to identify tissue in the image and exclude the bright white background.
    Args:
        I:
        luminosity_threshold:
        throw_error:

    Returns:

    """
    I = img_as_ubyte(I)
    I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
    if luminosity_threshold is None:
        return np.ones_like(L, dtype=bool)
    mask = L < luminosity_threshold

    # Check it's not empty
    if throw_error and mask.sum() == 0:
        raise TissueMaskException("Empty tissue mask computed")
    return mask
