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
    """Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.

    Typically, we use to identify tissue in the image and exclude the bright white background.
    If `luminosity_threshold` is None then entire image region is considered as tissue

    Args:
        image: RGB [0, 1]. -> BCHW
        luminosity_threshold: threshold of luminosity in range of [0, 1].  Pixels with intensity within (0, threshold)
            are considered as tissue. If None then all pixels are considered as tissue, effectively bypass this step.
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
    sum_pixel = mask.sum()
    if throw_error and sum_pixel == 0:
        raise TissueMaskException("Empty tissue mask computed")
    if true_when_empty and sum_pixel == 0:
        mask = torch.ones_like(L, dtype=torch.bool)
    return mask


# todo refactor later
def get_tissue_mask_np(I: np.ndarray, luminosity_threshold: float = 0.8,  throw_error: bool = True) -> np.ndarray:
    """Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.

    A numpy version for preprocessing purposes. Note that both Macenko and Vahadane may fail due to mathematical
    instability to process image that is mostly bright background and no tissue at all.

    Typically, we use to identify tissue in the image and exclude the bright white background.

    Args:
        I: numpy image. H x W x C. Input will be automatically converted to uint8 format and range [0, 255]
        luminosity_threshold: threshold of luminosity in range of [0, 1].  Pixels with intensity within (0, threshold)
            are considered as tissue. If None then all pixels are considered as tissue, effectively bypass this step.
        throw_error: Whether to throw an error if no tissue is found.

    Returns:
        binary mask in H x W
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
