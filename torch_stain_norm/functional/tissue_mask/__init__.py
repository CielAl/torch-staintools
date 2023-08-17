import torch
from torch_stain_norm.functional.conversion.lab import rgb_to_lab
from torchvision.transforms.functional import convert_image_dtype


class TissueMaskException(Exception):
    ...


def get_tissue_mask(image: torch.Tensor, luminosity_threshold=0.8) -> torch.Tensor:
    """
    Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
    Typically we use to identify tissue in the image and exclude the bright white background.
    Args:
        image: RGB [0, 1]. -> BCHW
        luminosity_threshold:

    Returns:
        mask (B1HW)
    """
    image = convert_image_dtype(image, torch.float32)
    # if use_kornia:
    #     I_LAB = rgb_to_lab(image[None, :, :, :].transpose(1, 3) / 255)
    #     L = (I_LAB[:, 0, :, :] / 100).squeeze()  # Convert to range [0,1].
    # else:
    #     I_LAB = torch.from_numpy(cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_RGB2LAB))
    #     L = (I_LAB[:, :, 0] / 255).squeeze()
    # also check for rgb == 255!
    img_lab: torch.Tensor = rgb_to_lab(image)
    img_lab = torch.atleast_3d(img_lab)
    # enforce the NCHW convention
    if img_lab.ndimension() <= 3:
        img_lab = img_lab.unsqueeze(0)

    L = img_lab[:, 0:1, :, :] / 100
    mask = (L < luminosity_threshold) & (
                L > 0)  # fix bug in original stain tools code where black background is not ignored.
    # Check it's not empty
    if mask.sum() == 0:
        raise TissueMaskException("Empty tissue mask computed")
    return mask
