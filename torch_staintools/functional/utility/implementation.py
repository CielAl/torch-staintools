import torch
from typing import Tuple, Optional


def transpose_trailing(mat: torch.Tensor):
    """Helper function to transpose the trailing dimension, since data is batchified.

    Args:
        mat: input tensor ixjxk

    Returns:
        output with flipped dimension from ixjxk --> ixkxj
    """
    assert mat.ndimension() == 3
    return torch.einsum("ijk -> ikj", mat)


def img_from_concentration(concentration: torch.Tensor,
                           stain_matrix: torch.Tensor, img_shape: Tuple[int, ...],
                           out_range: Tuple[float, float] = (0, 1)):
    """reconstruct image from concentration and stain matrix to RGB

    Args:
        concentration: B x (HW) x num_stain
        stain_matrix: B x num_stain x input channel
        img_shape:
        out_range:

    Returns:

    """
    out = torch.exp(-1 * torch.matmul(concentration, stain_matrix))
    out = transpose_trailing(out)
    return out.reshape(img_shape).clamp_(*out_range)


def default_device(device: Optional[torch.device] = None) -> Optional[torch.device]:
    """Default device if device is not given.

    Args:
        device: input device.

    Returns:
        torch.device('cpu') if None, otherwise the input device itself.
    """
    return torch.device('cpu') if device is None else device


def default_rng(rng: Optional[torch.Generator | int], device: Optional[torch.device]) -> Optional[torch.Generator]:
    """Helper function to get the default random number generator (torch.Generator)

    Args:
        rng: Optional. int seed or torch.Generator. If not set (None) then return None.
            Identity mapping if input is already a generator. Create a new generator and specify
            the seed if an int seed is given.
        device: device of the rng

    Returns:
        torch.Generator
    """
    if rng is None:
        return None
    if isinstance(rng, int):
        return torch.Generator(device=device).manual_seed(rng)
    assert isinstance(rng, torch.Generator)
    return rng


def nanstd(data: torch.Tensor, dim: Optional[int | tuple] = None,
           correction: float = 1) -> torch.Tensor:
    """Compute the standard deviation while ignoring NaNs.

    Always keep the dim.

    Args:
        data: Input tensor.
        dim: The dimension or dimensions to reduce. If None (default), reduces all dimensions.
        correction: Difference between the sample size and sample degrees of freedom. Defaults 1 (Bessel's).

    Returns:
        torch.Tensor: Standard deviation with NaNs ignored. If `dim` is provided,
        it reduces along that dimension(s), otherwise reduces all dimensions.
    """

    non_nan_mask = ~torch.isnan(data)
    # find not-nan element
    non_nan_count = non_nan_mask.sum(dim=dim, keepdim=True)
    # compute mean of not-nan elements
    mean = torch.nanmean(data, dim=dim, keepdim=True)

    # \Sigma (x - mean)^2 --> any x that is nan will be filtered by using nansum
    sum_dev2 = ((data - mean) ** 2).nansum(dim=dim,  keepdim=True)
    # sqrt and normalize by corrected degrees of freedom
    return torch.sqrt(sum_dev2 / (non_nan_count - correction))
