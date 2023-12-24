import torch
from typing import Tuple


def transpose_trailing(mat: torch.Tensor):
    """Helper function to transpose the trailing dimension, since data is batchified.

    Args:
        mat: input tensor ixjxk

    Returns:
        output with flipped dimension from ixjxk --> ixkxj
    """
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
