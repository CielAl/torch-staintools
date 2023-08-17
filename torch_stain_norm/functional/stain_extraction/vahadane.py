import torch
from torch_stain_norm.functional.dict_learning.dict_learning import dict_learning
from torch_stain_norm.functional.tissue_mask import get_tissue_mask
from typing import List
from torch_stain_norm.functional.conversion.od import rgb2od


def normalize_matrix_rows(A):
    """
    Normalize the rows of an array.
    :param A: An array.
    :return: Array with rows normalized.
    """
    return A / torch.linalg.norm(A, dim=1)[:, None]


def get_stain_matrix_helper(od: torch.Tensor, tissue_mask: torch.Tensor,
                            regularizer: float = 0.1) -> torch.Tensor:
    """
    Stain matrix estimation via method of:
    A. Vahadane et al. 'Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images'
    Args:
        od: BCHW
        tissue_mask: B1HW
        regularizer:

    Returns:
        List of HE matrix (B*2x3 - H on the 1st row in the 2nd dimension)
    """
    # convert to od and ignore background
    # h*w, c
    assert od.ndimension() == 4, f"{od.shape}"
    assert tissue_mask.ndimension() == 4, f"{tissue_mask.shape}"
    device = od.device
    #  B x (HxWx1)
    tissue_mask_flatten = tissue_mask.flatten(start_dim=1, end_dim=-1)
    # B x (H*W) x C
    od_flatten = od.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)

    out_dict_list = list()
    for od_single, mask_single in zip(od_flatten, tissue_mask_flatten):
        x = od_single[mask_single]

        dictionary, losses = dict_learning(x, n_components=2, alpha=regularizer, lambd=0.01,
                                           algorithm='ista', device=device, steps=30,
                                           constrained=True, progbar=False, persist=True, init='ridge')
    # H on first row.
        dictionary = dictionary.T
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]
        out_dict_list.append(normalize_matrix_rows(dictionary))
    # breakpoint()
    return torch.stack(out_dict_list)


def get_stain_matrix(image: torch.Tensor, luminosity_threshold: float = 0.8,
                     regularizer: float = 0.1) -> torch.Tensor:
    """
    Stain matrix estimation via method of:
    A. Vahadane et al. 'Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images'
    Args:
        image:
        luminosity_threshold:
        regularizer:

    Returns:
        List of HE matrix (2x3 - H on the 1st row)
    """
    # convert to od and ignore background
    # h*w, c
    device = image.device
    # B x 1 x H x W
    tissue_mask = get_tissue_mask(image, luminosity_threshold=luminosity_threshold)  # .reshape((-1,))
    #  B x (HxWx1)

    od = rgb2od(image)
    return get_stain_matrix_helper(od, tissue_mask, regularizer)

