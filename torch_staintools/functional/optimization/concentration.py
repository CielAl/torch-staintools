import torch

from torch_staintools.functional.conversion.od import rgb2od
from torch_staintools.functional.optimization.solver import coord_descent, ista, fista
from torch_staintools.functional.optimization.sparse_util import initialize_code, METHOD_FACTORIZE, _batch_supported
from torch_staintools.functional.utility import transpose_trailing


def get_concentrations_single(od_flatten, stain_matrix, regularizer=0.01,
                              method: METHOD_FACTORIZE = 'fista',
                              rng: torch.Generator = None,
                              positive: bool = False,
                              ):
    """Helper function to estimate concentration matrix given an image and stain matrix with shape: 2 x (H*W)

    For solvers without batch support. Inputs are individual data points from a batch

    Args:
        od_flatten: Flattened optical density vectors in shape of (H*W) x C (H and W dimensions flattened).
        stain_matrix: the computed stain matrices in shape of num_stain x input channel
        regularizer: regularization term if ISTA algorithm is used
        method: which method to compute the concentration: coordinate descent ('cd') or iterative-shrinkage soft
            thresholding algorithm ('ista')
        rng: torch.Generator for random initializations
        positive: enforce positive concentration
    Returns:
        computed concentration: num_stains x num_pixel_in_tissue_mask
    """
    z0 = initialize_code(od_flatten, stain_matrix.T, 'zero', rng=rng)
    match method:
        case 'cd':
            return coord_descent(od_flatten, z0, stain_matrix.T, alpha=regularizer, positive_code=positive).T
        case 'ista':
            return ista(od_flatten, z0, stain_matrix.T, alpha=regularizer, positive_code=positive).T
        case 'fista':
            return fista(od_flatten, z0, stain_matrix.T, alpha=regularizer, positive_code=positive).T
        case 'ls':
            return torch.linalg.lstsq(stain_matrix.T, od_flatten.T)[0].T

    raise NotImplementedError(f"{method} is not a valid optimizer")


def get_concentration_one_by_one(od_flatten, stain_matrix, regularizer, algorithm, rng):
    result = list()
    for od_single, stain_mat_single in zip(od_flatten, stain_matrix):
        result.append(get_concentrations_single(od_single, stain_mat_single, regularizer, algorithm, rng=rng))
    # get_concentrations_helper(od_flatten, stain_matrix, regularizer, method)
    return torch.stack(result)


def _ls_batch(od_flatten, stain_matrix):
    """Use least square to solve the factorization for concentration.

    Warnings:
        May fail on GPU for individual large input in cuSolver backend (e.g., 1000 x 1000), regardless of batch size.
        Better for multiple small inputs in terms of H and W.
        Magma backend may work: torch.backends.cuda.preferred_linalg_library('magma')

    Args:
        od_flatten: B * (HW) x num_input_channel
        stain_matrix: B x num_stains x num_input_channel

    Returns:
        concentration B x num_stains x (HW)
    """
    return torch.linalg.lstsq(transpose_trailing(stain_matrix), transpose_trailing(od_flatten))[0]


def get_concentration_batch(od_flatten, stain_matrix, regularizer, algorithm, rng):
    assert algorithm in _batch_supported
    if not _batch_supported[algorithm]:
        return get_concentration_one_by_one(od_flatten, stain_matrix, regularizer, algorithm, rng)
    match algorithm:
        case 'ls':
            return _ls_batch(od_flatten, stain_matrix)
        case _:
            ...

    raise NotImplementedError('Currently only least-square (ls) is implemented as batch concentration solver')


def get_concentrations(image, stain_matrix, regularizer=0.01,
                       algorithm: METHOD_FACTORIZE = 'fista',
                       rng: torch.Generator = None):
    """Estimate concentration matrix given an image and stain matrix.

    Warnings:
        algorithm = 'ls' May fail on GPU for individual large input (e.g., 1000 x 1000), regardless of batch size.
        Better for multiple small inputs in terms of H and W.
    Args:
        image: batched image(s) in shape of BxCxHxW
        stain_matrix: B x num_stain x input channel
        regularizer: regularization term if ISTA algorithm is used
        algorithm: which method to compute the concentration: Solve min||HExC - OD||p
            support 'ista', 'cd', and 'ls'. 'ls' simply solves the least square problem for factorization of
            min||HExC - OD||F (Frobenius norm) but is faster. 'ista'/cd enforce the sparse penalty (L1 norm) but slower.
        rng: torch.Generator for random initializations
    Returns:
        concentration matrix: B x num_stains x num_pixel_in_tissue_mask
    """
    device = image.device
    stain_matrix = stain_matrix.to(device)
    # BCHW
    od = rgb2od(image).to(device)
    # B (H*W) C
    od_flatten = od.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)
    return get_concentration_batch(od_flatten, stain_matrix, regularizer, algorithm, rng)
