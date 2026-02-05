from typing import Optional, Callable

import torch

from torch_staintools.constants import CONFIG, PARAM
from torch_staintools.functional.conversion.od import rgb2od
from torch_staintools.functional.optimization.sparse_solver import coord_descent, ista, fista
from torch_staintools.functional.optimization.sparse_util import initialize_code, METHOD_FACTORIZE, collate_params
from dataclasses import dataclass


# _batch_supported = {
#     get_args(METHOD_ISTA)[0]: True,
#     get_args(METHOD_FISTA)[0]: True,
#     get_args(METHOD_CD)[0]: False,
#     get_args(METHOD_LS)[0]: True,
# }


@dataclass(frozen=False)
class ConcentCfg:
    algorithm: METHOD_FACTORIZE = 'fista'
    regularizer: float = PARAM.OPTIM_DEFAULT_SPARSE_LAMBDA
    rng: Optional[torch.Generator] = None
    maxiter: int = PARAM.OPTIM_SPARSE_DEFAULT_MAX_ITER
    lr: Optional[float] = None
    positive: bool = CONFIG.DICT_POSITIVE_CODE
    # compatibility
    tol: Optional[float] = None  # PARAM.OPTIM_DEFAULT_TOL

DEFAULT_CONC_CFG = ConcentCfg()

def _get_concentrations(od_flatten: torch.Tensor,
                        stain_matrix: torch.Tensor,
                        regularizer: float,
                        algorithm: METHOD_FACTORIZE,
                        lr: Optional[float],
                        maxiter: int,
                        rng: Optional[torch.Generator],
                        positive: bool,
                        ):
    """Helper function to estimate concentration matrix given an image and stain matrix with shape: 2 x (H*W)

    For solvers without batch support. Inputs are individual data points from a batch

    Args:
        od_flatten: Flattened optical density vectors in shape of (H*W) x C (H and W dimensions flattened).
        stain_matrix: the computed stain matrices in shape of num_stain x input channel
        regularizer: regularization term if ISTA algorithm is used
        algorithm: which method to compute the concentration: coordinate descent ('cd') or iterative-shrinkage soft
            thresholding algorithm ('ista')
        lr: learning rate for ISTA and FISTA. No effect for other methods.
            If None, will be computed by 1 / Lipschitz constant.
        maxiter: maximum number of iterations for CD, ISTA, and FISTA
        rng: torch.Generator for random initializations
        positive: enforce positive concentration
    Returns:
        computed concentration: B x num_pixel_in_tissue_mask x num_stain
    """
    z0 = initialize_code(od_flatten, stain_matrix.mT, 'zero', rng=rng)
    lr, regularizer = collate_params(od_flatten, lr, stain_matrix.mT, regularizer)
    match algorithm:
        case 'cd':
            return coord_descent(od_flatten, z0, stain_matrix.mT,
                                 alpha=regularizer, maxiter=maxiter,
                                 positive_code=positive)
        case 'ista':
            return ista(od_flatten, z0, stain_matrix.mT, alpha=regularizer,
                        positive_code=positive, lr=lr, maxiter=maxiter)
        case 'fista':
            return fista(od_flatten, z0, stain_matrix.mT, alpha=regularizer,
                         positive_code=positive, lr=lr, maxiter=maxiter)
        case 'ls':
            return torch.linalg.lstsq(stain_matrix.mT, od_flatten.mT)[0].mT

    raise NotImplementedError(f"{algorithm} is not a valid optimizer")


def get_concentration_one_by_one(od_flatten: torch.Tensor,
                                 stain_matrix: torch.Tensor,
                                 regularizer: float,
                                 algorithm: METHOD_FACTORIZE,
                                 lr: Optional[float],
                                 maxiter: int,
                                 rng: Optional[torch.Generator],
                                 positive: bool,):
    result = list()
    for od_single, stain_mat_single in zip(od_flatten, stain_matrix):
        od_single = od_single[None, ...]
        stain_mat_single = stain_mat_single[None, ...]
        c = _get_concentrations(od_single, stain_mat_single,
                                regularizer, algorithm,
                                lr=lr, maxiter=maxiter,
                                rng=rng, positive=positive)
        result.append(c)
    # get_concentrations_helper(od_flatten, stain_matrix, regularizer, method)
    return torch.cat(result, dim=0)


# def _ls_batch(od_flatten: torch.Tensor, stain_matrix: torch.Tensor):
#     """Use least square to solve the factorization for concentration.
#
#     Warnings:
#         May fail on GPU for individual large input in cuSolver backend (e.g., 1000 x 1000), regardless of batch size.
#         Better for multiple small inputs in terms of H and W.
#         Magma backend may work: torch.backends.cuda.preferred_linalg_library('magma')
#
#     Args:
#         od_flatten: B * (HW) x num_input_channel
#         stain_matrix: B x num_stains x num_input_channel
#
#     Returns:
#         concentration B x num_stains x (HW)
#     """
#     return torch.linalg.lstsq(transpose_trailing(stain_matrix), transpose_trailing(od_flatten))[0]


def get_concentration_batch(od_flatten: torch.Tensor,
                            stain_matrix: torch.Tensor,
                            regularizer: float,
                            algorithm: METHOD_FACTORIZE,
                            lr: Optional[float],
                            maxiter: int,
                            rng: Optional[torch.Generator],
                            positive: bool,):
    # assert algorithm in _batch_supported
    if not CONFIG.ENABLE_VECTORIZE:
        return get_concentration_one_by_one(od_flatten, stain_matrix, regularizer, algorithm,
                                            lr=lr, maxiter=maxiter,
                                            rng=rng, positive=positive)
    return _get_concentrations(od_flatten, stain_matrix, regularizer, algorithm, lr=lr,
                               maxiter=maxiter, rng=rng, positive=positive)


def get_concentrations(image: torch.Tensor,
                       stain_matrix: torch.Tensor,
                       regularizer: float,
                       algorithm: METHOD_FACTORIZE,
                       lr: Optional[float],
                       maxiter: int,
                       rng: Optional[torch.Generator],
                       positive: bool,):
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
        lr: learning rate for FISTA and ISTA. If None then computed from 1 / Lipschitz constant.
        maxiter: max iteration for sparse code optimization.
        rng: torch.Generator for random initializations
        positive: whether to enforce positive concentration.
            Default: True
    Returns:
        concentration matrix: B x num_stains x num_pixel_in_tissue_mask
    """
    device = image.device
    stain_matrix = stain_matrix.to(device)
    # BCHW
    od = rgb2od(image).to(device)
    # B (H*W) C
    od_flatten = od.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)
    return get_concentration_batch(od_flatten, stain_matrix, regularizer, algorithm,
                                   lr=lr, maxiter=maxiter, rng=rng, positive=positive)

class ConcentrationSolver(Callable):
    cfg: ConcentCfg

    def __init__(self, cfg: ConcentCfg):
        self.cfg = cfg

    def __call__(self, image: torch.Tensor, stain_matrix, rng: Optional[torch.Generator]) -> torch.Tensor:
        return get_concentrations(image, stain_matrix,
                                  regularizer=self.cfg.regularizer,
                                  algorithm=self.cfg.algorithm,
                                  lr=self.cfg.lr,
                                  maxiter=self.cfg.maxiter,
                                  rng=rng, positive=CONFIG.DICT_POSITIVE_CODE)
