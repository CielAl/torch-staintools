from typing import Optional, Callable

import torch
from torch_staintools.functional.optimization.dict_learning import dict_learning
from .utils import normalize_matrix_rows
from dataclasses import dataclass

from ..optimization.sparse_util import METHOD_SPARSE, MODE_INIT
from ...constants import PARAM


@dataclass(frozen=False)
class Vcfg:
    """Configuration for Vahadane stain estimation

    Attributes:
    regularizer: lambda term used in dictionary learning if ISTA/FISTA is used.
    lambd_ridge: lambda term used in ridge algorithm in the dictionary step,
        if no positive constraint is enforced.
    steps: number of steps of the dictionary learning iteration.
    algorithm: which algorithm to use, iterative-shrinkage soft thresholding algorithm `ista`, 'fista' or
        coordinate descent `cd`.

    init: init method of the codes `a` in `X = D x a`. Selected from `ridge`, `zero`, `unif` (uniformly random),
        or `transpose`. Details see torch_staintools.functional.optimization.sparse_util.initialize_code
    maxiter: maximum number of iterations in ista loops and cd for code update
    tol: tolerance for convergence of code update.
    lr: step size for ista loops only. not applied to cd.  If None, the 1/Lipschitz will be selected.

"""
    regularizer: float
    algorithm: METHOD_SPARSE
    steps: int
    init: MODE_INIT  # ridge
    maxiter: int
    lr: Optional[float]
    tol: float
    lambd_ridge: float  # 1e-2

DEFAULT_VAHADANE_CONFIG = Vcfg(regularizer=PARAM.OPTIM_DEFAULT_SPARSE_LAMBDA,
                               algorithm="fista", steps=PARAM.DICT_ITER_STEPS,
                               init='transpose', maxiter=PARAM.OPTIM_SPARSE_DEFAULT_MAX_ITER,
                               lr=None, tol=PARAM.OPTIM_DEFAULT_TOL, lambd_ridge=PARAM.INIT_RIDGE_L2)


class VahadaneAlg(Callable):
    cfg: Vcfg

    def __init__(self, cfg: Vcfg):
        self.cfg = cfg


    def __call__(self, od: torch.Tensor,
                       tissue_mask: torch.Tensor,
                       num_stains: int,
                       rng: Optional[torch.Generator]) -> torch.Tensor:
        """
        Stain matrix estimation via method of:
        A. Vahadane et al. 'Structure-Preserving Color Normalization
        and Sparse Stain Separation for Histological Images'

        Args:
            od: optical density image in batch (BxCxHxW)
            tissue_mask: tissue mask so that only pixels in tissue regions will be evaluated
            num_stains: # of stains to separate

        Returns:
            List of stain (e.g., HE) matrices ( B x num_stain x num_channel - H on the 1st row in the 2nd dimension)
        """
        # convert to od and ignore background
        # h*w, c
        assert od.ndimension() == 4, f"{od.shape}"
        assert tissue_mask.ndimension() == 4, f"{tissue_mask.shape}"
        device = od.device
        #  B x (HxWx1)
        tissue_mask = tissue_mask.to(device)
        tissue_mask_flatten = tissue_mask.flatten(start_dim=1, end_dim=-1)
        # B x (H*W) x C
        od_flatten = od.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)

        out_dict_list = list()
        algorithm = self.cfg.algorithm
        regularizer = self.cfg.regularizer
        lambd_ridge = self.cfg.lambd_ridge
        steps = self.cfg.steps
        init = self.cfg.init
        maxiter = self.cfg.maxiter
        lr = self.cfg.lr
        tol = self.cfg.tol

        for od_single, mask_single in zip(od_flatten, tissue_mask_flatten):
            x = od_single[mask_single]
            # todo add num_stains here
            dictionary  = dict_learning(x, n_components=num_stains, algorithm=algorithm,
                                           alpha=regularizer, lambd_ridge=lambd_ridge,
                                           steps=steps,
                                           init=init,
                                           rng=rng, maxiter=maxiter, lr=lr,
                                           tol=tol)
        # H on first row.
            dictionary = dictionary.T
            # todo add num_stains here - sort?
            # if dictionary[0, 0] < dictionary[1, 0]:
            #     dictionary = dictionary[[1, 0], :]
            dictionary, _ = torch.sort(dictionary, dim=0, descending=True)
            out_dict_list.append(normalize_matrix_rows(dictionary))
        # breakpoint()
        return torch.stack(out_dict_list)
