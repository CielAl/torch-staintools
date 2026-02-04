from typing import Optional, Callable
import torch
from torch_staintools.functional.optimization.dict_learning import dict_learning
from .utils import validate_shape, post_proc_dict
from dataclasses import dataclass

from ..optimization.sparse_util import METHOD_SPARSE, MODE_INIT
from ...constants import PARAM, CONFIG


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


def stain_mat_loop(cfg: Vcfg, od: torch.Tensor,
                   tissue_mask: torch.Tensor,
                   num_stains: int, rng: Optional[torch.Generator]) -> torch.Tensor:
    algorithm = cfg.algorithm
    regularizer = cfg.regularizer
    lambd_ridge = cfg.lambd_ridge
    steps = cfg.steps
    init = cfg.init
    maxiter = cfg.maxiter
    lr = cfg.lr
    tol = cfg.tol
    out_dict_list = []

    validate_shape(od, tissue_mask)

    # B x (H*W) x C
    od_flatten = od.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)
    tissue_mask_flatten = tissue_mask.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)

    for od_single, mask_single in zip(od_flatten, tissue_mask_flatten):
        # 1 x num_pix x C
        x = od_single[None, ...]
        # 1 x num_pix x 1
        mask_single = mask_single[None, ...]
        # leave masking to dict_learning
        # todo add num_stains here
        dictionary = dict_learning(x,
                                   tissue_mask_flatten=mask_single,
                                   n_components=num_stains, algorithm=algorithm,
                                   alpha=regularizer, lambd_ridge=lambd_ridge,
                                   steps=steps,
                                   init=init,
                                   rng=rng, maxiter=maxiter, lr=lr,
                                   tol=tol)
        sm = post_proc_dict(dictionary)
        out_dict_list.append(sm)
    return torch.cat(out_dict_list, dim=0)

def stain_mat_vectorize(cfg,
                        od: torch.Tensor,
                        tissue_mask: torch.Tensor,
                        num_stains: int,
                        rng: Optional[torch.Generator]) -> torch.Tensor:
    algorithm = cfg.algorithm
    regularizer = cfg.regularizer
    lambd_ridge = cfg.lambd_ridge
    steps = cfg.steps
    init = cfg.init
    maxiter = cfg.maxiter
    lr = cfg.lr
    tol = cfg.tol
    # B x pix x C
    od_flatten = od.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)
    # B x pix x 1
    tissue_mask_flatten = tissue_mask.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)

    dictionary = dict_learning(od_flatten,
                               tissue_mask_flatten=tissue_mask_flatten,
                               n_components=num_stains, algorithm=algorithm,
                               alpha=regularizer, lambd_ridge=lambd_ridge,
                               steps=steps,
                               init=init,
                               rng=rng, maxiter=maxiter, lr=lr,
                               tol=tol)

    return post_proc_dict(dictionary)


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
        if not CONFIG.ENABLE_VECTORIZE:
            return stain_mat_loop(self.cfg, od, tissue_mask, num_stains, rng)
        return stain_mat_vectorize(self.cfg, od, tissue_mask, num_stains, rng)

