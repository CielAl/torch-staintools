from typing import Literal, Optional, Sequence
import torch
from .base import Augmentor
from ..constants import PARAM, CONFIG
from ..functional.concentration import ConcentCfg, ConcentrationSolver
from ..functional.optimization.sparse_util import METHOD_FACTORIZE, METHOD_SPARSE, MODE_INIT
from ..functional.stain_extraction.macenko import MckCfg, MacenkoAlg
from ..functional.stain_extraction.vahadane import Vcfg, VahadaneAlg

AUG_TYPE_VAHADANE = Literal['vahadane']
AUG_TYPE_MACENKO = Literal['macenko']

AUG_TYPE_SUPPORTED = Literal[AUG_TYPE_VAHADANE, AUG_TYPE_MACENKO]


class AugmentorBuilder:
    """Factory Builder for all supported normalizers: macenko/vahadane

    For any non-stain separation-based augmentation, the factory builders can be integrated here for a unified
    interface.
    """

    @staticmethod
    def build(method: AUG_TYPE_SUPPORTED,
              sparse_stain_solver: METHOD_SPARSE = 'fista',
              sparse_dict_steps: int = PARAM.DICT_ITER_STEPS,  # 60
              dict_init: MODE_INIT = 'transpose',
              concentration_solver: METHOD_FACTORIZE = 'fista',
              num_stains: int = 2,
              luminosity_threshold: Optional[float] = 0.8,
              perc: int = 1,
              regularizer: float = PARAM.OPTIM_DEFAULT_SPARSE_LAMBDA,  # 1e-2
              maxiter: int = PARAM.OPTIM_SPARSE_DEFAULT_MAX_ITER,  # 50
              lr: Optional[float] = None,
              target_stain_idx: Optional[Sequence[int]] = (0, 1),
              sigma_alpha: float = 0.2,
              sigma_beta: float = 0.2,
              use_cache: bool = False,
              cache_size_limit: int = -1,
              device: Optional[torch.device] = None,
              load_path: Optional[str] = None,
              rng: Optional[int | torch.Generator] = None,) -> Augmentor:
        """build from specified algorithm name `method` and augment the stain by alpha * concentration + beta

        Warnings:
            concentration_algorithm = 'ls' May fail on GPU for individual large input (e.g., 1000 x 1000),
            regardless of batch size. Therefore, 'ls' is better for multiple small inputs in terms of H and W.

        Args:
            method: Name of stain normalization algorithm. Support `macenko` and `vahadane`
            sparse_stain_solver: sparse solver for dictionary learning in Vahadane algorithm.
                Support 'ista', 'fista', and 'cd'.
            sparse_dict_steps: steps of iteration in dictionary learning in Vahadane algorithm.
            dict_init: code initialization method in dictionary learning in Vahadane algorithm.
            concentration_solver: solver to obtain the concentration. Default 'fista' for fast sparse solution on GPU.
                Only applied to StainSeparation-based approaches (macenko and vahadane).
                support 'fista', 'ista', 'cd', and 'ls'. 'ls' solves the least square problem for factorization of
                min||HExC - OD||, faster but no sparsity constraints. 'ista'/cd enforce the sparse penalty but slower.
            num_stains: number of stains to separate. Currently, Macenko only supports 2.
            luminosity_threshold: luminosity threshold (smaller than) to find tissue region.
            perc: Percentile threshold in Macenko algorithm to find the minimum angular term. min  as 1 percentile
                and max angular as (100 - perc) percentile. Default is 1.
            regularizer: regularization term in ISTA for dictionary learning (e.g., concentration computation)
            maxiter: the max iteration in code updating of dictionary learning for stain matrix and
                concentration estimation (e.g., ISTA)
            lr: learning rate for ISTA/FISTA-based optimization in code/concentration updating in ISTA/FISTA.
                If None, the invert of Lipschitz constant of the gradient is used.
            target_stain_idx: which stain to augment
            sigma_alpha: alpha sampled from (1-sigma_alpha, 1+sigma_alpha)
            sigma_beta: beta sampled from (-sigma_beta, sigma_beta)
            use_cache: whether to cache the stain matrix for each input image
            cache_size_limit: limit size of cache (how many matrices to cache). -1 means no limits.
            device: device of the cache
            load_path: whether to load prefetched cache. None means nothing will be loaded.
            rng: random seed for augmentation and any random initialization may incur.

        Returns:
            corresponding Augmentor object.
        """
        c_cfg = ConcentCfg(algorithm=concentration_solver,
                           regularizer=regularizer, rng=rng, maxiter=maxiter,
                           lr=lr,
                           positive=CONFIG.DICT_POSITIVE_CODE)
        csolver = ConcentrationSolver(c_cfg)

        match method:
            case 'macenko':
                mck_cfg = MckCfg(perc=int(perc))
                stain_alg = MacenkoAlg(mck_cfg)
            case 'vahadane':
                vhd_cfg = Vcfg(regularizer=regularizer,
                               algorithm=sparse_stain_solver,
                               steps=sparse_dict_steps,
                               init=dict_init,
                               maxiter=maxiter, lr=lr, lambd_ridge=PARAM.INIT_RIDGE_L2)
                stain_alg = VahadaneAlg(vhd_cfg)
            case _:
                raise NotImplementedError(f"{method} not implemented.")

        return Augmentor.build(stain_alg=stain_alg,
                               concentration_solver=csolver,
                               rng=rng,
                               target_stain_idx=target_stain_idx,
                               sigma_alpha=sigma_alpha,
                               sigma_beta=sigma_beta,
                               num_stains=num_stains,
                               luminosity_threshold=luminosity_threshold,
                               use_cache=use_cache,
                               cache_size_limit=cache_size_limit,
                               device=device, load_path=load_path)