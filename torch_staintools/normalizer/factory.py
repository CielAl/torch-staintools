from typing import Literal, Optional
from .base import Normalizer
from .reinhard import ReinhardNormalizer
from .separation import StainSeparation
from ..constants import PARAM, CONFIG
from ..functional.concentration import ConcentrateSolver, ConcentCfg
from ..functional.optimization.sparse_util import METHOD_FACTORIZE, METHOD_SPARSE, MODE_INIT
import torch

from ..functional.stain_extraction.macenko import MacenkoAlg, MckCfg
from ..functional.stain_extraction.vahadane import VahadaneAlg, Vcfg

TYPE_REINHARD = Literal['reinhard']
TYPE_VAHADANE = Literal['vahadane']
TYPE_MACENKO = Literal['macenko']

TYPE_SUPPORTED = Literal[TYPE_REINHARD, TYPE_VAHADANE, TYPE_MACENKO]


class NormalizerBuilder:
    """Factory Builder for all supported normalizers: reinhard, macenko, and vahadane

    """

    @staticmethod
    def build(method: TYPE_SUPPORTED,
              sparse_stain_solver: METHOD_SPARSE = 'fista',
              sparse_dict_steps: int = PARAM.DICT_ITER_STEPS, # 60
              dict_init: MODE_INIT = 'transpose',
              concentration_solver: METHOD_FACTORIZE = 'fista',
              num_stains: int = 2,
              luminosity_threshold: float = 0.8,
              perc: int = 1,
              regularizer: float = PARAM.OPTIM_DEFAULT_SPARSE_LAMBDA,  # 1e-2
              rng: Optional[int | torch.Generator] = None,
              maxiter: int = PARAM.OPTIM_SPARSE_DEFAULT_MAX_ITER,  # 50
              lr: Optional[float] = None,
              tol: float = PARAM.OPTIM_DEFAULT_TOL,  # 1e-5
              use_cache: bool = False,
              cache_size_limit: int = -1,
              device: Optional[torch.device] = None,
              load_path: Optional[str] = None,
              ) -> Normalizer:
        """build from specified algorithm name `method`.


        Warnings:
            concentration_algorithm = 'ls' May fail on GPU for individual large input (e.g., 1000 x 1000),
            regardless of batch size. Therefore, 'ls' is better for multiple small inputs in terms of H and W.

        Args:
            method: Name of stain normalization algorithm. Support `reinhard`, `macenko`, and `vahadane`
            sparse_stain_solver: sparse solver for dictionary learning in Vahadane algorithm.
                Support 'ista', 'fista', and 'cd'.
            sparse_dict_steps: steps of iteration in dictionary learning in Vahadane algorithm.
            dict_init: code initialization method in dictionary learning in Vahadane algorithm.
            concentration_solver: solver to obtain the concentration. Default 'fista' for fast sparse solution on GPU.
                Only applied to StainSeparation-based approaches (macenko and vahadane).
                support 'fista', 'ista', 'cd', and 'ls'. 'ls' solves the least square problem for factorization of
                min||HExC - OD||, faster but no sparsity constraints. 'ista'/cd enforce the sparse penalty but slower.
            num_stains: number of stains to separate. Currently, Macenko only supports 2. Only applies to `macenko` and
                'vahadane' methods.
            luminosity_threshold: luminosity threshold to ignore the background. None means all regions are considered
                as tissue. Scale of luminosity threshold is within [0, 1].  Only applies to `macenko` and
                'vahadane' methods.
            perc: Percentile threshold in Macenko algorithm to find the minimum angular term. min  as 1 percentile
                and max angular as (100 - perc) percentile. Default is 1.
            regularizer: regularizer term in ISTA for stain separation and concentration computation. Only applies
                to `macenko` and 'vahadane' methods if 'ista' is used.
            maxiter: the max iteration in code updating of dictionary learning for stain matrix and
                concentration estimation (e.g., ISTA)
            lr: learning rate for ISTA/FISTA-based optimization in code/concentration updating in ISTA/FISTA.
                If None, the invert of Lipschitz constant of the gradient is used.
            tol: tolerance threshold for early convergence in ISTA/FISTA.
            rng: seed or torch.Generator for any random initialization may incur.
            use_cache: whether to use cache to save the stain matrix of input image to normalize.  Only applies
                to `macenko` and 'vahadane'
            cache_size_limit: size limit of the cache. negative means no limits. Only applies
                to `macenko` and 'vahadane'
            device: what device to hold the cache and the normalizer. If none the device is set to cpu. Only applies
                to `macenko` and 'vahadane'
            load_path: If specified, then stain matrix cache will be loaded from the file path. See the `cache`
                module for more details. Only applies  to `macenko` and 'vahadane'

        Returns:

        """
        c_cfg = ConcentCfg(algorithm=concentration_solver,
                           regularizer=regularizer, rng=rng, maxiter=maxiter,
                           lr=lr,
                           tol=tol,
                           positive=CONFIG.DICT_POSITIVE_CODE)
        csolver = ConcentrateSolver(c_cfg)

        match method:
            case 'reinhard':
                return ReinhardNormalizer.build(luminosity_threshold=luminosity_threshold)
            case 'macenko':
                mck_cfg = MckCfg(perc=int(perc))
                stain_alg = MacenkoAlg(mck_cfg)
                return StainSeparation.build(stain_alg=stain_alg,
                                             concentration_solver=csolver,
                                             num_stains=num_stains,
                                             luminosity_threshold=luminosity_threshold,
                                             use_cache=use_cache,
                                             rng=rng,
                                             cache_size_limit=cache_size_limit,
                                             device=device,
                                             load_path=load_path)
            case 'vahadane':
                vhd_cfg = Vcfg(regularizer=regularizer,
                               algorithm=sparse_stain_solver,
                               steps=sparse_dict_steps,
                               init=dict_init,
                               maxiter=maxiter, lr=lr, tol=tol, lambd_ridge=PARAM.INIT_RIDGE_L2)
                stain_alg = VahadaneAlg(vhd_cfg)
                return StainSeparation.build(stain_alg=stain_alg,
                                             concentration_solver=csolver,
                                             num_stains=num_stains,
                                             luminosity_threshold=luminosity_threshold,
                                             use_cache=use_cache,
                                             rng=rng,
                                             cache_size_limit=cache_size_limit,
                                             device=device,
                                             load_path=load_path)
            case _:
                raise NotImplementedError(f"{method} not implemented.")
