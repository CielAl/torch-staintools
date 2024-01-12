from typing import Literal, Callable, Optional
from .base import Normalizer
from .reinhard import ReinhardNormalizer
from .separation import StainSeparation
from ..functional.optimization.dict_learning import METHOD_FACTORIZE
import torch
TYPE_REINHARD = Literal['reinhard']
TYPE_VAHADANE = Literal['vahadane']
TYPE_MACENKO = Literal['macenko']

TYPE_SUPPORTED = Literal[TYPE_REINHARD, TYPE_VAHADANE, TYPE_MACENKO]


class NormalizerBuilder:
    """Factory Builder for all supported normalizers: reinhard, macenko, and vahadane

    """

    @staticmethod
    def build(method: TYPE_SUPPORTED,
              concentration_method: METHOD_FACTORIZE = 'ista',
              num_stains: int = 2,
              luminosity_threshold: float = 0.8,
              regularizer: float = 0.1,
              rng: Optional[int | torch.Generator] = None,
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
            concentration_method: method to obtain the concentration. Default 'ista' for fast sparse solution on GPU
                only applied for StainSeparation-based approaches (macenko and vahadane).
                support 'ista', 'cd', and 'ls'. 'ls' simply solves the least square problem for factorization of
                min||HExC - OD|| but is faster. 'ista'/cd enforce the sparse penalty but slower.
            num_stains: number of stains to separate. Currently, Macenko only supports 2. Only applies to `macenko` and
                'vahadane' methods.
            luminosity_threshold: luminosity threshold to ignore the background. None means all regions are considered
                as tissue. Scale of luminosity threshold is within [0, 1].  Only applies to `macenko` and
                'vahadane' methods.
            regularizer: regularizer term in ISTA for stain separation and concentration computation. Only applies
                to `macenko` and 'vahadane' methods if 'ista' is used.
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
        norm_method: Callable
        match method:
            case 'reinhard':
                return ReinhardNormalizer.build(luminosity_threshold=luminosity_threshold)
            case 'macenko' | 'vahadane':
                return StainSeparation.build(method=method, concentration_method=concentration_method,
                                             num_stains=num_stains,
                                             luminosity_threshold=luminosity_threshold,
                                             regularizer=regularizer,
                                             use_cache=use_cache,
                                             rng=rng,
                                             cache_size_limit=cache_size_limit,
                                             device=device,
                                             load_path=load_path)
            case _:
                raise NotImplementedError(f"{method} not implemented.")
