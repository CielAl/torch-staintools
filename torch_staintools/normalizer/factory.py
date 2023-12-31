from typing import Literal, Callable, Optional
from .base import Normalizer
from .reinhard import ReinhardNormalizer
from .separation import StainSeparation
from functools import partial
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
              reconst_method: str = 'ista',
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

        Args:
            method: Name of stain normalization algorithm. Support `reinhard`, `macenko`, and `vahadane`
            reconst_method: method to obtain the concentration. default ista for computational efficiency on GPU.
                only applied for StainSeparation-based approaches (macenko and vahadane)
            num_stains: number of stains to separate. Currently, Macenko only supports 2. Only applies to `macenko` and
                'vahadane' methods.
            luminosity_threshold: luminosity threshold to ignore the background. None means all regions are considered
                as tissue. Scale of luminiosty threshold is within [0, 1].  Only applies to `macenko` and
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
               return ReinhardNormalizer.build()
            case 'macenko' | 'vahadane':
                return StainSeparation.build(method=method, reconst_method=reconst_method,
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
