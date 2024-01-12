from typing import Literal, Callable, Optional, Sequence
import torch
from .base import Augmentor
from ..functional.optimization.dict_learning import METHOD_FACTORIZE
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
              concentration_method: METHOD_FACTORIZE = 'ista',
              rng: Optional[int | torch.Generator] = None,
              target_stain_idx: Optional[Sequence[int]] = (0, 1),
              sigma_alpha: float = 0.2,
              sigma_beta: float = 0.2,
              luminosity_threshold: Optional[float] = 0.8,
              regularizer: float = 0.1,
              use_cache: bool = False,
              cache_size_limit: int = -1,
              device: Optional[torch.device] = None,
              load_path: Optional[str] = None) -> Augmentor:
        """build from specified algorithm name `method` and augment the stain by alpha * concentration + beta

        Warnings:
            concentration_algorithm = 'ls' May fail on GPU for individual large input (e.g., 1000 x 1000),
            regardless of batch size. Therefore, 'ls' is better for multiple small inputs in terms of H and W.

        Args:
            method: Name of stain normalization algorithm. Support `macenko` and `vahadane`
            concentration_method: method to obtain the concentration. Default 'ista' for fast sparse solution on GPU
                only applied for StainSeparation-based approaches (macenko and vahadane).
                support 'ista', 'cd', and 'ls'. 'ls' simply solves the least square problem for factorization of
                min||HExC - OD|| but is faster. 'ista'/cd enforce the sparse penalty but slower.
            rng: random seed for augmentation and any random initialization may incur.
            target_stain_idx: which stain to augment
            sigma_alpha: alpha sampled from (1-sigma_alpha, 1+sigma_alpha)
            sigma_beta: beta sampled from (-sigma_beta, sigma_beta)
            luminosity_threshold: luminosity threshold (smaller than) to find tissue region.
            regularizer: regularization term in ISTA for dictionary learning (e.g., concentration computation)
            use_cache: whether to cache the stain matrix for each input image
            cache_size_limit: limit size of cache (how many matrices to cache). -1 means no limits.
            device: device of the cache
            load_path: whether to load prefetched cache. None means nothing will be loaded.

        Returns:
            corresponding Augmentor object.
        """
        aug_method: Callable
        match method:
            case 'macenko' | 'vahadane':
                return Augmentor.build(method=method, concentration_method=concentration_method,
                                       rng=rng, target_stain_idx=target_stain_idx,
                                       sigma_alpha=sigma_alpha,
                                       sigma_beta=sigma_beta, luminosity_threshold=luminosity_threshold,
                                       use_cache=use_cache,
                                       regularizer=regularizer,
                                       cache_size_limit=cache_size_limit, device=device, load_path=load_path)
            case _:
                raise NotImplementedError(f"{method} not implemented.")
