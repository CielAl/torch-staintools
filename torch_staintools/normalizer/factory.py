from typing import Literal, Callable
from .base import Normalizer
from .reinhard import ReinhardNormalizer
from .separation import StainSeperation
from functools import partial
TYPE_REINHARD = Literal['reinhard']
TYPE_VAHADANE = Literal['vahadane']
TYPE_MACENKO = Literal['macenko']

TYPE_SUPPORTED = Literal[TYPE_REINHARD, TYPE_VAHADANE, TYPE_MACENKO]


class NormalizerBuilder:
    """Factory Builder for all supported normalizers: reinhard, macenko, and vahadane

    """

    @staticmethod
    def build(method: TYPE_SUPPORTED, *args, **kwargs) -> Normalizer:
        """build from specified algorithm name `method`.

        Args:
            method: Name of stain normalization algorithm. Support `reinhard`, `macenko`, and `vahadane`
            *args:
            **kwargs:

        Returns:

        """
        norm_method: Callable
        match method:
            case 'reinhard':
                norm_method = ReinhardNormalizer.build
            case 'macenko' | 'vahadane':
                norm_method = partial(StainSeperation.build, method=method)
            case _:
                raise NotImplementedError(f"{method} not implemented.")
        return norm_method(*args, **kwargs)
