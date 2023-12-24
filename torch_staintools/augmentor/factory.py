from typing import Literal, Callable
from .base import Augmentor
from functools import partial
AUG_TYPE_VAHADANE = Literal['vahadane']
AUG_TYPE_MACENKO = Literal['macenko']

AUG_TYPE_SUPPORTED = Literal[AUG_TYPE_VAHADANE, AUG_TYPE_MACENKO]


class AugmentorBuilder:
    """Factory Builder for all supported normalizers: reinhard, macenko, and vahadane

    """

    @staticmethod
    def build(method: AUG_TYPE_SUPPORTED, *args, **kwargs) -> Augmentor:
        """build from specified algorithm name `method`.

        Args:
            method: Name of stain normalization algorithm. Support `reinhard`, `macenko`, and `vahadane`
            *args: details see `Augmentor`
            **kwargs: details see `Augmentor`

        Returns:

        """
        aug_method: Callable
        match method:
            case 'macenko' | 'vahadane':
                aug_method = partial(Augmentor.build, method=method)
            case _:
                raise NotImplementedError(f"{method} not implemented.")
        return aug_method(*args, **kwargs)
