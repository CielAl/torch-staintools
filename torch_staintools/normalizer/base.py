from typing import TypedDict, Union, Optional
from abc import abstractmethod
import torch
import numpy as np
from PIL import Image
from ..base_module.base import CachedRNGModule
from ..cache.tensor_cache import TensorCache


TYPE_IMAGE = Union[np.ndarray, torch.Tensor, Image.Image]


class DataInput(TypedDict):
    """For future compatibility - e.g., moving average of stain matrix from same wsi which needs uri to identify.

    """
    # todo - for other information passed in the pipeline
    img: TYPE_IMAGE
    uri: str


class Normalizer(CachedRNGModule):
    """Generic normalizer interface with fit/transform, and the forward call that will at least call transform.

    Note that the inputs are always supposed to be pytorch tensors in BCHW convention.

    """

    @abstractmethod
    def fit(self, *args, **kwargs):
        ...

    @abstractmethod
    def transform(self, x, *args, **kwarags):
        ...

    @abstractmethod
    def forward(self, x: Union[DataInput, torch.Tensor], *args, **kwargs):
        ...

    def __init__(self, cache: Optional[TensorCache], device: Optional[torch.device],
                 rng: Optional[int | torch.Generator]):
        super().__init__(cache, device, rng)

    @classmethod
    def build(cls, *args, **kwargs) -> "Normalizer":
        ...
