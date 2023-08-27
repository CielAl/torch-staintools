from typing import TypedDict, Union
from abc import abstractmethod
import torch
from torch import nn
import numpy as np
from PIL import Image

TYPE_IMAGE = Union[np.ndarray, torch.Tensor, Image.Image]


class DataInput(TypedDict):
    """For future compatibility - e.g., moving average of stain matrix from same wsi which needs uri to identify.

    """
    # todo - for other information passed in the pipeline
    img: TYPE_IMAGE
    uri: str


class Normalizer(nn.Module):
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

    def __init__(self):
        super().__init__()

    @classmethod
    def build(cls, *args, **kwargs) -> "Normalizer":
        ...
