from typing import TypedDict, Union
from abc import abstractmethod
import torch
from torch import nn
import numpy as np
from PIL import Image

TYPE_IMAGE = Union[np.ndarray, torch.Tensor, Image.Image]


class DataInput(TypedDict):
    # todo - for other information passed in the pipeline
    img: TYPE_IMAGE
    uri: str


class Normalizer(nn.Module):

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
