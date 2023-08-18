from abc import ABC, abstractmethod
import torch

from torch_stain_tools.functional.tissue_mask import get_tissue_mask
from typing import Callable
from torch_stain_tools.functional.conversion.od import rgb2od


class BaseExtractor(ABC, Callable):

    @staticmethod
    def normalize_matrix_rows(A):
        """
        Normalize the rows of an array.
        :param A: An array.
        :return: Array with rows normalized.
        """
        return A / torch.linalg.norm(A, dim=1)[:, None]

    @staticmethod
    @abstractmethod
    def get_stain_matrix_from_od(od: torch.Tensor, tissue_mask: torch.Tensor,
                                 *args, **kwargs):
        ...

    @classmethod
    def __call__(cls, image: torch.Tensor, luminosity_threshold: float = 0.8,
                 *args, **kwargs):
        # device = image.device
        # B x 1 x H x W
        tissue_mask = get_tissue_mask(image, luminosity_threshold=luminosity_threshold)  # .reshape((-1,))
        #  B x (HxWx1)

        od = rgb2od(image)
        return cls.get_stain_matrix_from_od(od, tissue_mask, *args, **kwargs)
