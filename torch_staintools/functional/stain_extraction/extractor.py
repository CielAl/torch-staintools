from abc import ABC, abstractmethod
import torch

from torch_staintools.functional.tissue_mask import get_tissue_mask
from typing import Callable
from torch_staintools.functional.conversion.od import rgb2od


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
    def get_stain_matrix_from_od(od: torch.Tensor, tissue_mask: torch.Tensor, num_stains: int,
                                 *args, **kwargs):
        ...

    @classmethod
    def __call__(cls, image: torch.Tensor, *, luminosity_threshold: float = 0.8,  num_stains: int = 2,
                 regularizer: float = 0.1,
                 perc: int = 1,
                 **kwargs):
        """Interface of stain extractor

        Args:
            image: input image
            luminosity_threshold: luminosity threshold to discard background from stain computation
            num_stains: number of stains to separate. For Macenko, only 2 is supported.
            regularizer: regularization term in dictionary learning if used.
            perc: percentile for stain separation by selecting the perc and 100-perc angular components of OD vector
                projected on the first two singular vector planes.
            **kwargs: any extra keyword arguments

        Returns:

        """
        # device = image.device
        # B x 1 x H x W
        tissue_mask = get_tissue_mask(image, luminosity_threshold=luminosity_threshold)  # .reshape((-1,))
        #  B x (HxWx1)

        od = rgb2od(image)
        return cls.get_stain_matrix_from_od(od, tissue_mask, num_stains, **kwargs)
