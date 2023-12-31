from abc import ABC, abstractmethod
import torch

from torch_staintools.functional.tissue_mask import get_tissue_mask
from typing import Callable
from torch_staintools.functional.conversion.od import rgb2od
from functools import partial


class BaseExtractor(ABC, Callable):
    """Stain Extraction by stain matrix estimation.

    Regulate how a stain matrix estimation function should look like

    """

    @staticmethod
    def normalize_matrix_rows(A: torch.Tensor) -> torch.Tensor:
        """Normalize the rows of an array.
        Args:
            A: An array to normalize

        Returns:
            Array with rows normalized.
        """
        return A / torch.linalg.norm(A, dim=1)[:, None]

    @staticmethod
    @abstractmethod
    def get_stain_matrix_from_od(od: torch.Tensor, tissue_mask: torch.Tensor, num_stains: int,
                                 *args, **kwargs) -> torch.Tensor:
        """Abstract function to implement: how to estimate stain matrices from optical density vectors.

        Args:
            od: optical density image in batch (BxCxHxW)
            tissue_mask: tissue mask so that only pixels in tissue regions will be evaluated
            num_stains: number of stains.
            *args: other positional arguments
            **kwargs: other keyword arguments.

        Returns:
            output batch of stain matrices: B x num_stain x num_input_color_channel
        """
        raise NotImplementedError

    def __call__(self, image: torch.Tensor, *, luminosity_threshold: float = 0.8,  num_stains: int = 2,
                 regularizer: float = 0.1,
                 perc: int = 1,
                 rng: torch.Generator = None,
                 **kwargs) -> torch.Tensor:
        """Interface of stain extractor.  Adapted from StainTools.

        Args:
            image: input image in batch of shape - BxCxHxW
            luminosity_threshold: luminosity threshold to discard background from stain computation.
                scale of threshold are within (0, 1). Pixels with intensity in the interval (0, threshold) are
                considered as tissue. If None then all pixels are considered as tissue.
            num_stains: number of stains to separate. For Macenko, only 2 is supported.
            regularizer: regularization term in dictionary learning if used.
            perc: percentile for stain separation by selecting the perc and 100-perc angular components of OD vector
                projected on the first two singular vector planes.
            rng: torch.Generator for any random initializations incurred (e.g., if `init` is set to be unif)
            **kwargs: any extra keyword arguments

        Returns:
            Stain Matrices in shape of B x num_stains x num_input_color_channel. For H&E stain estimation, if the
            original image is RGB before converted to OD, then the stain matrix is B x 2 x 3, with the first row,
            i.e., stain_mat[:, 0, :] as H, and the second row, i.e., stain_mat[:, 1, :] as E.
        """
        # device = image.device
        # B x 1 x H x W
        tissue_mask = get_tissue_mask(image, luminosity_threshold=luminosity_threshold)  # .reshape((-1,))
        #  B x (HxWx1)

        od = rgb2od(image)
        return self.__class__.get_stain_matrix_from_od(od, tissue_mask, num_stains, **kwargs)

    def get_partial(self,
                    *, luminosity_threshold: float = 0.8, num_stains: int = 2,
                    regularizer: float = 0.1,
                    perc: int = 1,
                    rng: torch.Generator = None,
                    **kwargs
                    ) -> Callable:
        return partial(self, luminosity_threshold=luminosity_threshold,
                       num_stains=num_stains, regularizer=regularizer, perc=perc, rng=rng, **kwargs)
