from torch_staintools.functional.stain_extraction.factory import build_from_name
from torch import nn
import torch
from torch_staintools.functional.optimization.dict_learning import get_concentrations
from torch_staintools.functional.stain_extraction.extractor import BaseExtractor
from torch_staintools.functional.utility.implementation import transpose_trailing, img_from_concentration
from torch_staintools.functional.tissue_mask import get_tissue_mask
from operator import mul
from functools import reduce
from typing import Optional, Sequence, Tuple
import multiprocessing as mp
import ctypes
import numpy as np


class Augmentor(nn.Module):
    use_cache: bool
    target_stain_idx: Optional[Sequence[int]]
    rng: torch.Generator

    reconst_method: str
    get_stain_matrix: BaseExtractor  # can be any callable following the signature of BaseExtractor's __call__
    target_concentrations: torch.Tensor

    sigma_alpha: float
    sigma_beta: float

    num_stains: int
    luminosity_threshold: float
    regularizer: float

    def __init__(self, get_stain_matrix: BaseExtractor, reconst_method: str = 'ista',
                 rng: Optional[int | torch.Generator] = None,
                 target_stain_idx: Optional[Sequence[int]] = (0, 1),
                 sigma_alpha: float = 0.2,
                 sigma_beta: float = 0.2,
                 num_stains: int = 2,
                 luminosity_threshold: float = 0.8,
                 regularizer: float = 0.01):
        """Augment the stain concentration by alpha * concentration + beta

        Args:
            get_stain_matrix: the Callable to obtain stain matrix - e.g., Vahadane's dict learning or
                Macenko's SVD
            reconst_method:  How to get stain concentration from stain matrix
            rng: the specified torch.Generator or int (as seed) for reproducing the results
            sigma_alpha: bound of alpha (mean 1). Sampled from (1-sigma, 1+sigma)
            sigma_beta: bound of beta (mean 0). Sampled from (-sigma, sigma)
            num_stains: number of stains to separate. 2 Recommended.
            luminosity_threshold: luminosity threshold to obtain tissue region and ignore brighter backgrounds.
                If None, all image pixels will be considered as tissue for stain matrix/concentration computation.
            regularizer: the regularizer to compute concentration used in ISTA or CD algorithm.

        """
        super().__init__()
        self.reconst_method = reconst_method
        self.get_stain_matrix = get_stain_matrix

        self.target_stain_idx = target_stain_idx
        self.rng = Augmentor._default_rng(rng)
        self.sigma_alpha = sigma_alpha
        self.sigma_beta = sigma_beta

        self.num_stains = num_stains
        self.luminosity_threshold = luminosity_threshold
        self.regularizer = regularizer

    @staticmethod
    def _default_rng(rng: Optional[torch.Generator | int]):
        if rng is None:
            return torch.Generator()
        if isinstance(rng, int):
            return torch.Generator().manual_seed(rng)
        assert isinstance(rng, torch.Generator)
        return rng

    @staticmethod
    def new_cache(shape):
        """
        Args:
            shape:

        Returns:

        """
        # Todo map the key to the corresponding cached data -- cached in file or to memory?
        #
        shared_array_base = mp.Array(ctypes.c_float, reduce(mul, shape))
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(*shape)
        return shared_array

    @staticmethod
    def __concentration_selected(target_concentration: torch.Tensor,
                                 target_stain_idx: Optional[Sequence[int]],
                                 ):
        """Return concentration of selected stain channels

        Args:
            target_concentration: B x num_stains x num_pixel_in_mask
            target_stain_idx:

        Returns:

        """
        if target_stain_idx is None:
            return target_concentration
        return target_concentration[:, target_stain_idx, :]

    @staticmethod
    def __inplace_augment_helper(target_concentration: torch.Tensor, *,
                                 tissue_mask: torch.Tensor,
                                 alpha: torch.Tensor, beta: torch.Tensor):
        """Helper function to augment a given row(s) of stain concentration: alpha * concentration + beta

        Args:
            target_concentration: B x num_stains x num_pixel
            tissue_mask: mask of tissue regions. only augment concentration within the mask
            alpha:
            beta:
        Returns:

        """
        alpha = alpha.to(target_concentration.device)
        beta = beta.to(target_concentration.device)
        tissue_mask = tissue_mask.ravel()
        target_concentration[..., tissue_mask] *= alpha
        target_concentration += beta
        return target_concentration

    @staticmethod
    def randn_range(*size, low, high, rng: torch.Generator):
        rand_num = torch.randn(*size, generator=rng)
        return low + (high - low) * rand_num

    @staticmethod
    def channel_rand(target_concentration_selected: torch.Tensor, rng: torch.Generator,
                     sigma_alpha: float,
                     sigma_beta: float)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            target_concentration_selected: concentrations to work on (e.g., the entire or a subset of concentration
                matrix
            rng: torch.Generator object
            sigma_alpha: sample alpha values in range (1-sigma, 1+ sigma)
            sigma_beta: sample beta values in range (-sigma, sigma)

        Returns:
            sampled alpha and beta as a tuple
        """
        assert target_concentration_selected.ndimension() == 3
        b, num_stain, _ = target_concentration_selected.shape
        size = (b, num_stain, 1)  # torch.randn(b, num_stain, 1, generator=rng)
        alpha = Augmentor.randn_range(*size, low=1 - sigma_alpha, high=1 + sigma_alpha, rng=rng)
        beta = Augmentor.randn_range(*size, low=-sigma_beta, high=sigma_beta, rng=rng)

        return alpha, beta

    @staticmethod
    def __inplace_tensor(target_concentration, inplace: bool) -> torch.Tensor:
        if not inplace:
            target_concentration = target_concentration.clone()
        return target_concentration

    @staticmethod
    def augment(*,
                target_concentration: torch.Tensor,
                tissue_mask: torch.Tensor,
                target_stain_idx: Optional[Sequence[int]],
                inplace: bool,
                rng: torch.Generator,
                sigma_alpha: float,
                sigma_beta: float,
                ):
        """

        Args:
            target_concentration: concentration matrix of input image. B x num_stains x num_pixel
            tissue_mask: region of the tissue
            target_stain_idx: which stain channel to operate on.
            inplace: whether augment the concentration matrix in-place
            rng: rng for alpha and beta generation
            sigma_alpha: sample values in range (-sigma, sigma)
            sigma_beta: same semantic of sigma_alpha but applied to beta

        Returns:

        """
        target_concentration = Augmentor.__inplace_tensor(target_concentration, inplace)

        target_concentration_selected = Augmentor.__concentration_selected(target_concentration, target_stain_idx)
        alpha, beta = Augmentor.channel_rand(target_concentration_selected, rng, sigma_alpha, sigma_beta)
        target_concentration = Augmentor.__inplace_augment_helper(target_concentration_selected,
                                                                  tissue_mask=tissue_mask,
                                                                  alpha=alpha, beta=beta)
        return target_concentration

    def forward(self, target: torch.Tensor, **stain_mat_kwargs):
        """

        Args:
            target: input tensor to augment. Shape B x C x H x W and intensity range is [0, 1].
            **stain_mat_kwargs: all extra keyword arguments other than regularizer/num_stains/luminosity_threshold set
                in __init__.

        Returns:
            Augmented output.
        """
        # stain_matrix_target -- B x num_stain x num_input_color_channel
        # todo cache
        target_stain_matrix = self.get_stain_matrix(target, luminosity_threshold=self.luminosity_threshold,
                                                    num_stains=self.num_stains,
                                                    regularizer=self.regularizer,
                                                    **stain_mat_kwargs)

        #  B x num_stains x num_pixel_in_mask
        concentration = get_concentrations(target, target_stain_matrix, regularizer=self.regularizer,
                                           algorithm=self.reconst_method, )
        tissue_mask = get_tissue_mask(target, luminosity_threshold=self.luminosity_threshold, throw_error=False,
                                      true_when_empty=False)
        concentration_aug = Augmentor.augment(target_concentration=concentration,
                                              tissue_mask=tissue_mask,
                                              target_stain_idx=self.target_stain_idx,
                                              inplace=False, rng=self.rng, sigma_alpha=self.sigma_alpha,
                                              sigma_beta=self.sigma_beta)
        # transpose to B x num_pixel x num_stains

        concentration_aug = transpose_trailing(concentration_aug)
        return img_from_concentration(concentration_aug, target_stain_matrix, img_shape=target.shape, out_range=(0, 1))

    @classmethod
    def build(cls,
              method: str, *, reconst_method: str = 'ista',
              rng: Optional[int | torch.Generator] = None,
              target_stain_idx: Optional[Sequence[int]] = (0, 1),
              sigma_alpha: float = 0.2,
              sigma_beta: float = 0.2):
        method = method.lower()
        extractor = build_from_name(method)
        return cls(extractor, reconst_method=reconst_method, rng=rng, target_stain_idx=target_stain_idx,
                   sigma_alpha=sigma_alpha, sigma_beta=sigma_beta)

