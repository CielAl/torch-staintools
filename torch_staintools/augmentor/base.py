from functools import partial

import torch
from typing import Optional, Sequence, Tuple, Hashable, List
from torch_staintools.functional.concentration import ConcentrationSolver
from ..constants import CONFIG
from ..functional.stain_extraction.extractor import StainExtraction, StainAlg
from ..functional.stain_extraction.utils import percentile
from ..functional.utility.implementation import img_from_concentration
from ..functional.tissue_mask import get_tissue_mask, TissueMaskException
from ..cache.tensor_cache import TensorCache
from ..base_module.base import CachedRNGModule
from ..loggers import GlobalLoggers

logger = GlobalLoggers.instance().get_logger(__name__)

TYPE_RNG = Optional[int | torch.Generator]


class Augmentor(CachedRNGModule):
    """Basic augmentation object as a nn.Module with stain matrices cache.

    """
    device: torch.device

    # _tensor_cache: TensorCache
    # CACHE_FIELD: str = '_tensor_cache'
    rng: Optional[torch.Generator]
    target_stain_idx: Optional[Sequence[int]]

    concentration_solver: ConcentrationSolver
    get_stain_matrix: StainExtraction  # can be any callable following the signature of BaseExtractor's __call__
    target_concentrations: torch.Tensor

    sigma_alpha: float
    sigma_beta: float

    num_stains: int
    luminosity_threshold: float
    regularizer: float

    def __init__(self, stain_alg: StainAlg,
                 concentration_solver: ConcentrationSolver,
                 rng: TYPE_RNG = None,
                 target_stain_idx: Optional[Sequence[int]] = (0, 1),
                 sigma_alpha: float = 0.2,
                 sigma_beta: float = 0.2,
                 num_stains: int = 2,
                 luminosity_threshold: Optional[float] = 0.8,
                 cache: Optional[TensorCache] = None,
                 device: Optional[torch.device] = None):
        """Augment the stain concentration by alpha * concentration + beta

        Warnings:
            concentration_algorithm = 'ls' May fail on GPU for individual large input (e.g., 1000 x 1000),
            regardless of batch size. Therefore, 'ls' is better for multiple small inputs in terms of H and W.

        Args:
            stain_alg: the Callable to obtain stain matrix - e.g., Vahadane's dict learning or
                Macenko's SVD
            concentration_solver:  How to get stain concentration from stain matrix
            rng: the specified torch.Generator or int (as seed) for reproducing the results
            sigma_alpha: bound of alpha (mean 1). Sampled from (1-sigma, 1+sigma)
            sigma_beta: bound of beta (mean 0).
                Sampled from (-sigma * 99 perc of concentration, sigma * 99 perc of concentration)
            num_stains: number of stains to separate. 2 Recommended.
            luminosity_threshold: luminosity threshold to obtain tissue region and ignore brighter backgrounds.
                If None, all image pixels will be considered as tissue for stain matrix/concentration computation.
            cache: the external cache object

        """
        super().__init__(cache, device, rng)
        self.concentration_solver = concentration_solver
        self.get_stain_matrix = StainExtraction(stain_alg)

        self.target_stain_idx = target_stain_idx
        self.sigma_alpha = sigma_alpha
        self.sigma_beta = sigma_beta

        self.num_stains = num_stains
        self.luminosity_threshold = luminosity_threshold

    @staticmethod
    def __concentration_selected(target_concentration: torch.Tensor,
                                 target_stain_idx: Optional[Sequence[int]],
                                 ) -> torch.Tensor:
        """Return concentration of selected stain channels

        Args:
            target_concentration: B x num_pixel_in_mask * num_stains
            target_stain_idx: Basic indices of the selected stains. If None then returns all stain.

        Returns:
            The view of the selected stain (no copies)
        """
        if target_stain_idx is None:
            return target_concentration
        return target_concentration[..., target_stain_idx]

    @staticmethod
    def __inplace_augment_helper(target_concentration: torch.Tensor, *,
                                 tissue_mask: torch.Tensor,
                                 alpha: torch.Tensor, beta: torch.Tensor):
        """Helper function to augment a given row(s) of stain concentration: alpha * concentration + beta

        Args:
            target_concentration: B x num_pixel x num_stains
            tissue_mask: mask of tissue regions. only augment concentration within the mask
            alpha: alpha value for augmentation
            beta: beta value for augmentation

        Returns:
            augmented concentration. The result is in-place (as for the target_concentration here). It might be
            cloned beforehand in prior.
        """
        # concentration: B x num_pix x num_stain
        # B x 1 x num_stain
        alpha = alpha.to(target_concentration.device)
        beta = beta.to(target_concentration.device)
        tissue_mask = tissue_mask.to(target_concentration.device)
        mask = tissue_mask.flatten(start_dim=-2, end_dim=-1).mT.to(target_concentration.dtype)
        scale =  (1 + mask * (alpha - 1))
        bias = mask * beta
        return scale * target_concentration + bias

    @staticmethod
    def randn_range(*size, low, high, device: torch.device, rng: torch.Generator):
        """Helper function to get the uniform random float within low/high given the torch.Generator

        Args:
            *size: size of the random number
            low: lower bound (inclusive)
            high: upper bound (inclusive)
            device: device to create the random numbers
            rng: random number generator

        Returns:
            random sample given the size and the bounds using the specified rng.
        """
        rand_num = torch.rand(*size, device=device, generator=rng)
        return low + (high - low) * rand_num

    @staticmethod
    def channel_rand(target_concentration_selected: torch.Tensor, rng: torch.Generator,
                     sigma_alpha: float,
                     sigma_beta: float)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            target_concentration_selected: concentrations to work on (e.g., the entire or a subset of concentration
                matrix)
            rng: torch.Generator object
            sigma_alpha: sample alpha values in range (1-sigma, 1+ sigma)
            sigma_beta: sample beta values in range (-sigma * perc99_concentration, sigma *  * perc99_concentration)

        Returns:
            sampled alpha and beta as a tuple
        """
        assert target_concentration_selected.ndimension() == 3
        b, _, num_stain = target_concentration_selected.shape
        size = (b, 1, num_stain)  # torch.randn(b, num_stain, 1, generator=rng)
        device = target_concentration_selected.device
        # B x 1 x num_stain
        alpha = Augmentor.randn_range(*size, low=1 - sigma_alpha, high=1 + sigma_alpha, device=device, rng=rng)
        maxC = percentile(target_concentration_selected, q=99, dim=-2).clamp_min(0)
        maxC = maxC.unsqueeze(-2)
        radius = maxC * sigma_beta
        beta = Augmentor.randn_range(*size, low=-radius, high=radius, device=device, rng=rng)

        return alpha, beta

    @staticmethod
    def __inplace_tensor(target_concentration, inplace: bool) -> torch.Tensor:
        """Helper function to get the clone or the original concentration.

        Args:
            target_concentration: concentration tensor
            inplace: bool. If True then returns itself, otherwise returns a clone.

        Returns:
            The original or cloned concentration tensor
        """
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
        # B x num_pixel x num_stain
        target_concentration = Augmentor.__inplace_tensor(target_concentration, inplace)

        target_concentration_selected = Augmentor.__concentration_selected(target_concentration, target_stain_idx)
        alpha, beta = Augmentor.channel_rand(target_concentration_selected, rng, sigma_alpha, sigma_beta)
        target_concentration = Augmentor.__inplace_augment_helper(target_concentration_selected,
                                                                  tissue_mask=tissue_mask,
                                                                  alpha=alpha, beta=beta)
        return target_concentration

    def forward(self, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                cache_keys: Optional[List[Hashable]] = None):
        """

        Args:
            target: input tensor to augment. Shape B x C x H x W and intensity range is [0, 1].
            mask: Optional custom masking. Overriding the internal luminosity-based tissue masking.
            cache_keys: unique keys point the input batch to the cached stain matrices. `None` means no cache.

        Returns:
            Augmented output.
        """
        # stain_matrix_target -- B x num_stain x num_input_color_channel
        # todo cache

        get_stain_partial = partial(self.get_stain_matrix,
                                    luminosity_threshold=self.luminosity_threshold,
                                    num_stains=self.num_stains, rng=self.rng)
        # B x num_stain x num_channels
        target_stain_matrix = self.tensor_from_cache(cache_keys=cache_keys, func=get_stain_partial,
                                                     target=target)

        #  B x  num_pixel x num_stain
        concentration = self.concentration_solver(target, target_stain_matrix, rng=self.rng)
        try:
            tissue_mask = mask if mask is not None else get_tissue_mask(target,
                                          luminosity_threshold=self.luminosity_threshold,
                                          throw_error=True,
                                          true_when_empty=False)
            concentration_aug = Augmentor.augment(target_concentration=concentration,
                                                  tissue_mask=tissue_mask,
                                                  target_stain_idx=self.target_stain_idx,
                                                  inplace=False, rng=self.rng, sigma_alpha=self.sigma_alpha,
                                                  sigma_beta=self.sigma_beta)
            if CONFIG.AUG_POSITIVE_CONCENTRATION:
                concentration_aug = concentration_aug.clamp_min(0)
            # transpose to B x num_pixel x num_stains
            return img_from_concentration(concentration_aug, target_stain_matrix,
                                          img_shape=target.shape, out_range=(0, 1))
        except TissueMaskException:
            logger.error(f"Empty mask encountered. Dismiss and return the clone of input. Cache Key: {cache_keys}")
            return target.clone()
        except Exception:
            return target.clone()


    @classmethod
    def build(cls,
              stain_alg: StainAlg, *,
              concentration_solver: ConcentrationSolver,
              rng: TYPE_RNG = None,
              target_stain_idx: Optional[Sequence[int]] = (0, 1),
              sigma_alpha: float = 0.2,
              sigma_beta: float = 0.2,
              num_stains: int = 2,
              luminosity_threshold: Optional[float] = 0.8,
              use_cache: bool = False,
              cache_size_limit: int = -1,
              device: Optional[torch.device] = None,
              load_path: Optional[str] = None
              ):
        """Factory builder of the augmentor which manipulate the stain concentration by alpha * concentration + beta.

        Args:
            stain_alg: algorithm name to extract stain - support 'vahadane' or 'macenko'
            concentration_solver: method to obtain the concentration. Default 'ista' for fast sparse solution on GPU
                only applied for StainSeparation-based approaches (macenko and vahadane).
                support 'ista', 'cd', and 'ls'. 'ls' simply solves the least square problem for factorization of
                min||HExC - OD|| but is faster. 'ista'/cd enforce the sparse penalty but slower.
            rng: an optional seed (either an int or a torch.Generator) to determine the random number generation.
            target_stain_idx: what stains to augment: e.g., for HE cases, it can be either or both from [0, 1]
            sigma_alpha: alpha is uniformly randomly selected from (1-sigma_alpha, 1+sigma_alpha)
            sigma_beta: beta is uniformly randomly selected from (-sigma_beta, sigma_beta)
            num_stains: number of stains to separate.
            luminosity_threshold: luminosity threshold to find tissue regions (smaller than but positive)
                a pixel is considered as being tissue if the intensity falls in the open interval of (0, threshold).
            use_cache: whether to use cache to save the stain matrix to avoid re-computation
            cache_size_limit: size limit of the cache. negative means no limits.
            device: what device to hold the cache.
            load_path: If specified, then stain matrix cache will be loaded from the file path. See the `cache`
                module for more details.

        Returns:
            Augmentor.
        """
        cache = cls._init_cache(use_cache, cache_size_limit=cache_size_limit, device=device,
                                load_path=load_path)
        return cls(stain_alg, concentration_solver=concentration_solver, rng=rng, target_stain_idx=target_stain_idx,
                   sigma_alpha=sigma_alpha, sigma_beta=sigma_beta, num_stains=num_stains,
                   luminosity_threshold=luminosity_threshold,
                   cache=cache, device=device).to(device)
