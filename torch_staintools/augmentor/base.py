from torch import nn
import torch
from typing import Optional, Sequence, Tuple, Hashable, List
from ..functional.utility.implementation import default_device
# from operator import mul
# from functools import reduce
# import multiprocessing as mp
# import ctypes
# import numpy as np
from ..functional.stain_extraction.factory import build_from_name
from ..functional.optimization.dict_learning import get_concentrations
from ..functional.stain_extraction.extractor import BaseExtractor
from ..functional.utility.implementation import transpose_trailing, img_from_concentration
from ..functional.tissue_mask import get_tissue_mask
from ..cache.tensor_cache import TensorCache
from ..loggers import GlobalLoggers

logger = GlobalLoggers.instance().get_logger(__name__)


class Augmentor(nn.Module):
    device: torch.device

    _tensor_cache: TensorCache
    CACHE_FIELD: str = '_tensor_cache'

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

    @staticmethod
    def _init_cache(use_cache: bool, cache_size_limit: int, device: Optional[torch.device] = None,
                    load_path: Optional[str] = None) -> Optional[TensorCache]:
        if not use_cache:
            return None
        return TensorCache.build(size_limit=cache_size_limit, device=device, path=load_path)

    def __init__(self, get_stain_matrix: BaseExtractor, reconst_method: str = 'ista',
                 rng: Optional[int | torch.Generator] = None,
                 target_stain_idx: Optional[Sequence[int]] = (0, 1),
                 sigma_alpha: float = 0.2,
                 sigma_beta: float = 0.2,
                 num_stains: int = 2,
                 luminosity_threshold: Optional[float] = 0.8,
                 regularizer: float = 0.1,
                 cache: Optional[TensorCache] = None,
                 device: Optional[torch.device] = None):
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
            cache: the external cache object
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

        self._tensor_cache = cache
        self.device = default_device(device)

    def to(self, device: torch.device):
        self.device = device
        if self.cache_initialized():
            self.tensor_cache.to(device)
        return super().to(device)

    @property
    def cache_size_limit(self) -> int:
        if self.cache_initialized():
            return self.tensor_cache.size_limit
        return 0

    def dump_cache(self, path: str):
        assert self.cache_initialized()
        self.tensor_cache.dump(path)

    @staticmethod
    def _default_rng(rng: Optional[torch.Generator | int]):
        if rng is None:
            return torch.Generator()
        if isinstance(rng, int):
            return torch.Generator().manual_seed(rng)
        assert isinstance(rng, torch.Generator)
        return rng

    # @staticmethod
    # def new_cache(shape):
    #     """
    #     Args:
    #         shape:
    #
    #     Returns:
    #
    #     """
    #     # Todo map the key to the corresponding cached data -- cached in file or to memory?
    #     #
    #     shared_array_base = mp.Array(ctypes.c_float, reduce(mul, shape))
    #     shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    #     shared_array = shared_array.reshape(*shape)
    #     return shared_array

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

        tissue_mask_flattened = tissue_mask.flatten(start_dim=-2, end_dim=-1).expand(target_concentration.shape)
        alpha_expanded = alpha.expand(target_concentration.shape)
        target_concentration[..., tissue_mask_flattened] *= alpha_expanded[..., tissue_mask_flattened]

        beta_expanded = beta.expand(target_concentration.shape)
        target_concentration[..., tissue_mask_flattened] += beta_expanded[..., tissue_mask_flattened]
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
                matrix)
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

    @staticmethod
    def _stain_mat_kwargs_helper(luminosity_threshold,
                                 num_stains,
                                 regularizer,
                                 **stain_mat_kwargs):
        arg_dict = {
            'luminosity_threshold': luminosity_threshold,
            'num_stains': num_stains,
            'regularizer': regularizer,
        }
        stain_mat_kwargs = {k: v for k, v in stain_mat_kwargs.items()}
        stain_mat_kwargs.update(arg_dict)
        return stain_mat_kwargs

    @staticmethod
    def stain_mat_from_cache(cache: TensorCache, *,
                             cache_keys: List[Hashable],
                             get_stain_matrix: BaseExtractor,
                             target,
                             luminosity_threshold,
                             num_stains,
                             regularizer,
                             **stain_mat_kwargs) -> torch.Tensor:
        cache_func_kwargs = Augmentor._stain_mat_kwargs_helper(luminosity_threshold, num_stains, regularizer,
                                                               **stain_mat_kwargs)
        stain_mat_list = cache.get_batch(cache_keys, get_stain_matrix, target, **cache_func_kwargs)
        if isinstance(stain_mat_list, torch.Tensor):
            return stain_mat_list

        return torch.stack(stain_mat_list, dim=0)

    def _tensor_cache_helper(self) -> Optional[TensorCache]:
        return getattr(self, Augmentor.CACHE_FIELD)

    def cache_initialized(self):
        return hasattr(self, Augmentor.CACHE_FIELD) and self._tensor_cache_helper() is not None

    @property
    def tensor_cache(self) -> Optional[TensorCache]:
        return self._tensor_cache_helper()

    def stain_matrix_helper(self,
                            *,
                            cache_keys: Optional[List[Hashable]],
                            get_stain_matrix: BaseExtractor,
                            target,
                            luminosity_threshold,
                            num_stains,
                            regularizer,
                            **stain_mat_kwargs) -> torch.Tensor:
        if not self.cache_initialized() or cache_keys is None:
            logger.debug(f'{self.cache_initialized()} + {cache_keys is None} - no cache')
            return get_stain_matrix(target, luminosity_threshold=luminosity_threshold,
                                    num_stains=num_stains,
                                    regularizer=regularizer,
                                    **stain_mat_kwargs)
        # if use cache
        assert self.cache_initialized(), f"Attempt to fetch data from cache but cache is not initialized"
        assert cache_keys is not None, f"Attempt to fetch data from cache but key is not given"
        # move fetched stain matrix to the same device of the target
        logger.debug(f"{cache_keys[0:3]}. cache initialized")
        return Augmentor.stain_mat_from_cache(cache=self.tensor_cache, cache_keys=cache_keys,
                                              get_stain_matrix=get_stain_matrix,
                                              target=target,
                                              luminosity_threshold=luminosity_threshold, num_stains=num_stains,
                                              regularizer=regularizer, **stain_mat_kwargs,
                                              ).to(target.device)

    def forward(self, target: torch.Tensor, cache_keys: Optional[List[Hashable]] = None, **stain_mat_kwargs):
        """

        Args:
            target: input tensor to augment. Shape B x C x H x W and intensity range is [0, 1].
            cache_keys: a unique key point the input entry to the cached stain matrix. `None` means no cache.
            **stain_mat_kwargs: all extra keyword arguments other than regularizer/num_stains/luminosity_threshold set
                in __init__.

        Returns:
            Augmented output.
        """
        # stain_matrix_target -- B x num_stain x num_input_color_channel
        # todo cache
        target_stain_matrix = self.stain_matrix_helper(cache_keys=cache_keys, get_stain_matrix=self.get_stain_matrix,
                                                       target=target, luminosity_threshold=self.luminosity_threshold,
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
              sigma_beta: float = 0.2,
              luminosity_threshold: Optional[float] = 0.8,
              regularizer: float = 0.1,
              use_cache: bool = False,
              cache_size_limit: int = -1,
              device: Optional[torch.device] = None,
              load_path: Optional[str] = None
              ):
        """Factory builder of the augmentor which manipulate the stain concentration by alpha * concentration + beta.

        Args:
            method: algorithm name to extract stain - support 'vahadane' or 'macenko'
            reconst_method: algorithm to compute concentration. default ista
            rng: a optional seed (either an int or a torch.Generator) to determine the random number generation.
            target_stain_idx: what stains to augment: e.g., for HE cases, it can be either or both from [0, 1]
            sigma_alpha: alpha is uniformly randomly selected from (1-sigma_alpha, 1+sigma_alpha)
            sigma_beta: beta is uniformly randomly selected from (-sigma_beta, sigma_beta)
            luminosity_threshold: luminosity threshold to find tissue regions (smaller than but positive)
                a pixel is considered as being tissue if the intensity falls in the open interval of (0, threshold).
            regularizer: regularization term in ISTA algorithm
            use_cache: whether use cache to save the stain matrix to avoid recomputation
            cache_size_limit: size limit of the cache. negative means no limits.
            device: what device to hold the cache.
            load_path: If specified, then stain matrix cache will be loaded from the file path. See the `cache`
                module for more details.

        Returns:

        """
        method = method.lower()
        extractor = build_from_name(method)
        cache = cls._init_cache(use_cache, cache_size_limit=cache_size_limit, device=device,
                                load_path=load_path)
        return cls(extractor, reconst_method=reconst_method, rng=rng, target_stain_idx=target_stain_idx,
                   sigma_alpha=sigma_alpha, sigma_beta=sigma_beta,
                   luminosity_threshold=luminosity_threshold, regularizer=regularizer,
                   cache=cache, device=device)
