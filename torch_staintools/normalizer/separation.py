"""Note that some of the codes are derived from torchvahadane and staintools

"""

import torch
from torch_staintools.functional.stain_extraction.extractor import BaseExtractor
from torch_staintools.functional.optimization.dict_learning import get_concentrations, METHOD_FACTORIZE
from torch_staintools.functional.stain_extraction.factory import build_from_name
from torch_staintools.functional.stain_extraction.utils import percentile
from torch_staintools.functional.utility.implementation import transpose_trailing, img_from_concentration
from .base import Normalizer
from ..cache.tensor_cache import TensorCache
from typing import Optional, List, Hashable


class StainSeparation(Normalizer):
    """Stain Separation-based normalizer's interface: Macenko and Vahadane

    The stain matrix of the reference image (i.e., target image) will be dumped to the state_dict should torch.save().
    is used to export the normalizer's state dict.

    Warnings:
        concentration_algorithm = 'ls' May fail on GPU for individual large input (e.g., 1000 x 1000),
        regardless of batch size. Therefore, 'ls' is better for multiple small inputs in terms of H and W.
    """
    get_stain_matrix: BaseExtractor
    stain_matrix_target: torch.Tensor
    target_concentrations: torch.Tensor

    num_stains: int
    regularizer: float
    rng: torch.Generator
    concentration_method: METHOD_FACTORIZE

    def __init__(self, get_stain_matrix: BaseExtractor, concentration_method: METHOD_FACTORIZE = 'ista',
                 num_stains: int = 2,
                 luminosity_threshold: float = 0.8,
                 regularizer: float = 0.1,
                 rng: Optional[int | torch.Generator] = None,
                 cache: Optional[TensorCache] = None,
                 device: Optional[torch.device] = None):
        """Init

        Warnings:
            concentration_algorithm = 'ls' May fail on GPU for individual large input (e.g., 1000 x 1000),
            regardless of batch size. Therefore, 'ls' is better for multiple small inputs in terms of H and W.

        Args:
            get_stain_matrix: the Callable to obtain stain matrix - e.g., Vahadane's dict learning or
                macenko's SVD
            concentration_method:  How to get stain concentration from stain matrix and OD through factorization.
                support 'ista', 'cd', and 'ls'. 'ls' simply solves the least square problem for factorization of
                min||HExC - OD|| but is faster. 'ista'/cd enforce the sparse penalty but slower.
                'ls' may fail on individual large image due to resource limit.
            num_stains: number of stains to separate. For macenko only 2 is supported.
                In general cases it is recommended to set num_stains as 2.
            luminosity_threshold: luminosity threshold to ignore the background. None means all regions are considered
                as tissue.
            regularizer: Regularizer term in dict learning. Note that similar to staintools, for image
                reconstruction step, we also use dictionary learning to get the target stain concentration.
        """
        super().__init__(cache=cache, device=device, rng=rng)
        self.concentration_method = concentration_method
        self.get_stain_matrix = get_stain_matrix
        self.num_stains = num_stains
        self.luminosity_threshold = luminosity_threshold
        self.regularizer = regularizer

    def fit(self, target, concentration_method: Optional[METHOD_FACTORIZE] = None, **stainmat_kwargs):
        """Fit to a target image.

        Note that the stain matrices are registered into buffers so that it's move to specified device
        along with the nn.Module object.

        Args:
            target: BCHW. Assume it's cast to torch.float32 and scaled to [0, 1]
            concentration_method: method to obtain concentration. Use the `self.concentration_method` if not specified
                in the signature.
            **stainmat_kwargs: Extra keyword argument of stain seperator, besides the num_stains/luminosity_threshold
              that are set in the __init__

        Returns:

        """
        assert target.shape[0] == 1
        stain_matrix_target = self.get_stain_matrix(target, num_stains=self.num_stains,
                                                    regularizer=self.regularizer,
                                                    luminosity_threshold=self.luminosity_threshold,
                                                    rng=self.rng,
                                                    **stainmat_kwargs)

        self.register_buffer('stain_matrix_target', stain_matrix_target)
        target_conc = get_concentrations(target, self.stain_matrix_target, regularizer=self.regularizer,
                                         algorithm='ista', rng=self.rng)
        self.register_buffer('target_concentrations', target_conc)
        # B x (HW) x 2
        conc_transpose = transpose_trailing(self.target_concentrations)
        # along HW dim
        max_c_target = percentile(conc_transpose, 99, dim=1)
        self.register_buffer('maxC_target', max_c_target)
        # self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))

    @staticmethod
    def repeat_stain_mat(stain_mat: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Helper function for vectorization and broadcasting

        Args:
            stain_mat: a (usually source) stain matrix obtained from fitting
            image: input batch image

        Returns:
            repeated stain matrix
        """
        stain_mat = torch.atleast_3d(stain_mat)
        repeat_dim = (image.shape[0],) + (1,) * (stain_mat.ndimension() - 1)
        return stain_mat.repeat(*repeat_dim)

    def transform(self, image: torch.Tensor,
                  cache_keys: Optional[List[Hashable]] = None, **stain_mat_kwargs) -> torch.Tensor:
        """Transformation operation.

        Stain matrix is extracted from source image use specified stain seperator (dict learning or svd)
        Target concentration is by default computed by dict learning for both macenko and vahadane, same as
        staintools.
        Normalize the concentration and reconstruct image to OD.

        Args:
            image: Image input must be BxCxHxW cast to torch.float32 and rescaled to [0, 1]
                Check torchvision.transforms.convert_image_dtype.
            cache_keys: unique keys point the input batch to the cached stain matrices. `None` means no cache.
            **stain_mat_kwargs: Extra keyword argument of stain seperator besides the num_stains
                and luminosity_threshold that was already set in __init__.
                For instance, in Macenko, an angular percentile argument "perc" may be selected to separate
                the angles of OD vector projected on SVD and the x-positive axis.

        Returns:
            torch.Tensor: normalized output in BxCxHxW shape and float32 dtype. Note that some pixel value may exceed
            [0, 1] and therefore a clipping operation is applied.
        """
        # one source matrix - multiple target
        get_stain_mat_partial = self.get_stain_matrix.get_partial(luminosity_threshold=self.luminosity_threshold,
                                                                  num_stains=self.num_stains,
                                                                  regularizer=self.regularizer,
                                                                  rng=self.rng,
                                                                  **stain_mat_kwargs)
        stain_matrix_source = self.tensor_from_cache(cache_keys=cache_keys, func_partial=get_stain_mat_partial,
                                                     target=image)

        # stain_matrix_source -- B x 2 x 3 wherein B is 1. Note that the input batch size is independent of how many
        # template were used and for now we only accept one template a time. todo - multiple template for sampling later
        stain_matrix_source: torch.Tensor = torch.atleast_3d(stain_matrix_source)
        # not necessary here since stain_matrix source is computed from image. Only check potential edge case
        # such that the precomputed stain matrix is squeezed in the cache.
        if stain_matrix_source.shape[0] != image.shape[0] and stain_matrix_source.shape[0] == 1:
            stain_matrix_source = StainSeparation.repeat_stain_mat(stain_matrix_source, image)
        # B * 2 * (HW)
        source_concentration = get_concentrations(image, stain_matrix_source, algorithm=self.concentration_method,
                                                  regularizer=self.regularizer, rng=self.rng)
        # individual shape (2,) (HE)
        # note that c_transposed_src is just a view of source_concentration and therefore any inplace operation on
        # them will be reflected to each other, but this should be avoided for better readability
        c_transposed_src = transpose_trailing(source_concentration)
        maxC = percentile(c_transposed_src, 99, dim=1)
        # 1 x B x 2
        c_scale = transpose_trailing((self.maxC_target / maxC).unsqueeze(-1))

        c_transposed_src *= c_scale
        # note this is the reconstruction in B x (HW) x C --> need to shuffle the channel first before reshape
        return img_from_concentration(c_transposed_src, self.stain_matrix_target, image.shape, (0, 1))

    def forward(self, x: torch.Tensor,
                cache_keys: Optional[List[Hashable]] = None,  **stain_mat_kwargs) -> torch.Tensor:
        """

        Args:
            x: input batch image tensor in shape of BxCxHxW
            cache_keys: unique keys point the input batch to the cached stain matrices. `None` means no cache.
            **stain_mat_kwargs: Other keyword arguments for stain matrix estimators than those defined in __init__,
                i.e., luminosity_threshold, regularizer, and num_stains.

        Returns:
            torch.Tensor: normalized output in BxCxHxW shape and float32 dtype. Note that some pixel value may exceed
            [0, 1] and therefore a clipping operation is applied.
        """
        return self.transform(x, cache_keys, **stain_mat_kwargs)

    @classmethod
    def build(cls, method: str,
              concentration_method: METHOD_FACTORIZE = 'ista',
              num_stains: int = 2,
              luminosity_threshold: float = 0.8,
              regularizer: float = 0.1,
              rng: Optional[int | torch.Generator] = None,
              use_cache: bool = False,
              cache_size_limit: int = -1,
              device: Optional[torch.device] = None,
              load_path: Optional[str] = None
              ) -> "StainSeparation":
        """Builder.

        Args:
            method: method of stain extractor name: vadahane or macenko
            concentration_method: method to obtain the concentration. default ista for computational efficiency on GPU.
                support 'ista', 'cd', and 'ls'. 'ls' simply solves the least square problem for factorization of
                min||HExC - OD|| but is faster. 'ista'/cd enforce the sparse penalty but slower.
            num_stains: number of stains to separate. Currently, Macenko only supports 2. In general cases it is
                recommended to set num_stains as 2.
            luminosity_threshold: luminosity threshold to ignore the background. None means all regions are considered
                as tissue.
            regularizer: regularizer term in ista for stain separation and concentration computation.
            rng: seed or torch.Generator for any random initialization might incur.
            use_cache: whether to use cache to save the stain matrix of input image to normalize
            cache_size_limit: size limit of the cache. negative means no limits.
            device: what device to hold the cache and the normalizer. If none the device is set to cpu.
            load_path: If specified, then stain matrix cache will be loaded from the file path. See the `cache`
                module for more details.

        Returns:
            StainSeparation normalizer.
        """
        method = method.lower()
        extractor = build_from_name(method)
        cache = cls._init_cache(use_cache, cache_size_limit=cache_size_limit, device=device,
                                load_path=load_path)
        return cls(extractor, concentration_method=concentration_method, num_stains=num_stains,
                   luminosity_threshold=luminosity_threshold, regularizer=regularizer, rng=rng,
                   cache=cache, device=device).to(device)
