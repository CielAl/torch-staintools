"""Note that some of the codes are derived from torchvahadane and staintools

"""

import torch
from torch_staintools.functional.stain_extraction.extractor import BaseExtractor
from torch_staintools.functional.optimization.dict_learning import get_concentrations
from torch_staintools.functional.stain_extraction.factory import build_from_name
from torch_staintools.functional.stain_extraction.utils import percentile
from torch_staintools.functional.utility.implementation import transpose_trailing, img_from_concentration
from .base import Normalizer


class StainSeparation(Normalizer):
    """Stain Seperation-based normalizer's interface: Macenko and Vahadane

    """
    get_stain_matrix: BaseExtractor
    stain_matrix_target: torch.Tensor
    target_concentrations: torch.Tensor

    num_stains: int
    regularizer: float

    def __init__(self, get_stain_matrix: BaseExtractor, reconst_method: str = 'ista', num_stains: int = 2,
                 luminosity_threshold: float = 0.8,
                 regularizer: float = 0.1):
        """Init

        Args:
            get_stain_matrix: the Callable to obtain stain matrix - e.g., Vahadane's dict learning or
                macenko's SVD
            reconst_method:  How to get stain concentration from stain matrix
            num_stains: number of stains to separate
            luminosity_threshold: threshold of luminosity to get tissue
            regularizer: Regularizer term in dict learning. Note that similar to staintools, for image
                reconstruction step, we also use dictionary learning to get the target stain concentration.
        """
        super().__init__()
        self.reconst_method = reconst_method
        self.get_stain_matrix = get_stain_matrix
        self.num_stains = num_stains
        self.luminosity_threshold = luminosity_threshold
        self.regularizer = regularizer

    def fit(self, target, **stainmat_kwargs):
        """Fit to a target image.

        Note that the stain matrices are registered into buffers so that it's move to specified device
        along with the nn.Module object.

        Args:
            target: BCHW. Assume it's cast to torch.float32 and scaled to [0, 1]
            **stainmat_kwargs: Extra keyword argument of stain seperator, besides the num_stains/luminosity_threshold
              that are set in the __init__

        Returns:

        """
        assert target.shape[0] == 1
        stain_matrix_target = self.get_stain_matrix(target, num_stains=self.num_stains,
                                                    regularizer=self.regularizer,
                                                    luminosity_threshold=self.luminosity_threshold,
                                                    **stainmat_kwargs)

        self.register_buffer('stain_matrix_target', stain_matrix_target)
        target_conc = get_concentrations(target, self.stain_matrix_target, regularizer=self.regularizer,
                                         algorithm=self.reconst_method)
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

    def transform(self, image: torch.Tensor, **stain_mat_kwargs):
        """Transformation operation.


        Stain matrix is extracted from source image use specified stain seperator (dict learning or svd)

        Target concentration is by default computed by dict learning for both macenko and vahadane, same as
        staintools.

        Normalize the concentration and reconstruct image to OD.

        Args:
            image: Image input must be BxCxHxW cast to torch.float32 and rescaled to [0, 1]
                Check torchvision.transforms.convert_image_dtype.
            **stain_mat_kwargs: Extra keyword argument of stain seperator besides the num_stains
                and luminosity_threshold that was already set in __init__.
                For Macenko, an angular percentile argument "perc" may be selected to separate the angles of OD
                vector projected on SVD and the x-positive axis.

        Returns:
            torch.Tensor: normalized output in BxCxHxW and float32. Note that some pixel value may exceed [0, 1] and
            must be scaled/clamped by torchvision.transforms.convert_image_dtype.
        """
        # one source matrix - multiple target
        if not hasattr(self, 'stain_matrix_source'):
            stain_matrix_source = self.get_stain_matrix(image, num_stains=self.num_stains,
                                                        regularizer=self.regularizer,
                                                        luminosity_threshold=self.luminosity_threshold,
                                                        **stain_mat_kwargs)
        else:
            stain_matrix_source = self.stain_matrix_source
        # stain_matrix_source -- B x 2 x 3 wherein B is 1. Note that the input batch size is independent of how many
        # template were used and for now we only accept one template a time. todo - multiple template for sampling later
        stain_matrix_source: torch.Tensor
        stain_matrix_source = StainSeparation.repeat_stain_mat(stain_matrix_source, image)
        # B * 2 * (HW)
        source_concentration = get_concentrations(image, stain_matrix_source, algorithm=self.reconst_method,
                                                  regularizer=self.regularizer)
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

    def forward(self, x: torch.Tensor, **stain_mat_kwargs):
        return self.transform(x, **stain_mat_kwargs)

    @classmethod
    def build(cls, method: str,
              reconst_method: str = 'ista',
              num_stains: int = 2,
              luminosity_threshold: float = 0.8,
              regularizer: float = 0.1) -> "StainSeparation":
        """Builder.

        Args:
            method: method of stain extractor name: vadahane or macenko
            reconst_method: method to obtain the concentration. default ista.
            num_stains: number of stains to separate. Currently, Macenko only supports 2.
            luminosity_threshold: luminosity threshold to ignore the background. None means all regions are considered
                as tissue.
            regularizer: regularizer term in dictionary learning for stain separation and concentration computation.
        Returns:
            StainSeparation normalizer.
        """
        method = method.lower()
        extractor = build_from_name(method)
        return cls(extractor, reconst_method=reconst_method, num_stains=num_stains,
                   luminosity_threshold=luminosity_threshold, regularizer=regularizer)
