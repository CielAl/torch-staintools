from typing import Callable

import torch

from torch_stain_tools.functional.optimization.dict_learning import get_concentrations
from torch_stain_tools.functional.stain_extraction.factory import build_from_name
from torch_stain_tools.functional.stain_extraction.utils import percentile
from .base import Normalizer


class StainSeperation(Normalizer):
    get_stain_matrix: Callable
    stain_matrix_target: torch.Tensor
    target_concentrations: torch.Tensor

    def __init__(self, get_stain_matrix, reconst_method: str = 'ista'):
        """
        Args:
            get_stain_matrix:
            reconst_method:  How to get stain concentration from stain matrix
        """
        super().__init__()
        self.reconst_method = reconst_method
        self.get_stain_matrix = get_stain_matrix

    @staticmethod
    def _transpose_trailing(mat):
        return torch.einsum("ijk -> ikj", mat)

    def fit(self, target, regularizer: float = 0.01, *args, **kwargs):
        """
        Fit to a target image.
        Args:
            target: BCHW
            regularizer:
            *args:
            **kwargs:

        Returns:

        """
        assert target.shape[0] == 1
        stain_matrix_target = self.get_stain_matrix(target, *args, **kwargs)

        self.register_buffer('stain_matrix_target', stain_matrix_target)
        target_conc = get_concentrations(target, self.stain_matrix_target, regularizer=regularizer,
                                         algorithm=self.reconst_method)
        self.register_buffer('target_concentrations', target_conc)
        # B x (HW) x 2
        conc_transpose = StainSeperation._transpose_trailing(self.target_concentrations)
        # along HW dim
        max_c_target = percentile(conc_transpose, 99, dim=1)
        self.register_buffer('maxC_target', max_c_target)
        # self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))

    def fix_source_stain_matrix(self, image, *args, **kwargs):
        """there could also be a way of fixing a target matrix
        If a set of transformations are done on same wsi, stain matrix should not meaningfully change over time,
        have a moving average and if average converges to new samples then set as matrix?"""

        self.register_buffer('stain_matrix_source', self.get_stain_matrix(image, *args, **kwargs))

    @staticmethod
    def repeat_stain_mat(stain_mat: torch.Tensor, image: torch.Tensor):
        stain_mat = torch.atleast_3d(stain_mat)
        repeat_dim = (image.shape[0], ) + (1, ) * (stain_mat.ndimension() - 1)
        return stain_mat.repeat(*repeat_dim)

    def transform(self, image: torch.Tensor, *args, **kwargs):
        """
        Args:
            image:
        Returns:

        """
        # one source matrix - multiple target
        if not hasattr(self, 'stain_matrix_source'):
            stain_matrix_source = self.get_stain_matrix(image, *args, **kwargs)
        else:
            stain_matrix_source = self.stain_matrix_source
        # stain_matrix_source -- B x 2 x 3 wherein B is 1. Note that the input batch size is independent of how many
        # template were used and for now we only accept one template a time. todo - multiple template for sampling later
        stain_matrix_source: torch.Tensor
        stain_matrix_source = StainSeperation.repeat_stain_mat(stain_matrix_source, image)
        # B * 2 * (HW)
        source_concentration = get_concentrations(image, stain_matrix_source, algorithm=self.reconst_method)
        # individual shape (2,) (HE)
        c_transposed_srd = StainSeperation._transpose_trailing(source_concentration)
        maxC = percentile(c_transposed_srd, 99, dim=1)
        # 1 x B x 2
        c_scale = (self.maxC_target / maxC).unsqueeze(-1)
        source_concentration *= c_scale
        # note this is the reconstruction in B x (HW) x C --> need to shuffle the channel first before reshape
        out = torch.exp(-1 * torch.matmul(c_transposed_srd, self.stain_matrix_target))
        out = StainSeperation._transpose_trailing(out)
        return out.reshape(image.shape)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.transform(x, *args, **kwargs)

    @classmethod
    def build(cls, method: str, *args, **kwargs):
        method = method.lower()
        extractor = build_from_name(method)
        return cls(extractor, *args, **kwargs)
