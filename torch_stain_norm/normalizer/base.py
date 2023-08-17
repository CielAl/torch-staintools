from typing import TypedDict, Union, Callable
import torch
from torch import nn
import numpy as np
from PIL import Image
from torch_stain_norm.functional.stain_extraction.utils import percentile
from torch_stain_norm.functional.dict_learning.dict_learning import get_concentrations
TYPE_IMAGE = Union[np.ndarray, torch.Tensor, Image.Image]


class DataInput(TypedDict):
    img: TYPE_IMAGE
    uri: str


class Normalizer(nn.Module):

    def fit(self, *args, **kwargs):
        ...

    def forward(self, x: Union[DataInput, torch.Tensor], *args, **kwargs):
        ...

    def __init__(self):
        super().__init__()


class StainSeperation(Normalizer):
    get_stain_matrix: Callable
    stain_matrix_target: torch.Tensor
    target_concentrations: torch.Tensor

    def __init__(self, method):
        super().__init__()
        self.method = method

    @staticmethod
    def _transpose_trailing(mat):
        return torch.einsum("ijk -> ikj", mat)

    def fit(self, target):
        """
        Fit to a target image.

        :param target: Image RGB uint8.
        :return:
        """
        assert target.shape[0] == 1
        stain_matrix_target = self.get_stain_matrix(target)

        self.register_buffer('stain_matrix_target', stain_matrix_target)
        target_conc = get_concentrations(target, self.stain_matrix_target)
        self.register_buffer('target_concentrations', target_conc)
        # B x (HW) x 2
        conc_transpose = StainSeperation._transpose_trailing(self.target_concentrations)
        # along HW dim
        max_c_target = percentile(conc_transpose, 99, dim=1)
        self.register_buffer('maxC_target', max_c_target)
        # self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))

    def fix_source_stain_matrix(self, I):
        """there could also be a way of fixing a target matrix
        If a set of transformations are done on same wsi, stain matrix should not meaningfully change over time,
        have a moving average and if average converges to new samples then set as matrix?"""

        self.register_buffer('stain_matrix_source', self.get_stain_matrix(I))

    @staticmethod
    def repeat_stain_mat(stain_mat: torch.Tensor, image: torch.Tensor):
        stain_mat = torch.atleast_3d(stain_mat)
        repeat_dim = (image.shape[0], ) + (1, ) * (stain_mat.ndimension() - 1)
        return stain_mat.repeat(*repeat_dim)

    def transform(self, image: torch.Tensor):
        """
        Args:
            image:
        Returns:

        """
        # one source matrix - multiple target
        if not hasattr(self, 'stain_matrix_source'):
            stain_matrix_source = self.get_stain_matrix(image)
        else:
            stain_matrix_source = self.stain_matrix_source
        # stain_matrix_source -- B x 2 x 3 wherein B is 1. Note that the input batch size is independent of how many
        # template were used and for now we only accept one template a time. todo - multiple template for sampling later
        stain_matrix_source: torch.Tensor
        stain_matrix_source = StainSeperation.repeat_stain_mat(stain_matrix_source, image)
        # B * 2 * (HW)
        source_concentration = get_concentrations(image, stain_matrix_source, method=self.method)
        # individual shape (2,) (HE)
        c_transposed_srd = StainSeperation._transpose_trailing(source_concentration)
        maxC = percentile(c_transposed_srd, 99, dim=1)
        # 1 x B x 2
        c_scale = (self.maxC_target / maxC).unsqueeze(-1)
        source_concentration *= c_scale
        # note this is the reconstruction in B x (HW) x C --> need to shuffle the channel first before reshape
        out = 255 * torch.exp(-1 * torch.matmul(c_transposed_srd, self.stain_matrix_target))
        out = StainSeperation._transpose_trailing(out)
        return out.reshape(image.shape)

