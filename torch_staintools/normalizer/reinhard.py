import torch

from torch_staintools.functional.conversion.lab import rgb_to_lab, lab_to_rgb
from torch_staintools.normalizer.base import Normalizer
from torch_staintools.functional.tissue_mask import get_tissue_mask
from typing import Tuple, Optional
from torch_staintools.functional.eps import get_eps
from torch_staintools.functional.utility.implementation import nanstd


class ReinhardNormalizer(Normalizer):
    """Very simple Reinhard normalizer.

    """
    target_means: torch.Tensor
    target_stds: torch.Tensor
    luminosity_threshold: float

    def __init__(self, luminosity_threshold: Optional[float]):
        super().__init__(cache=None, device=None, rng=None)
        self.luminosity_threshold = luminosity_threshold

    @staticmethod
    def _mean_std_helper(image: torch.Tensor, *, mask: Optional[torch.Tensor] = None)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the channel-wise mean and std of input

        Args:
            image: BCHW scaled to [0, 1] torch.float32. Usually in LAB.
            mask: luminosity tissue mask of image. Mean and std are only computed within the tissue regions.
        Returns:
            means,
        """
        assert mask is None or mask.dtype is torch.bool, f"{mask.dtype}"
        assert image.ndimension() == 4, f"{image.shape}"
        if mask is None:
            mask = torch.ones_like(image, dtype=torch.bool)

        image_masked = image * mask
        image_masked[mask.expand_as(image_masked) == 0] = torch.nan
        means = image_masked.nanmean(dim=(2, 3), keepdim=True)
        stds = nanstd(image_masked, dim=(2, 3))
        return means, stds

    def fit(self, image: torch.Tensor):
        """Fit - compute the means and stds of template in lab space.

        Statistics are computed within tissue regions if a luminosity threshold is given to the normalizer upon
        creation.

        Args:
            image: template. BCHW. [0, 1] torch.float32.

        Returns:

        """
        # BCHW
        img_lab: torch.Tensor = rgb_to_lab(image)
        assert img_lab.ndimension() == 4 and img_lab.shape[1] == 3, f"{img_lab.shape}"
        mask = get_tissue_mask(image, luminosity_threshold=self.luminosity_threshold)
        # B1HW
        # 1, C, 1, 1
        means, stds = ReinhardNormalizer._mean_std_helper(img_lab, mask=mask)

        self.register_buffer('target_means', means)
        self.register_buffer('target_stds', stds)

    @staticmethod
    def normalize_helper(image: torch.Tensor, target_means: torch.Tensor, target_stds: torch.Tensor,
                         mask: Optional[torch.Tensor] = None):
        """Helper.

        Args:
            image: BCHW format. torch.float32 type in range [0, 1].
            target_means: channel-wise means of template
            target_stds: channel-wise stds of template
            mask: Optional luminosity tissue mask to compute the stats within masked region
        Returns:

        """
        means_input, stds_input = ReinhardNormalizer._mean_std_helper(image, mask=mask)
        return (image - means_input) * (target_stds / (stds_input + get_eps(image))) + target_means

    def transform(self, x: torch.Tensor, *args, **kwargs):
        """Normalize by (input-mean_input) * (target_std/input_std) + target_mean

        Performed in LAB space. Output is convert back to RGB

        Args:
            x: input tensor
            *args: for compatibility of interface.
            **kwargs: for compatibility of interface.

        Returns:
            output torch.float32 RGB in range [0, 1] and shape BCHW
        """
        # 1 C 1 1
        lab_input = rgb_to_lab(x)
        mask = get_tissue_mask(x, luminosity_threshold=self.luminosity_threshold, throw_error=False,
                               true_when_empty=True)
        normalized_lab = ReinhardNormalizer.normalize_helper(lab_input, self.target_means, self.target_stds,
                                                             mask)
        return lab_to_rgb(normalized_lab).clamp_(0, 1)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.transform(x)

    @classmethod
    def build(cls, luminosity_threshold: Optional[float] = None, **kwargs):
        return cls(luminosity_threshold=luminosity_threshold)
