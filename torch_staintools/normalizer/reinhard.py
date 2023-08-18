import torch

from torch_staintools.functional.conversion.lab import rgb_to_lab, lab_to_rgb
from torch_staintools.normalizer.base import Normalizer


class ReinhardNormalizer(Normalizer):
    target_means: torch.Tensor
    target_stds: torch.Tensor

    def __init__(self):
        super().__init__()

    @staticmethod
    def _mean_std_helper(image: torch.Tensor):
        """

        Args:
            image:

        Returns:

        """
        assert image.ndimension() == 4, f"{image.shape}"
        means = image.mean(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
        stds = image.std(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
        return means, stds

    def fit(self, image: torch.Tensor):
        """

        Args:
            image: BCHW. [0, 1] float.

        Returns:

        """
        # BCHW
        img_lab: torch.Tensor = rgb_to_lab(image)
        assert img_lab.ndimension() == 4 and img_lab.shape[1] == 3, f"{img_lab.shape}"
        # B1HW
        # 1, C, 1, 1
        means, stds = ReinhardNormalizer._mean_std_helper(img_lab)

        self.register_buffer('target_means', means)
        self.register_buffer('target_stds', stds)

    @staticmethod
    def normalize_helper(image: torch.Tensor, target_means: torch.Tensor, target_stds: torch.Tensor):
        means_input, stds_input = ReinhardNormalizer._mean_std_helper(image)
        return (image - means_input) * (target_stds / stds_input) + target_means

    def transform(self, x: torch.Tensor, *args, **kwargs):
        # 1 C 1 1
        lab_input = rgb_to_lab(x)
        normalized_lab = ReinhardNormalizer.normalize_helper(lab_input, self.target_means, self.target_stds)
        return lab_to_rgb(normalized_lab)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.transform(x)

    @classmethod
    def build(cls, *args, **kwargs):
        return cls()
