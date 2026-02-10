import torch
from typing import Callable, Protocol, runtime_checkable, Optional


@runtime_checkable
class StainAlg(Protocol):
    """Interface of stain separation algorithms.

    """
    cfg: object

    def __init__(self, cfg):
        ...

    def __call__(self,
                 od: torch.Tensor,
                 tissue_mask: torch.Tensor,
                 num_stains: int,
                 rng: Optional[torch.Generator],
                 ) -> torch.Tensor:
        """
        Args:
            od: images in optical density. BxCxHxW.
            tissue_mask: mask of tissue regions. Bx1xCxW
            num_stains:  number of stains to separate.

        Returns:

        """
        ...

class StainExtraction(Callable):
    """Stain Extraction by stain matrix estimation.
    """

    rng: Optional[torch.Generator]
    num_stains: int
    stain_algorithm: StainAlg

    def __init__(self, stain_algorithm: StainAlg, num_stains: int, rng: Optional[torch.Generator]) -> None:
        """

        Args:
            stain_algorithm: which stain algorithm to invoke.
            num_stains: number of stains to separate. For Macenko, only 2 is supported.
            rng: torch.Generator. If None, no specific generator is used and randomness can be controlled somewhere
                outside, globally.

        """
        self.stain_algorithm = stain_algorithm
        self.num_stains = num_stains
        self.rng = rng

    def __call__(self,
                 od: torch.Tensor,
                 mask: torch.Tensor,
                 ) -> torch.Tensor:
        """Interface of stain extractor.  Adapted from StainTools.

        Args:
            od: input image in batch of shape - BxCxHxW
            mask: mask the background by 0, foreground by 1.

        Returns:
            Stain Matrices in shape of B x num_stains x num_input_color_channel. For H&E stain estimation, if the
            original image is RGB before converted to OD, then the stain matrix is B x 2 x 3, with the first row,
            i.e., stain_mat[:, 0, :] as H, and the second row, i.e., stain_mat[:, 1, :] as E.
        """
        # device = image.device
        # B x 1 x H x W
        # now directly using od and a defined mask
        # tissue_mask = get_tissue_mask(image, mask=mask, luminosity_threshold=luminosity_threshold).contiguous()
        # od = rgb2od(od).contiguous()
        assert mask is not None
        return self.stain_algorithm(od, mask, self.num_stains, self.rng)
