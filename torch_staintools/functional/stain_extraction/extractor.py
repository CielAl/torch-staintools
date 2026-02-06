import torch
from torch_staintools.functional.tissue_mask import get_tissue_mask
from typing import Callable, Protocol, runtime_checkable, Optional
from torch_staintools.functional.conversion.od import rgb2od


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
    stain_algorithm: StainAlg

    def __init__(self, stain_algorithm: StainAlg) -> None:
        self.stain_algorithm = stain_algorithm

    def __call__(self, image: torch.Tensor,
                 *, luminosity_threshold: Optional[float],
                 mask: Optional[torch.Tensor],
                 num_stains: int,
                 rng: Optional[torch.Generator],
                 ) -> torch.Tensor:
        """Interface of stain extractor.  Adapted from StainTools.

        Args:
            image: input image in batch of shape - BxCxHxW
            luminosity_threshold: luminosity threshold to discard background from stain computation.
                scale of threshold are within (0, 1). Pixels with intensity in the interval (0, threshold) are
                considered as tissue. If None then all pixels are considered as tissue.
                Ignored if ```mask``` is present.
            num_stains: number of stains to separate. For Macenko, only 2 is supported.


        Returns:
            Stain Matrices in shape of B x num_stains x num_input_color_channel. For H&E stain estimation, if the
            original image is RGB before converted to OD, then the stain matrix is B x 2 x 3, with the first row,
            i.e., stain_mat[:, 0, :] as H, and the second row, i.e., stain_mat[:, 1, :] as E.
        """
        # device = image.device
        # B x 1 x H x W

        tissue_mask = get_tissue_mask(image, mask=mask,
                                      luminosity_threshold=luminosity_threshold).contiguous()
        od = rgb2od(image).contiguous()
        return self.stain_algorithm(od, tissue_mask, num_stains, rng)
