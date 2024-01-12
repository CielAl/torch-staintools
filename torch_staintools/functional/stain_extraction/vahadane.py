import torch
from torch_staintools.functional.optimization.dict_learning import dict_learning
from torch_staintools.functional.tissue_mask import get_tissue_mask
from torch_staintools.functional.conversion.od import rgb2od
from .extractor import BaseExtractor


class VahadaneExtractor(BaseExtractor):

    @staticmethod
    def get_stain_matrix_from_od(od: torch.Tensor, tissue_mask: torch.Tensor, *,
                                 regularizer: float = 0.1, lambd=0.01,
                                 num_stains: int = 2,
                                 algorithm='ista', steps=30,
                                 constrained=True, persist=True, init='ridge', verbose: bool = False,
                                 rng: torch.Generator = None) -> torch.Tensor:
        """
        Stain matrix estimation via method of:
        A. Vahadane et al. 'Structure-Preserving Color Normalization
        and Sparse Stain Separation for Histological Images'

        Args:
            od: optical density image in batch (BxCxHxW)
            tissue_mask: tissue mask so that only pixels in tissue regions will be evaluated
            regularizer: regularization term in ista for dictionary learning
            lambd: lambda term for the sparse penalty in objective of dictionary learning
            num_stains: # of stains to separate
            algorithm: which algorithm to use, iterative-shrinkage soft thresholding algorithm `ista` or
                coordinate descent `cd`.
            steps: max number of steps if still not converged
            constrained: whether to force dictionary to be positive
            persist: whether retain the previous z value for its update or initialize every time in the iteration.
            init: init method of the codes `a` in `X = D x a`. Selected from `ridge`, `zero`, `unif` (uniformly random),
                or `transpose`. Details see torch_staintools.functional.optimization.sparse_util.initialize_code
            verbose: whether to print progress messages.
            rng: torch.Generator for any random initializations incurred (e.g., if `init` is set to be unif)

        Returns:
            List of HE matrix (B*2x3 - H on the 1st row in the 2nd dimension)
        """
        # convert to od and ignore background
        # h*w, c
        assert od.ndimension() == 4, f"{od.shape}"
        assert tissue_mask.ndimension() == 4, f"{tissue_mask.shape}"
        device = od.device
        #  B x (HxWx1)
        tissue_mask_flatten = tissue_mask.flatten(start_dim=1, end_dim=-1)
        # B x (H*W) x C
        od_flatten = od.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)

        out_dict_list = list()
        for od_single, mask_single in zip(od_flatten, tissue_mask_flatten):
            x = od_single[mask_single]
            # todo add num_stains here
            dictionary, losses = dict_learning(x, n_components=num_stains, alpha=regularizer, lambd=lambd,
                                               algorithm=algorithm, device=device, steps=steps,
                                               constrained=constrained, progbar=False, persist=True, init=init,
                                               verbose=verbose, rng=rng)
        # H on first row.
            dictionary = dictionary.T
            # todo add num_stains here - sort?
            # if dictionary[0, 0] < dictionary[1, 0]:
            #     dictionary = dictionary[[1, 0], :]
            dictionary, _ = torch.sort(dictionary, dim=0, descending=True)
            out_dict_list.append(VahadaneExtractor.normalize_matrix_rows(dictionary))
        # breakpoint()
        return torch.stack(out_dict_list)

    @classmethod
    def __call__(cls, image: torch.Tensor, *, luminosity_threshold: float = 0.8,
                 regularizer: float = 0.1, num_stains: int = 2, perc: int = 1,
                 rng: torch.Generator = None,
                 **kwargs) -> torch.Tensor:
        """Use ISTA to solve Vahadane Stain matrix estimation.

        By default, the `fast` flag in ISTA implementation is set to True, so it's technically FISTA.
        From A. Vahadane et al. 'Structure-Preserving Color Normalization
        and Sparse Stain Separation for Histological Images'

        Args:
            image: batch image in shape of BxCxHxW
            luminosity_threshold:  luminosity threshold to discard background from stain computation.
                scale of threshold are within (0, 1). Pixels with intensity in the interval (0, threshold) are
                considered as tissue. If None then all pixels are considered as tissue.
            regularizer: regularization term in ISTA of dictionary learning.
            perc: Not used in Vahadane. For compatibility of other stain extractors.
            rng: torch.Generator for any random initializations incurred (e.g., if `init` is set to be unif)
        Returns:
            Stain Matrices in shape of B x num_stains x num_input_color_channel. For H&E stain estimation, if the
            original image is RGB before converted to OD, then the stain matrix is B x 2 x 3, with the first row,
            i.e., stain_mat[:, 0, :] as H, and the second row, i.e., stain_mat[:, 1, :] as E.
        """
        # convert to od and ignore background
        # h*w, c
        # device = image.device
        # B x 1 x H x W
        tissue_mask = get_tissue_mask(image, luminosity_threshold=luminosity_threshold)  # .reshape((-1,))
        #  B x (HxWx1)

        od = rgb2od(image)
        return VahadaneExtractor.get_stain_matrix_from_od(od, tissue_mask, regularizer=regularizer, algorithm='ista',
                                                          num_stains=num_stains, rng=rng, **kwargs)
