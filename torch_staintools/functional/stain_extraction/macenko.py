from typing import Callable, Optional

import torch
from .utils import percentile, batch_masked_cov, batch_masked_perc, cov, validate_shape
from dataclasses import dataclass

from ..compile import lazy_compile
from ...constants import CONFIG


@dataclass(frozen=False)
class MckCfg:
    """Configration of Macenko Stain Estimation.

    Attributes:
        perc: Percentile number to find the minimum angular term. min angular as 1 percentile
            max angular as 100 - perc percentile.
    """
    perc: int

DEFAULT_MACENKO_CONFIG = MckCfg(perc=int(1))



def stain_matrix_helper_single(t_hat: torch.Tensor, perc: int, eig_vecs: torch.Tensor):
    """Helper function to compute the stain matrix. (no vectorization)

    Separate the projected OD vectors on singular vectors (SVD of OD in Macenko paper, which is also the
    eigen vector of the covariance matrix of the OD)

    Args:
        t_hat: projection of OD on the plane of most significant singular vectors of OD.
        perc:  perc --> min angular term, 100 - perc --> max angular term
        eig_vecs: eigen vectors of the cov(OD), which may also be the singular vectors of OD.

    Returns:
        sorted stain matrix in shape of B x num_stains x num_input_color_channel. For H&E cases, the first row
        in dimension of num_stains is H and the second is E (only num_stains=2 supported for now).
    """
    phi = torch.atan2(t_hat[..., 1], t_hat[..., 0])
    # phi -> num_pix
    min_phi = percentile(phi, perc, dim=0)
    max_phi = percentile(phi, 100 - perc, dim=0)
    v_min = torch.matmul(eig_vecs, torch.stack((torch.cos(min_phi), torch.sin(min_phi)))).unsqueeze(1)
    v_max = torch.matmul(eig_vecs, torch.stack((torch.cos(max_phi), torch.sin(max_phi)))).unsqueeze(1)
    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    # noinspection PyTypeChecker
    flag: torch.Tensor = v_min[0] > v_max[0]
    stain_mat = torch.where(flag,
                            torch.cat((v_min, v_max), dim=1), torch.cat((v_max, v_min), dim=1))
    return stain_mat


def stain_mat_loop(od: torch.Tensor, tissue_mask: torch.Tensor,
                   num_stains: int, perc: int,
                   ):
    # barricade - od/tissue_must be 4D with aligned spatial size.
    # mask channel must be either aligned with od or be 1.
    validate_shape(od, tissue_mask)
    # B x (H*W) x C
    od_flatten = od.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)
    stain_mat_list = []
    # collapse the singleton C dim

    # mask: B x 1 x H x W --> need B x (HxW), so in for loop mask_single is (HxW)
    tissue_mask_flatten = tissue_mask.flatten(start_dim=1, end_dim=-1).bool()
    for od_single, mask_single in zip(od_flatten, tissue_mask_flatten):
        x = od_single[mask_single]

        # compute eigenvectors
        _, eig_vecs = torch.linalg.eigh(cov(x.T))
        eig_vecs = eig_vecs[:, -num_stains:]

        # HW * C x C x num_stains --> HW x num_stains
        t_hat = torch.matmul(x, eig_vecs)
        # HW
        # t_hat -> num_pixels x num_stain
        # eig_vecs -> C x num_stain
        stain_mat = stain_matrix_helper_single(t_hat, perc, eig_vecs)
        stain_mat = stain_mat.T
        stain_mat_list.append(stain_mat)
    return torch.stack(stain_mat_list)

def stain_matrix_helper(t_hat: torch.Tensor, mask_flatten: torch.Tensor,
                        perc: int, eig_vecs: torch.Tensor):
    """Helper function to compute the stain matrix.

    Separate the projected OD vectors on singular vectors (SVD of OD in Macenko paper, which is also the
    eigen vector of the covariance matrix of the OD)

    Args:
        t_hat: projection of OD on the plane of most significant singular vectors of OD.
            B x num_pixel. Not masked.
        mask_flatten: the flattened mask. B x num_pixel x 1.
        perc:  perc --> min angular term, 100 - perc --> max angular term. integer [0, 100].
        eig_vecs: eigen vectors of the cov(OD), which may also be the singular vectors of OD.
            B x C x num_stains

    Returns:
        sorted stain matrix in shape of B x num_stains x num_input_color_channel. For H&E cases, the first row
        in dimension of num_stains is H and the second is E (only num_stains=2 supported for now).
    """
    # batchified. t_hat as B x num_pixel x num_stains
    # phi as B x num_pixels. Unmasked at this point.
    phi = torch.atan2(t_hat[..., 1], t_hat[..., 0])
    # phi -> num_pix
    # requires mask and phi has the same number of dimension.
    # therefore collapse the final dim
    min_phi = batch_masked_perc(phi, mask_flatten.squeeze(-1), perc, dim=1)
    max_phi = batch_masked_perc(phi, mask_flatten.squeeze(-1), 100 - perc, dim=1)

    # B x 2 x 1
    rot_min = torch.stack([torch.cos(min_phi), torch.sin(min_phi)], dim=-1).unsqueeze(-1)
    rot_max = torch.stack([torch.cos(max_phi), torch.sin(max_phi)], dim=-1).unsqueeze(-1)
    # B x C x num_stain  @ B x num_stain x 1
    # = B x C x 1
    v_min = torch.bmm(eig_vecs, rot_min)
    v_max = torch.bmm(eig_vecs, rot_max)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second. (OD_red)

    flag: torch.Tensor = v_min[:, 0: 1, :] > v_max[:, 0: 1, :]
    stain_mat = torch.where(flag,
                            torch.cat((v_min, v_max), dim=-1),
                            torch.cat((v_max, v_min), dim=-1))
    return stain_mat


def stain_mat_vectorize_body(od_flatten: torch.Tensor,
                             tissue_mask_flatten: torch.Tensor,
                             num_stains: int, perc: int, ):
    # unsqueeze
    # tissue_mask_flatten = tissue_mask_flatten[..., None]
    cov_mat = batch_masked_cov(od_flatten, tissue_mask_flatten)
    _, eig_vecs = torch.linalg.eigh(cov_mat)
    eig_vecs = eig_vecs[:, :, -num_stains:]
    # unmasked. handle masking later
    t_hat = torch.bmm(od_flatten, eig_vecs)
    stain_mat = stain_matrix_helper(t_hat, tissue_mask_flatten,
                                    perc, eig_vecs)
    stain_mat = stain_mat.transpose(1, 2)
    return stain_mat


def stain_mat_vectorize(od: torch.Tensor,
                        tissue_mask: torch.Tensor,
                        num_stains: int, perc: int, ):
    # B x (H*W) x C
    validate_shape(od, tissue_mask)
    od_flatten = od.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)
    # add a singleton dim for batchification
    #  B x (HxWx1)
    tissue_mask_flatten = tissue_mask.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)
    return stain_mat_vectorize_body(od_flatten, tissue_mask_flatten, num_stains, perc)


class MacenkoAlg(Callable):
    cfg: MckCfg

    def __init__(self, cfg: MckCfg):
        super().__init__()
        self.cfg = cfg


    @staticmethod
    def angular_helper(t_hat, ):
        # todo deal with multi-dimensional scenario
        raise NotImplementedError

    def __call__(self, od: torch.Tensor,
                 tissue_mask: torch.Tensor,
                 num_stains: int,
                 rng: Optional[torch.Generator]):
        """Macenko stain estimation. Adapted from StainTools.

        From M Macenko et al. 'A method for normalizing histology slides for quantitative analysis'.


        Args:
            od: Image in the Optical Density space with shape of BxCxHxW
            tissue_mask: tissue mask so that only pixels in tissue regions will be evaluated
            num_stains: number of stains to separate. For now only support 2 as it might be complicated to separate
                angular terms in a 3D or higher dimensional space.

        Returns:
            Stain Matrices in shape of B x num_stains x num_input_color_channel. For H&E stain estimation, if the
            original image is RGB before converted to OD, then the stain matrix is B x 2 x 3, with the first row,
            i.e., stain_mat[:, 0, :] as H, and the second row, i.e., stain_mat[:, 1, :] as E.
        """

        assert od.ndimension() == 4, f"{od.shape}"
        assert tissue_mask.ndimension() == 4, f"{tissue_mask.shape}"
        perc = self.cfg.perc
        # todo - generic ND angular component computation to stratify the space into N parts for stain separation
        assert num_stains == 2, f"Num stains: {num_stains} not currently supported in Macenko. Only support: 2"

        max_stains = od.shape[-1]
        assert num_stains <= max_stains, f"number of stains exceeds maximum stains allowed." \
                                         f" {num_stains} vs {max_stains}"

        tissue_mask = tissue_mask.to(od.device)

        if CONFIG.ENABLE_VECTORIZE:
            return stain_mat_vectorize(od, tissue_mask, num_stains, perc)
        return stain_mat_loop(od, tissue_mask, num_stains, perc)


