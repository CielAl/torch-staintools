import torch
from .extractor import BaseExtractor
from torch_staintools.functional.tissue_mask import get_tissue_mask
from torch_staintools.functional.conversion.od import rgb2od
from .utils import percentile


class MacenkoExtractor(BaseExtractor):

    @staticmethod
    def cov(x):
        """Covariance matrix for eigen decomposition.
        https://en.wikipedia.org/wiki/Covariance_matrix
        """
        E_x = x.mean(dim=1)
        x = x - E_x[:, None]
        return torch.mm(x, x.T) / (x.size(1) - 1)

    @staticmethod
    def angular_helper(t_hat, ):
        # todo deal with 3D angle
        raise NotImplementedError

    @staticmethod
    def stain_matrix_helper(t_hat: torch.Tensor, perc: int, eig_vecs: torch.Tensor):
        """Helper function to compute the stain matrix.

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
        phi = torch.atan2(t_hat[:, 1], t_hat[:, 0])

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

    @staticmethod
    def get_stain_matrix_from_od(od: torch.Tensor, tissue_mask: torch.Tensor, *,
                                 perc: int = 1, num_stains: int = 2, **kwargs):
        """

        Args:
            od: Image in the Optical Density space with shape of BxCxHxW
            tissue_mask: tissue mask so that only pixels in tissue regions will be evaluated
            perc: Percentile number to find the minimum angular term. min angular as perc percentile
                max angular as 100 - perc percentile.
            num_stains: number of stains to separate. For now only support 2 as it might be complicated to separate
                angular terms in a 3D or higher dimensional space.
            **kwargs: Not used. For compatibility.

        Returns:
            Stain Matrices in shape of B x num_stains x num_input_color_channel. For H&E stain estimation, if the
            original image is RGB before converted to OD, then the stain matrix is B x 2 x 3, with the first row,
            i.e., stain_mat[:, 0, :] as H, and the second row, i.e., stain_mat[:, 1, :] as E.
        """

        assert od.ndimension() == 4, f"{od.shape}"
        assert tissue_mask.ndimension() == 4, f"{tissue_mask.shape}"
        device = od.device
        if num_stains != 2:
            # todo - generic ND angular component computation to stratify the space into N parts for stain separation
            raise NotImplementedError(f"Num stains: {num_stains} not currently supported in Macenko. Only support: 2")
        #  B x (HxWx1)
        tissue_mask_flatten = tissue_mask.flatten(start_dim=1, end_dim=-1).to(device)
        # B x (H*W) x C
        od_flatten = od.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)
        max_stains = od_flatten.shape[-1]
        assert num_stains <= max_stains, f"number of stains exceeds maximum stains allowed." \
                                         f" {num_stains} vs {max_stains}"
        stain_mat_list = []
        for od_single, mask_single in zip(od_flatten, tissue_mask_flatten):
            x = od_single[mask_single]

            # compute eigenvectors
            _, eig_vecs = torch.linalg.eigh(MacenkoExtractor.cov(x.T))
            eig_vecs = eig_vecs[:, -num_stains:]

            # HW * C x C x num_stains --> HW x num_stains
            t_hat = torch.matmul(x, eig_vecs)
            # HW
            stain_mat = MacenkoExtractor.stain_matrix_helper(t_hat, perc, eig_vecs)
            stain_mat = stain_mat.T
            stain_mat_list.append(stain_mat)
        return torch.stack(stain_mat_list)

    @classmethod
    def __call__(cls, image: torch.Tensor, *, luminosity_threshold: float = 0.8,
                 regularizer: float = 0.01,
                 num_stains: int = 2, perc: int = 1,
                 rng: torch.Generator = None,
                 **kwargs):
        """Macenko stain estimation. Adapted from StainTools.

        From M Macenko et al. 'A method for normalizing histology slides for quantitative analysis'.

        Args:
            image: input image in batch of shape - BxCxHxW
            luminosity_threshold: luminosity threshold to discard background from stain computation.
                scale of threshold are within (0, 1). Pixels with intensity in the interval (0, threshold) are
                considered as tissue. If None then all pixels are considered as tissue.
            regularizer: For compatibility. Not used.
            num_stains: Number of stains to separate
            perc: Percentile number to select the max and min angles for stain separation, after projected OD vectors
                to the plane of the most significant singular vectors.
            rng: torch.Generator for any incurred random initialization. So far not used in macenko's stain separation.
            **kwargs:

        Returns:
            Stain Matrices in shape of B x num_stains x num_input_color_channel. For H&E stain estimation, if the
            original image is RGB before converted to OD, then the stain matrix is B x 2 x 3, with the first row,
            i.e., stain_mat[:, 0, :] as H, and the second row, i.e., stain_mat[:, 1, :] as E.
        """
        # device = image.device
        # B x 1 x H x W
        tissue_mask = get_tissue_mask(image, luminosity_threshold=luminosity_threshold)  # .reshape((-1,))
        #  B x (HxWx1)

        od = rgb2od(image)
        return cls.get_stain_matrix_from_od(od, tissue_mask, num_stains=num_stains, perc=perc, **kwargs)
