import torch
from .extractor import BaseExtractor
from torch_staintools.functional.tissue_mask import get_tissue_mask
from torch_staintools.functional.conversion.od import rgb2od
from .utils import percentile


class MacenkoExtractor(BaseExtractor):

    @staticmethod
    def cov(x):
        """
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
    def __call__(cls, image: torch.Tensor, luminosity_threshold: float = 0.8,
                 num_stains: int = 2, perc: int = 1,
                 **kwargs):
        # device = image.device
        # B x 1 x H x W
        tissue_mask = get_tissue_mask(image, luminosity_threshold=luminosity_threshold)  # .reshape((-1,))
        #  B x (HxWx1)

        od = rgb2od(image)
        return cls.get_stain_matrix_from_od(od, tissue_mask, num_stains=num_stains, perc=1, **kwargs)
