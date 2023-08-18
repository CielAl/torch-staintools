import torch
from .extractor import BaseExtractor
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
    def get_stain_matrix_from_od(od: torch.Tensor, tissue_mask: torch.Tensor,
                                 perc: int = 1, *args, **kwargs):

        assert od.ndimension() == 4, f"{od.shape}"
        assert tissue_mask.ndimension() == 4, f"{tissue_mask.shape}"
        device = od.device
        #  B x (HxWx1)
        tissue_mask_flatten = tissue_mask.flatten(start_dim=1, end_dim=-1).to(device)
        # B x (H*W) x C
        od_flatten = od.flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)
        stain_mat_list = []
        for od_single, mask_single in zip(od_flatten, tissue_mask_flatten):
            x = od_single[mask_single]

            # compute eigenvectors
            _, eig_vecs = torch.linalg.eigh(MacenkoExtractor.cov(x.T))
            eig_vecs = eig_vecs[:, [1, 2]]

            # HW * C x C x 2 --> HW x 2
            t_hat = torch.matmul(x, eig_vecs)
            # HW
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
            stain_mat = stain_mat.T
            stain_mat_list.append(stain_mat)
        return torch.stack(stain_mat_list)
