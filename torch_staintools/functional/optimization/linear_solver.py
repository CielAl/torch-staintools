import torch

from torch_staintools.constants import PARAM


def _positive_c(c: torch.Tensor, positive: bool) -> torch.Tensor:
    cond = torch.tensor(positive, device=c.device)
    return torch.where(cond, c.clamp_min(0), c)

def lstsq_solver(od_flatten: torch.Tensor,
                 dictionary: torch.Tensor, positive: bool):
    """Least squares solver for concentration.

    Warnings:
        for cusolver backend, algorithm = 'ls' May fail on GPU for individual large input (e.g., 1000 x 1000),
        regardless of batch size. To use 'ls' on large image, consider using magma backend:
        ```torch.backends.cuda.preferred_linalg_library('magma')```

    Args:
        od_flatten: B x num_pixels x C
        dictionary: Transpose of stain matrix. B x C x num_stain
        positive: enforce positive concentration

    Returns:
        concentration (flattened):  B x num_pixels x num_stains
    """

    c = torch.linalg.lstsq(dictionary, od_flatten.mT)[0].mT
    return _positive_c(c, positive)

def pinv_solver(od_flatten: torch.Tensor,
                dictionary: torch.Tensor,
                positive: bool):
    """Pseudo-inverse solver for concentration.

    Args:
        od_flatten: B x num_pixels x C
        dictionary: Transpose of stain matrix. B x C x num_stains
        positive: enforce positive concentration

    Returns:
        concentration (flattened):  B x num_pixels x num_stains
    """
    # tol = 1e-6 * dictionary.abs().amax(dim=(-1, -2), keepdim=True)
    mag = dictionary.abs().amax(dim=(-1, -2))
    tol = PARAM.SOLVER_PINV_ATOL_SCALE * torch.quantile(mag,
                                                        PARAM.SOLVER_PINV_ATOL_QUANTILE)

    proj = torch.linalg.pinv(dictionary.mT, rtol=0, atol=tol)
    c = torch.matmul(od_flatten, proj)
    return _positive_c(c, positive)


def qr_solver(od_flatten, dictionary, positive: bool):
    """QR solver for concentration.

    Args:
        od_flatten: B x num_pixels x C
        dictionary: Transpose of stain matrix. B x C x num_stains
        positive: enforce positive concentration

    Returns:
        concentration (flattened):  B x num_pixels x num_stains
    """
    A = dictionary
    Q, R = torch.linalg.qr(A, mode="reduced")

    rhs = Q.mT @ od_flatten.mT
    # add diagonal eps to avoid 0-div issue
    # somehow this is essential to give numerically stable output.
    eps = torch.finfo(R.dtype).eps
    R = R + eps * torch.eye(R.size(-1), device=R.device, dtype=R.dtype)[None, ...]
    c = torch.linalg.solve_triangular(R, rhs, upper=True).mT
    return _positive_c(c, positive)


def qr_solver_two_stain(od_flatten, dictionary, positive: bool):
    """QR solver for concentration (hardcoded 2-stain computation)

    Args:
        od_flatten: B x num_pixels x C
        dictionary: Transpose of stain matrix. B x C x num_stains
        positive: enforce positive concentration

    Returns:
        concentration (flattened):  B x num_pixels x num_stains
    """
    A = dictionary
    Q, R = torch.linalg.qr(A, mode='reduced')
    rhs = torch.matmul(Q.mT, od_flatten.mT)
    r00 = R[:, 0:1, 0:1] + torch.finfo(R.dtype).eps
    r01 = R[:, 0:1, 1:2]
    r11 = R[:, 1:2, 1:2] + torch.finfo(R.dtype).eps
    z0 = rhs[:, 0:1, :]
    z1 = rhs[:, 1:2, :]
    x1 = z1 / r11
    x0 = (z0 - r01 * x1) / r00
    return torch.cat([x0, x1], dim=1).mT