import torch

from torch_staintools.constants import PARAM
from torch_staintools.functional.compile import lazy_compile


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


@lazy_compile(dynamic=True)
def qr_solver_generic(od_flatten, dictionary, positive: bool):
    """QR solver for concentration.

    Warnings:
        This is not scalable against the size of individual images compared to the 2-stain
        unrolled version ```qr_solver_two_stain```.

    Args:
        od_flatten: B x num_pixels x C
        dictionary: Transpose of stain matrix. B x C x num_stains
        positive: enforce positive concentration

    Returns:
        concentration (flattened):  B x num_pixels x num_stains
    """
    A = dictionary
    Q, R = torch.linalg.qr(A, mode="reduced")

    rhs = od_flatten @ Q

    diag = R.diagonal(dim1=-2, dim2=-1)
    eps = torch.finfo(R.dtype).eps
    diag_eps = eps * diag.abs().amax(dim=-1,)

    c = torch.zeros_like(rhs)
    num_stains = rhs.size(-1)

    # manual backsub.
    # somehow solve_triangular with left=True path is way too slow with larger images.
    # if optimize it use left=False it gives completely distorted output.
    for k in range(num_stains - 1, -1, -1):

        term = (c[..., k+1:] * R[:, k, k+1:][:, None, :]).sum(dim=-1)

        denom = (R[:, k, k] + diag_eps)[:, None]
        c[..., k] = (rhs[..., k] - term) / denom

    return _positive_c(c, positive)


def _qr2_helper(q: torch.Tensor, r: torch.Tensor, od_flatten: torch.Tensor):
    rhs = torch.matmul(q.mT, od_flatten.mT)
    eps = torch.finfo(r.dtype).eps
    r00 = r[:, 0:1, 0:1] + eps
    r01 = r[:, 0:1, 1:2]
    r11 = r[:, 1:2, 1:2] + eps
    z0 = rhs[:, 0:1, :]
    z1 = rhs[:, 1:2, :]
    x1 = z1 / r11
    x0 = (z0 - r01 * x1) / r00
    return x0, x1


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
    x0, x1 = _qr2_helper(Q, R, od_flatten)
    return _positive_c(torch.cat([x0, x1], dim=1).mT, positive)


def qr_solver(od_flatten, dictionary, positive: bool):
    """QR solver for concentration.

    Args:
        od_flatten: B x num_pixels x C
        dictionary: Transpose of stain matrix. B x C x num_stains
        positive: enforce positive concentration

    Returns:
        concentration (flattened):  B x num_pixels x num_stains
    """
    if dictionary.size(-1) == 2:
        return qr_solver_two_stain(od_flatten, dictionary, positive)
    return qr_solver_generic(od_flatten, dictionary, positive)

