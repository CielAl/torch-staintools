from typing import Optional, Literal, get_args, Tuple, cast
import torch
from torch.nn import functional as F
from torch_staintools.constants import PARAM

METHOD_ISTA = Literal['ista']
METHOD_FISTA = Literal['fista']
METHOD_CD = Literal['cd']
METHOD_LS = Literal['ls']
METHOD_SPARSE = Literal[METHOD_ISTA, METHOD_CD, METHOD_FISTA]
METHOD_NON_SPARSE = Literal[METHOD_LS]
METHOD_FACTORIZE = Literal[METHOD_SPARSE, METHOD_NON_SPARSE]

_init_defaults = {
    get_args(METHOD_ISTA)[0]: 'transpose',
    get_args(METHOD_FISTA)[0]: 'transpose',
    get_args(METHOD_CD)[0]: 'zero',
}

INIT_ZERO = Literal['zero']
INIT_TRANSPOSE = Literal['transpose']
INIT_UNIF = Literal['unif']
INIT_RIDGE = Literal['ridge']
MODE_INIT = Literal[INIT_ZERO, INIT_TRANSPOSE, INIT_UNIF, INIT_RIDGE]

def ridge(b: torch.Tensor, a: torch.Tensor, alpha: Optional[float] = None):
    # right-hand side
    if alpha is None:
        alpha = PARAM.INIT_RIDGE_L2
    rhs = torch.matmul(a.mT, b)
    # regularized gram matrix
    M = torch.matmul(a.mT, a)
    # ignore batch dim
    M.diagonal(dim1=-2, dim2=-1).add_(alpha)
    # solve
    L, _ = torch.linalg.cholesky_ex(M)
    x = torch.cholesky_solve(rhs, L)
    return x


def initialize_code(x: torch.Tensor, weight: torch.Tensor, mode: MODE_INIT, rng: torch.Generator):
    """ code initialization in dictionary learning.

    The dictionary learning is to find the sparse decomposition of data X = D * A,
    wherein D is the dictionary and A is the code.
    For ridge initialization, the L2 penalty is customized with constants.INIT_RIDGE_L2
    Args:
        x: data. B x num_pixel x num_channel
        weight: dictionary. B x num_channel x num_stain.
            Essentially the transposed stain mat
        mode: code initialization method
        rng: torch.Generator for random initialization modes
    Returns:

    """
    batch_size, n_samples, n_features = x.shape
    n_components = weight.shape[-1]
    match mode:
        case 'zero':
            z0 = x.new_zeros(batch_size, n_samples, n_components)
        case 'unif':
            z0 = x.new(batch_size, n_samples, n_components).uniform_(-0.1, 0.1, generator=rng)
        case 'transpose':
            # bmm.
            z0 = torch.matmul(x, weight)
        case 'ridge':
            z0 = ridge(x.mT, weight).mT
        case _:
            raise ValueError("invalid init parameter '{}'.".format(mode))
    return z0


def initialize_dict(shape: Tuple[int, ...], *,
                    device: torch.device | str, rng: torch.Generator,
                    norm_dim: int,
                    positive_dict: bool):
    if positive_dict:
        # uniform [0, 1]
        weight = torch.rand(shape, device=device, generator=rng)
    else:
        weight = torch.randn(shape, device=device, generator=rng)
    #
    return F.normalize(weight, dim=norm_dim, eps=1e-12)


def validate_code(algorithm: METHOD_SPARSE,
                  init: Optional[MODE_INIT], z0: Optional[torch.Tensor],
                  x: torch.Tensor, weight: torch.Tensor, rng):
    # initialize code variable
    batch_size, n_samples, n_features = x.shape
    n_components = weight.shape[-1]
    init = _init_defaults.get(algorithm, 'zero') if init is None else init
    if z0 is None:
        z0 = initialize_code(x, weight, mode=cast(MODE_INIT, init), rng=rng)
    assert z0.shape == (batch_size, n_samples, n_components)
    return z0


def lipschitz_constant(w: torch.Tensor):
    """find the Lipschitz constant to compute the learning rate in ISTA

    Args:
        w: weights w in f(z) = ||Wz - x||^2

    Returns:

    """
    # L = torch.linalg.norm(W, ord=2) ** 2
    # W has nan
    # WtW = torch.matmul(w.t(), w)
    # WtW += torch.eye(WtW.size(0)).to(w.device) * get_eps(WtW)
    # L = torch.linalg.eigvalsh(WtW)[-1].squeeze()
    # L_is_finite = torch.isfinite(L).all()
    # L = torch.where(L_is_finite, L, torch.linalg.norm(w, ord=2) ** 2)
    # L = L.abs()
    L = torch.linalg.norm(w, ord=2, dim=(-2, -1)) ** 2
    return L + torch.finfo(L.dtype).eps


def collate_params(x: torch.Tensor,
                   lr: Optional[float | torch.Tensor],
                   weight: torch.Tensor,
                   alpha: float | torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    if lr is None:
        L = lipschitz_constant(weight)
        lr = 1. / L

    alpha = to_tensor(alpha, x)
    assert alpha.ndim == 0, f"alpha must be a scalar. {alpha.shape}"
    lr = to_tensor(lr, x)

    assert lr.ndim == 0 or (lr.ndim >= 1 and lr.shape[0] == x.shape[0]), f"{lr.shape}. {x.shape}"
    if lr.ndim == 1:
        lr = lr[..., None, None]
    return lr, alpha


def to_tensor(v: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        # will except on non-scalar
        dest =  v.to(device=like.device, dtype=like.dtype)
    else:
        dest = torch.tensor(v, device=like.device, dtype=like.dtype)
    assert dest.ndim == 0 or dest.shape[0] == like.shape[0], f"{dest.shape} - {like.shape}"
    return dest
