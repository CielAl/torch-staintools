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
    rhs = torch.matmul(a.T, b)
    # regularized gram matrix
    M = torch.matmul(a.T, a)
    M.diagonal().add_(alpha)
    # solve
    L, info = torch.linalg.cholesky_ex(M)
    assert info == 0, "The Gram matrix is not positive definite. " +\
                      "Try increasing 'alpha'."
    x = torch.cholesky_solve(rhs, L)
    return x


def initialize_code(x: torch.Tensor, weight: torch.Tensor, mode: MODE_INIT, rng: torch.Generator):
    """ code initialization in dictionary learning.

    The dictionary learning is to find the sparse decomposition of data X = D * A,
    wherein D is the dictionary and A is the code.
    For ridge initialization, the L2 penalty is customized with constants.INIT_RIDGE_L2
    Args:
        x: data
        weight: dictionary
        mode: code initialization method
        rng: torch.Generator for random initialization modes
    Returns:

    """
    n_samples = x.size(0)
    n_components = weight.size(1)
    match mode:
        case 'zero':
            z0 = x.new_zeros(n_samples, n_components)
        case 'unif':
            z0 = x.new(n_samples, n_components).uniform_(-0.1, 0.1, generator=rng)
        case 'transpose':
            z0 = torch.matmul(x, weight)
        case 'ridge':
            z0 = ridge(x.T, weight).T
        case _:
            raise ValueError("invalid init parameter '{}'.".format(mode))
    return z0


def initialize_dict(n_features: int, n_components: int,
                    device: torch.device | str, rng: torch.Generator,
                    positive_dict: bool):
    if positive_dict:
        # uniform [0, 1]
        weight = torch.rand(n_features, n_components, device=device, generator=rng)
    else:
        weight = torch.randn(n_features, n_components, device=device, generator=rng)
    return F.normalize(weight, dim=0, eps=1e-12)


def validate_code(algorithm: METHOD_SPARSE,
                  init: Optional[MODE_INIT], z0: Optional[torch.Tensor],
                  x: torch.Tensor, weight: torch.Tensor, rng):
    # initialize code variable
    n_samples = x.size(0)
    n_components = weight.size(1)
    init = _init_defaults.get(algorithm, 'zero') if init is None else init
    if z0 is None:
        z0 = initialize_code(x, weight, mode=cast(MODE_INIT, init), rng=rng)
    assert z0.shape == (n_samples, n_components)
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
    L = torch.linalg.norm(w, ord=2) ** 2
    return L + torch.finfo(L.dtype).eps


def collate_params(x: torch.Tensor,
                   lr: Optional[float | torch.Tensor],
                   weight: torch.Tensor,
                   alpha: float | torch.Tensor,
                   tol: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if lr is None:
        L = lipschitz_constant(weight)
        lr = 1 / L

    # if tol is None:
    #     tol = PARAM.OPTIM_DEFAULT_TOL
    # handle it inside optimization.
    # tol = z0.numel() * tol

    # if alpha is None:
    #     alpha = PARAM.OPTIM_DEFAULT_SPARSE_ISTA_LAMBDA

    alpha = as_scalar(alpha, x)
    lr = as_scalar(lr, x)
    return lr, alpha, tol


def as_scalar(v: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        # will except on non-scalar
        return v.to(device=like.device, dtype=like.dtype).reshape(())
    return torch.tensor(v, device=like.device, dtype=like.dtype)
