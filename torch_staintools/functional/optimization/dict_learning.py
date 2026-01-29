"""
code directly adapted from https://github.com/rfeinman/pytorch-lasso
"""
from .solver import coord_descent, ista, fista
from .sparse_util import METHOD_SPARSE, validate_code, initialize_dict, collate_params
import torch
import torch.nn.functional as F
from typing import Optional, cast, Tuple
from ..eps import get_eps
from torch_staintools.constants import CONFIG


@torch.compile
def update_dict_cd(dictionary: torch.Tensor, x: torch.Tensor, code: torch.Tensor,
                   positive: bool = True,
                   dead_thresh=1e-7,
                   rng: torch.Generator = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update the dictionary (stain matrix) using Block Coordinate Descent algorithm.

    Can satisfy the positive constraint of dictionaries if specified.
    Side effects: code is updated inplace.

    Args:
        dictionary: Tensor of shape (n_features, n_components).
            Value of the dictionary at the previous iteration.
        x: Tensor of shape (n_samples, n_components)
            Sparse coding of the data against which to optimize the dictionary.
        code:  Tensor of shape (n_samples, n_components)
            Sparse coding of the data against which to optimize the dictionary.
        positive: Whether to enforce positivity when finding the dictionary.
        dead_thresh: Minimum vector norm before considering "degenerate"
        rng: torch.Generator for initialization of dictionary and code.

    Returns:
        torch.Tensor, torch.Tensor, corresponding to the weight and the updated code.
    """
    n_components = dictionary.size(1)

    x_hat = torch.matmul(code, dictionary.T)  # (n_samples, n_features)
    # Residuals
    R = x - x_hat
    for k in range(n_components):
        d_k = dictionary[:, k]
        z_k = code[:, k]

        # vanilla.  new_d =  (R + z*d^T)^T * z
        # new_d = R^T*z + (d*z^T)*z = R^T*z + d*(z^T*z)
        # update_term = torch.outer(z_k, d_k)
        # R += update_term
        # new_d_k = torch.mv(R.T, z_k) # target

        # R^T*z
        rtz = torch.mv(R.T, z_k)
        ztz = torch.dot(z_k, z_k)
        new_d_k = rtz + (d_k * ztz)

        if positive:
            new_d_k = torch.clamp(new_d_k, min=0)

        d_norm = torch.norm(new_d_k)
        is_dead = (d_norm < dead_thresh)

        # random reset for dead atoms
        d_k_random = torch.randn(new_d_k.shape,
                                 device=new_d_k.device,
                                 dtype=new_d_k.dtype,
                                 generator=rng)
        if positive:
            d_k_random = torch.abs(d_k_random)
        d_k_random = F.normalize(d_k_random, dim=0, eps=1e-12)

        d_k_standard = new_d_k / (d_norm + get_eps(dictionary))
        d_k_final = torch.where(is_dead, d_k_random, d_k_standard)
        z_k_final = torch.where(is_dead, torch.zeros_like(z_k), z_k)

        # fused
        # must be done before updating the dict
        r_delta = torch.outer(z_k, d_k) - torch.outer(z_k_final, d_k_final)

        dictionary[:, k] = d_k_final
        code[:, k] = z_k_final

        #R -= torch.outer(z_k_final, d_k_final)
        R += r_delta

    return dictionary, code


@torch.compile
def update_dict_ridge(x: torch.Tensor, code: torch.Tensor, lambd: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update an (unconstrained) dictionary with ridge regression

    This is equivalent to a Newton step with the (L2-regularized) squared
    error objective:
    f(V) = (1/2N) * ||Vz - x||_2^2 + (lambd/2) * ||V||_2^2

    Args:
        x:  a batch of observations with shape (n_samples, n_features)
        code: (z) a batch of code vectors with shape (n_samples, n_components)
        lambd:  weight decay parameter

    Returns:
        torch.Tensor, torch.Tensor, corresponding to the weight and the unmodified code.
    """

    rhs = torch.mm(code.T, x)
    M = torch.mm(code.T, code)
    M.diagonal().add_(lambd * x.size(0))
    L = torch.linalg.cholesky(M)
    weight = torch.cholesky_solve(rhs, L).T

    weight = F.normalize(weight, dim=0, eps=1e-12)
    return weight, code


def sparse_code(x: torch.Tensor,
                weight: torch.Tensor,
                alpha: torch.Tensor,
                z0: torch.Tensor,
                algorithm: METHOD_SPARSE,
                lr: torch.Tensor,
                maxiter: int,
                tol: float,
                positive_code: bool):
    n_samples = x.size(0)
    n_components = weight.size(1)

    assert z0.shape == (n_samples, n_components)
    # perform inference
    match algorithm:
        case 'cd':
            z = coord_descent(x, z0, weight, alpha, maxiter=maxiter, tol=tol, positive_code=positive_code)
        case 'ista':
            z = ista(x, z0, weight, alpha, lr=lr, maxiter=maxiter, tol=tol, positive_code=positive_code)
        case 'fista':
            z = fista(x, z0, weight, alpha, lr=lr, maxiter=maxiter, tol=tol, positive_code=positive_code)
        case _:
            raise ValueError("invalid algorithm parameter '{}'.".format(algorithm))
    return z

def dict_learning_loop(x: torch.Tensor,
                       z0: torch.Tensor,
                       weight: torch.Tensor,
                       alpha: torch.Tensor,
                       algorithm: METHOD_SPARSE,
                       *,
                       lambd_ridge: float,
                       steps: int,
                       rng: torch.Generator,
                       init: Optional[str],
                       lr: torch.Tensor,
                       maxiter: int,
                       tol: float, ):

    for _ in range(steps):
        # infer sparse coefficients and compute loss
        z = sparse_code(x, weight, alpha, z0, algorithm=cast(METHOD_SPARSE, algorithm),
                        lr=lr, maxiter=maxiter, tol=tol,
                        positive_code=CONFIG.DICT_POSITIVE_CODE).contiguous()
        weight = weight.contiguous()

        # use the code from previous steps if persist
        if CONFIG.DICT_PERSIST_CODE:
            z0 = z
        else:
            z0 = validate_code(algorithm, init, z0=None, x=x, weight=weight, rng=rng)

        # update dictionary
        if CONFIG.DICT_POSITIVE_DICTIONARY:
            weight, z = update_dict_cd(weight, x, z, positive=True, rng=rng)
        else:
            weight, z = update_dict_ridge(x, z, lambd=lambd_ridge)

    return weight


def dict_learning(x: torch.Tensor,
                  n_components: int,
                  algorithm: METHOD_SPARSE,
                  *, alpha: float,
                  lambd_ridge: float,
                  steps: int,
                  rng: Optional[torch.Generator],
                  init: Optional[str],
                  lr: Optional[float],
                  maxiter: int,
                  tol: float, ):
    n_samples, n_features = x.shape
    # pixel x c
    x = x.contiguous()
    # c x stain
    weight = initialize_dict(n_features=n_features, n_components=n_components, device=x.device,
                             rng=rng, positive_dict=CONFIG.DICT_POSITIVE_DICTIONARY)

    # initialize
    z0 = validate_code(algorithm, init, z0=None, x=x, weight=weight, rng=rng)
    assert z0 is not None
    lr, alpha, tol = collate_params(z0, x, lr, weight, alpha, tol)
    return dict_learning_loop(x, z0, weight, alpha, algorithm, lambd_ridge=lambd_ridge,
                              steps=steps, rng=rng, init=init, lr=lr, maxiter=maxiter, tol=tol)

