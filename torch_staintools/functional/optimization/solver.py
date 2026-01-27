"""
code directly adapted from https://github.com/rfeinman/pytorch-lasso
"""
from typing import Tuple

import torch
from ..eps import get_eps
import torch.nn.functional as F

from .sparse_util import as_scalar


def coord_descent(x: torch.Tensor, z0: torch.Tensor, weight: torch.Tensor,
                  alpha: float = 1.0,
                  maxiter: int = 50, tol: float = 1e-6,
                  positive_code: bool = False):
    """ modified coord_descent"""
    input_dim, code_dim = weight.shape  # [D,K]
    batch_size, input_dim1 = x.shape  # [N,D]
    assert input_dim1 == input_dim
    tol = tol * code_dim
    if z0 is None:
        z = x.new_zeros(batch_size, code_dim)  # [N,K]
    else:
        assert z0.shape == (batch_size, code_dim)
        z = z0

    b = torch.mm(x, weight)  # [N,K]

    # precompute S = I - W^T @ W
    S = - torch.mm(weight.T, weight)  # [K,K]
    S.diagonal().add_(1.)


    def cd_update(z, b):
        if positive_code:
            z_next = (b - alpha).clamp_min(0)
        else:
            z_next = F.softshrink(b, alpha)  # [N,K]
        z_diff = z_next - z  # [N,K]
        k = z_diff.abs().argmax(1)  # [N]
        kk = k.unsqueeze(1)  # [N,1]
        b = b + S[:, k].T * z_diff.gather(1, kk)  # [N,K] += [N,K] * [N,1]
        z = z.scatter(1, kk, z_next.gather(1, kk))
        return z, b

    active = torch.arange(batch_size, device=weight.device)
    for i in range(maxiter):
        if len(active) == 0:
            break
        z_old = z[active]
        z_new, b[active] = cd_update(z_old, b[active])
        update = (z_new - z_old).abs().sum(1)
        z[active] = z_new
        active = active[update > tol]

    z = F.softshrink(b, alpha)
    return z

def rss_grad(z_k: torch.Tensor, x: torch.Tensor, weight: torch.Tensor):
    resid = torch.matmul(z_k, weight.T) - x
    return torch.matmul(resid, weight)


def softshrink(x: torch.Tensor, lambd: torch.Tensor) -> torch.Tensor:
    lambd = lambd.clamp_min(0)
    return x.sign() * (x.abs() - lambd).clamp_min(0)


def ista_step(
    z: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    alpha: torch.Tensor,
    lr: torch.Tensor,
    positive: bool,
) -> torch.Tensor:
    """

    Args:
        z: code. num_pixels x num_stain
        x: OD space. num_pixels x num_channel
        weight: init from stain matrix --> num_channel x num_stain
        alpha: tensor form of the ista penalizer
        lr: tensor form of step size
        positive: if force z to be positive
    Returns:

    """


    z_k_safe = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    g = rss_grad(z_k_safe, x, weight) # same shape as z
    g_safe = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    # guard lr
    lr_safe = torch.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
    z_proposal = z - lr * g_safe
    threshold = alpha * lr
    if positive:
        z_next = (z_proposal - threshold).clamp_min(0)
    else:
        # z_next = F.softshrink(z_prev - lr * rss_grad(z_prev, x, weight), alpha * lr)
        z_next = softshrink(z_k_safe - lr_safe * g_safe, alpha * lr_safe)
    finite_mask = torch.isfinite(z) & torch.isfinite(g) & torch.isfinite(lr)
    return torch.where(finite_mask, z_next, z)


def fista_step(
        z: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        alpha: torch.Tensor,
        lr: torch.Tensor,
        positive_code: bool,
        tol: float
):

    z_next = ista_step(y, x, weight, alpha, lr, positive_code)
    delta = z_next - z
    diff = delta.abs().sum()
    just_finished = diff <= tol
    t_next = (1 + torch.sqrt(1 + 4 * t ** 2)) / 2
    m = (t - 1) / t_next
    y_next = z_next + m * delta
    return z_next, y_next, t_next, just_finished



@torch.compile
def ista_loop(z: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
              alpha: torch.Tensor, lr: torch.Tensor,
              tol: float, maxiter: int, positive_code: bool):
    is_converged = torch.tensor(False, device=z.device, dtype=torch.bool)
    for _ in range(maxiter):
        z_next = ista_step(z, x, weight, alpha, lr, positive_code)
        # check convergence
        diff = (z - z_next).abs().sum()
        just_finished = diff <= tol
        # lock the status - so in future loops is_converged will never be False again
        is_converged = is_converged | just_finished
        z = torch.where(is_converged, z, z_next)
    return z


@torch.compile
def fista_loop(
        z: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        alpha: torch.Tensor,
        lr: torch.Tensor,
        tol: float,
        maxiter: int,
        positive_code: bool = True,
) -> torch.Tensor:
    """  FISTA Loop

    Args:
        z: Initial guess
        x: Data input (OD space)
        weight: Dictionary matrix
        alpha: Regularization strength
        lr: Learning rate
        maxiter: Maximum iterations
        tol: Convergence tolerance
        positive_code:
    """

    # momentum
    y = z.clone()
    # step size for the momentum
    t = torch.tensor(1, dtype=z.dtype).to(z.device)
    is_converged = torch.tensor(False, device=z.device, dtype=torch.bool)

    for i in range(maxiter):

        z_next, y_next, t_next, just_finished = fista_step(z, y, t,
                                                           x, weight,
                                                           alpha, lr,
                                                           positive_code, tol)

        z = torch.where(is_converged, z, z_next)
        y = torch.where(is_converged, y, y_next)
        t = torch.where(is_converged, t, t_next)
        is_converged = is_converged | just_finished

    return z


def ista(x, z0, weight, alpha=0.01, lr: str | float = 'auto',
         maxiter: int = 50,
         tol: float = 1e-5, positive_code: bool = False):
    """ISTA solver

    Args:
        x: data
        z0: code, or the initialization mode of the code.
        weight: dict
        alpha: penalty term for code
        lr: learning rate/step size. If `auto` then it will be specified by
            the Lipschitz constant of f(z) = ||Wz - x||^2
        maxiter: max number of iteration if not converge.
        tol: tolerance term of convergence test.
        positive_code: whether enforce the positive z constraint
    Returns:

    """
    # lr, alpha, tol = collate_params(z0, x, lr, weight, alpha, tol)
    z0 = z0.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()

    return ista_loop(z0, x, weight, alpha, lr, tol, maxiter, positive_code)


def fista(x: torch.Tensor, z0: torch.Tensor,
          weight: torch.Tensor,
          alpha: torch.Tensor, lr: str | float = 'auto',
          maxiter: int = 50,
          tol: float = 1e-5, positive_code: bool = False):
    """Fast ISTA solver

    Args:
        x: data
        z0: code, or the initialization mode of the code.
        weight: dict
        alpha: penalty term for code
        lr: learning rate/step size. If `auto` then it will be specified by
            the Lipschitz constant of f(z) = ||Wz - x||^2
        maxiter: max number of iteration if not converge.
        tol: tolerance term of convergence test.
        positive_code: whether enforce the positive z constraint
    Returns:

    """
   #  lr, alpha, tol = collate_params(z0, x, lr, weight, alpha, tol)
    z0 = z0.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()

    return fista_loop(z0, x, weight, alpha, lr, tol, maxiter, positive_code)