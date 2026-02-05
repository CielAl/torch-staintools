"""
code directly adapted from https://github.com/rfeinman/pytorch-lasso
"""
from typing import Optional

import torch

from torch_staintools.constants import PARAM
from torch_staintools.functional.compile import lazy_compile
from torch_staintools.functional.optimization.sparse_util import collate_params

def _preprocess_input(z0: torch.Tensor,
                      x: torch.Tensor,
                      lr: Optional[float | torch.Tensor],
                      weight: torch.Tensor,
                      alpha: float | torch.Tensor,
                      ):
    lr, alpha = collate_params(x, lr, weight, alpha)
    z0 = z0.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()
    #
    # num_pix = z0.shape[-2]
    # num_stain = z0.shape[-1]
    # tol = tol * num_pix * num_stain
    return z0, x, weight, lr, alpha,


def _grad_precompute(x: torch.Tensor, weight: torch.Tensor):
    # return Hessian and bias
    return torch.matmul(weight.mT, weight), torch.matmul(x, weight)


def rss_grad(z_k: torch.Tensor, x: torch.Tensor, weight: torch.Tensor):
    # kernelize it?
    resid = torch.matmul(z_k, weight.mT) - x
    return torch.matmul(resid, weight)


def rss_grad_fast(z_k: torch.Tensor, hessian: torch.Tensor, b: torch.Tensor):
    return torch.matmul(z_k, hessian) - b


def _softshrink(x: torch.Tensor, lambd: torch.Tensor) -> torch.Tensor:
    lambd = lambd.clamp_min(0)
    return x.sign() * (x.abs() - lambd).clamp_min(0)


def softshrink(x: torch.Tensor, lambd: torch.Tensor, positive: bool) -> torch.Tensor:
    if positive:
        return (x - lambd).clamp_min(0)
    return _softshrink(x, lambd)


def cd_step(
    z: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    alpha: torch.Tensor,
    positive_code: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

    z_proposal = softshrink(b, alpha, positive_code)

    z_diff = z_proposal - z
    active_idx = z_diff.abs().argmax(dim=-1)
    active_idx_view = active_idx[..., None]

    z_diff_selected = z_diff.gather(-1, active_idx_view)

    one_hot = torch.nn.functional.one_hot(
        active_idx, num_classes=z.shape[-1]
    ).to(dtype=z.dtype)
    s_col_vec = torch.matmul(one_hot, s) #todo mT

    b_next = b + s_col_vec * z_diff_selected

    z_next_selected = z_proposal.gather(-1, active_idx_view)
    z_next = z.scatter(-1, active_idx_view, z_next_selected)

    finite_row = (
        torch.isfinite(z).all(dim=-1) &
        torch.isfinite(b).all(dim=-1) &
        torch.isfinite(z_next).all(dim=-1) &
        torch.isfinite(b_next).all(dim=-1)
    )[..., None]
    z_next = torch.where(finite_row, z_next, z)
    b_next = torch.where(finite_row, b_next, b)

    return z_next, b_next


@lazy_compile
def cd_loop(
    z: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    alpha: torch.Tensor,
    maxiter: int,
    positive_code: bool,
) -> torch.Tensor:

    # is_converged = torch.zeros_like(z[:, 0], dtype=torch.bool)
    for _ in range(maxiter):
        z_next, b_next = cd_step(z, b, s, alpha, positive_code)
        #
        # update = (z_next - z).abs().sum(dim=1)  # [N]
        # just_finished = update <= tol
        #
        # # freeze if converged. can't early break here.
        # cvf_2d = is_converged.unsqueeze(1)
        # z = torch.where(cvf_2d, z, z_next)
        # b = torch.where(cvf_2d, b, b_next)
        #
        # is_converged = is_converged | just_finished
        z = z_next
        b = b_next

    return softshrink(b, alpha, positive=positive_code)


def coord_descent(x: torch.Tensor,
                  z0: torch.Tensor,
                  weight: torch.Tensor,
                  alpha: torch.Tensor,
                  maxiter: int,
                  positive_code: bool):
    """ modified coord_descent"""
    # lr set to one to avoid L computation. Lr is not used in CD
    z0, x, weight, lr, alpha = _preprocess_input(z0, x, 1, weight, alpha)

    hessian, b = _grad_precompute(x, weight)
    code_dim = weight.shape[-1]
    batch_size = x.shape[0]
    # S = I - H
    eye = torch.eye(code_dim, device=x.device, dtype=x.dtype)[None, ...].expand(batch_size, -1, -1)
    s = eye - hessian
    z = cd_loop(z0, b, s, alpha, maxiter=maxiter, positive_code=positive_code)
    return z


def ista_step(
    z: torch.Tensor,
    hessian: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    lr: torch.Tensor,
    positive: bool,
) -> torch.Tensor:
    """

    Args:
        z: code. B x num_pixels x num_stain
        # x: OD space. B x num_pixels x num_channel
        # weight: init from stain matrix --> B x num_channel x num_stain
        hessian: precomputed wtw
        b: precomputed xw
        alpha: tensor form of the ista penalizer
        lr: tensor form of step size
        positive: if force z to be positive
    Returns:

    """


    z_k_safe = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    # g = rss_grad(z_k_safe, x, weight) # same shape as z
    # B x num_pix x num_stain
    g = rss_grad_fast(z_k_safe, hessian, b)
    g_safe = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    # guard lr
    # B
    lr_safe = torch.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
    z_proposal = z - lr_safe * g_safe
    threshold = alpha * lr_safe

    z_next = softshrink(z_proposal, threshold, positive)
    finite_mask = torch.isfinite(z) & torch.isfinite(g) & torch.isfinite(lr)
    return torch.where(finite_mask, z_next, z)


# @torch.compile
# @static_compile
@lazy_compile
def ista_loop(z: torch.Tensor, hessian: torch.Tensor, b: torch.Tensor, *,
              alpha: torch.Tensor, lr: torch.Tensor,
              maxiter: int, positive_code: bool):
    # may trigger recompile after change of batch size. be aware
    # is_converged = torch.tensor(False, device=z.device, dtype=torch.bool)
    # batch_size = z.shape[0]
    # is_converged = torch.zeros(batch_size, device=z.device, dtype=torch.bool)
    for _ in range(maxiter):
        # B x num_pix x num_stain
        z_next = ista_step(z, hessian, b, alpha, lr, positive_code)
        # B
        # diff = (z - z_next).abs().sum(dim=(-1, -2))
        # just_finished = diff <= tol
        # lock the status - so in future loops is_converged will never be False again
        # is_converged = is_converged | just_finished
        # z = torch.where(is_converged[..., None, None], z, z_next)
        z = z_next
    return z


def ista(x: torch.Tensor, z0: torch.Tensor,
         weight: torch.Tensor, alpha: torch.Tensor,
         lr: torch.Tensor,
         maxiter: int,
         positive_code: bool):
    """ISTA solver

    Args:
        x: data
        z0: code, or the initialization mode of the code.
        weight: dict
        alpha: penalty term for code
        lr: learning rate/step size. If `auto` then it will be specified by
            the Lipschitz constant of f(z) = ||Wz - x||^2
        maxiter: max number of iteration if not converge.
        positive_code: whether enforce the positive z constraint
    Returns:

    """
    z0, x, weight, lr, alpha = _preprocess_input(z0, x, lr, weight, alpha)
    hessian, b = _grad_precompute(x, weight)
    return ista_loop(z0, hessian, b, alpha=alpha, lr=lr,
                     maxiter=maxiter, positive_code=positive_code)


def fista_step(
        z: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        hessian: torch.Tensor,
        b: torch.Tensor,
        alpha: torch.Tensor,
        lr: torch.Tensor,
        positive_code: bool,
        keep: torch.Tensor,
):
    # B x num_pix x num_stain
    z_next = ista_step(y, hessian, b, alpha, lr, positive_code)
    # delta = z_next - z
    # diff = delta.abs().sum()
    # just_finished = diff <= tol
    # B x 1 x 1
    delta_z = (z_next - z)
    t_next = (1 + torch.sqrt(1 + 4 * t ** 2)) / 2
    m = (t - 1) / t_next * keep
    y_next = z_next + m * delta_z
    return z_next, y_next, t_next, delta_z


# @torch.compile
# @static_compile
@lazy_compile
def fista_loop(
        z: torch.Tensor,
        hessian: torch.Tensor,
        b: torch.Tensor,
        alpha: torch.Tensor,
        lr: torch.Tensor,
        maxiter: int,
        positive_code: bool = True,
) -> torch.Tensor:
    """  FISTA Loop

    Args:
        z: Initial guess
        # x: Data input (OD space)
        # weight: Dictionary matrix
        hessian: precomputed wtw
        b: precomputed xw
        alpha: Regularization strength
        lr: Learning rate
        maxiter: Maximum iterations
        positive_code:
    """


    y = z.clone()
    # step size for the momentum
    # B x num_pixel x num_stain
    batch_size = z.shape[0]
    # momentum
    # t = torch.tensor(1, dtype=z.dtype).to(z.device)
    t = torch.ones((batch_size, 1, 1), dtype=z.dtype, device=z.device)

    # is_converged = torch.tensor(False, device=z.device, dtype=torch.bool)
    # is_converged = torch.zeros(batch_size, device=z.device, dtype=torch.bool)
    it = torch.arange(1, maxiter + 1, dtype=z.dtype, device=z.device)
    restart_flags = (it % PARAM.OPTIM_MOMEMTUM_RESTART_INTERVAL == 0)
    one = torch.ones((), dtype=z.dtype, device=z.device)
    for i in range(maxiter):
        # is_restart = ((i + 1) % 10 == 0)
        r = restart_flags[i].to(dtype=z.dtype)
        keep = one - r
        z_next, y_next, t_next, _ = fista_step(z, y, t,
                                               hessian, b,
                                               alpha, lr,
                                               positive_code, keep)


        # diff = delta_z.abs().sum(dim=(-1, -2))
        # just_finished = diff <= tol
        # is_converged = is_converged | just_finished
        # is_converged = diff <= tol

        # mask = is_converged[..., None, None]
        # z = torch.where(mask, z, z_next)
        # y = torch.where(mask, y, y_next)
        # t = torch.where(mask, t, t_next)
        z = z_next
        y = y_next
        t = t_next * keep + r
    return z


def fista(x: torch.Tensor, z0: torch.Tensor,
          weight: torch.Tensor,
          alpha: torch.Tensor, lr: torch.Tensor,
          maxiter: int,
          positive_code):
    """Fast ISTA solver

    Args:
        x: data
        z0: code, or the initialization mode of the code.
        weight: dict
        alpha: penalty term for code
        lr: learning rate/step size. If `auto` then it will be specified by
            the Lipschitz constant of f(z) = ||Wz - x||^2
        maxiter: max number of iteration if not converge.
        positive_code: whether enforce the positive z constraint
    Returns:

    """
    z0, x, weight, lr, alpha = _preprocess_input(z0, x, lr, weight, alpha)
    hessian, b = _grad_precompute(x, weight)
    return fista_loop(z0, hessian, b, alpha, lr=lr, maxiter=maxiter, positive_code=positive_code)