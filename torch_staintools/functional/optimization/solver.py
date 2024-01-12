"""
code directly adapted from https://github.com/rfeinman/pytorch-lasso
"""
from scipy.sparse.linalg import eigsh
import torch
from .sparse_util import initialize_code
from ..eps import get_eps
import torch.nn.functional as F
import numpy as np


def coord_descent(x, W, z0=None, alpha=1.0, lambda1=0.01, maxiter=1000, tol=1e-6, verbose=False):
    """ modified coord_descent"""
    input_dim, code_dim = W.shape  # [D,K]
    batch_size, input_dim1 = x.shape  # [N,D]
    assert input_dim1 == input_dim
    tol = tol * code_dim
    if z0 is None:
        z = x.new_zeros(batch_size, code_dim)  # [N,K]
    else:
        assert z0.shape == (batch_size, code_dim)
        z = z0

    # initialize b
    # TODO: how should we initialize b when 'z0' is provided?
    b = torch.mm(x, W)  # [N,K]

    # precompute S = I - W^T @ W
    S = - torch.mm(W.T, W)  # [K,K]
    S.diagonal().add_(1.)

    def fn(z):
        x_hat = torch.matmul(W, z.T)
        loss = 0.5 * (x - x_hat).norm(p=2).pow(2) + z.norm(p=1) * lambda1
        return loss

    def cd_update(z, b):
        z_next = F.softshrink(b, alpha)  # [N,K]
        z_diff = z_next - z  # [N,K]
        k = z_diff.abs().argmax(1)  # [N]
        kk = k.unsqueeze(1)  # [N,1]
        b = b + S[:, k].T * z_diff.gather(1, kk)  # [N,K] += [N,K] * [N,1]
        z = z.scatter(1, kk, z_next.gather(1, kk))
        return z, b

    active = torch.arange(batch_size, device=W.device)
    for i in range(maxiter):
        if len(active) == 0:
            break
        z_old = z[active]
        z_new, b[active] = cd_update(z_old, b[active])
        update = (z_new - z_old).abs().sum(1)
        z[active] = z_new
        active = active[update > tol]
        if verbose:
            print('iter %i - loss: %0.4f' % (i, fn(F.softshrink(b, alpha))))

    z = F.softshrink(b, alpha)

    return z


def _lipschitz_constant(W):
    """find the Lipscitz constant to compute the learning rate in ISTA

    Args:
        W: weights w in f(z) = ||Wz - x||^2

    Returns:

    """
    # L = torch.linalg.norm(W, ord=2) ** 2
    # W has nan
    WtW = torch.matmul(W.t(), W)
    WtW += torch.eye(WtW.size(0)).to(W.device) * get_eps(WtW)
    # L = torch.linalg.eigvalsh(WtW)[-1]
    # scipy.sparse.linalg._eigen.arpack.arpack.ArpackError: ARPACK error 3: No shifts could be applied during
    # a cycle of the Implicitly restarted Arnoldi iteration.
    # One possibility is to increase the size of NCV relative to NEV.

    L = eigsh(WtW.detach().cpu().numpy(), k=1, which='LM', return_eigenvectors=False).item()
    if not np.isfinite(L):  # sometimes L is not finite because of potential cublas error.
        L = torch.linalg.norm(W, ord=2) ** 2
    return L


def ista(x, z0, weight, alpha=1.0, fast=True, lr='auto', maxiter=50,
         tol=1e-5, lambda1=0.01, verbose=False, rng: torch.Generator = None):
    """ISTA solver

    Args:
        x: data
        z0: code, or the initialization mode of the code.
        weight: dict
        alpha: eps term for code initialization
        fast: whether to use FISTA (fast-ista) instead of ISTA
        lr: learning rate/step size. If `auto` then it will be specified by
            the Lipschitz constant of f(z) = ||Wz - x||^2
        maxiter: max number of iteration if not converge.
        tol: tolerance term of convergence test.
        lambda1: lambda of the sparse terms.
        verbose: whether to print the progress
        rng: torch.Generator for random initialization

    Returns:

    """
    if type(z0) is str:
        z0 = initialize_code(x, weight, alpha, z0, rng=rng)

    if lr == 'auto':
        # set lr based on the maximum eigenvalue of W^T @ W; i.e. the
        # Lipschitz constant of \grad f(z), where f(z) = ||Wz - x||^2
        L = _lipschitz_constant(weight)
        lr = 1 / L
    tol = z0.numel() * tol

    def loss_fn(z_k):
        x_hat = torch.matmul(weight, z_k.T)
        loss = 0.5 * (x.T - x_hat).norm(p=2).pow(2) + z_k.norm(p=1) * lambda1
        return loss

    def rss_grad(z_k):
        resid = torch.matmul(z_k, weight.T) - x
        return torch.matmul(resid, weight)
    # optimize
    z = z0
    if fast:
        y, t = z0, torch.tensor(1, dtype=torch.float32).to(z0.device)
    for _ in range(maxiter):
        if verbose:
            print('loss: %0.4f' % loss_fn(z), "weight:", weight, "lr:", lr, "z:", z)
        # ista update
        z_prev = y if fast else z
        try:
            z_next = F.softshrink(z_prev - lr * rss_grad(z_prev), alpha * lr)
        except RuntimeError as e:
            print(e)
            print('lr error ', lr, 'did not update z')
            z_next = z_prev  # if there is a failure just reset state.

        # check convergence
        if (z - z_next).abs().sum() <= tol:
            z = z_next
            break

        # update variables
        if fast:
            t_next = (1 + torch.sqrt(1 + 4 * t ** 2)) / 2
            y = z_next + ((t - 1) / t_next) * (z_next - z)
            t = t_next
        z = z_next

    return z
