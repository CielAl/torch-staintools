import torch


def ridge(b: torch.Tensor, A: torch.Tensor, alpha: float = 1e-4):
    # right-hand side
    rhs = torch.matmul(A.T, b)
    # regularized gram matrix
    M = torch.matmul(A.T, A)
    M.diagonal().add_(alpha)
    # solve
    L, info = torch.linalg.cholesky_ex(M)
    if info != 0:
        raise RuntimeError("The Gram matrix is not positive definite. "
                           "Try increasing 'alpha'.")
    x = torch.cholesky_solve(rhs, L)
    return x


def initialize_code(x, weight, alpha, mode, rng: torch.Generator):
    """ code initialization in dictionary learning.

    The dictionary learning is to find the sparse decomposition of data X = D * A,
    wherein D is the dictionary and A is the code.

    Args:
        x: data
        weight: dictionary
        alpha: small eps term on diagonal for ridge initialization
        mode: code initialization method
        rng: torch.Generator for random initialization modes
    Returns:

    """
    n_samples = x.size(0)
    n_components = weight.size(1)
    if mode == 'zero':
        z0 = x.new_zeros(n_samples, n_components)
    elif mode == 'unif':
        z0 = x.new(n_samples, n_components).uniform_(-0.1, 0.1, generator=rng)
    elif mode == 'transpose':
        z0 = torch.matmul(x, weight)
    elif mode == 'ridge':
        z0 = ridge(x.T, weight, alpha=alpha).T
    else:
        raise ValueError("invalid init parameter '{}'.".format(mode))

    return z0
