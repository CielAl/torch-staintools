import torch


def normalize_matrix_rows(a: torch.Tensor) -> torch.Tensor:
    """Normalize the rows of an array.
    Args:
        a: An array to normalize

    Returns:
        Array with rows normalized.
    """
    return a / torch.linalg.norm(a, dim=1)[:, None]


def cov(x: torch.Tensor) -> torch.Tensor:
    """Covariance matrix for eigen decomposition.
    https://en.wikipedia.org/wiki/Covariance_matrix
    """
    # x: C x num_pixel
    E_x = x.mean(dim=1)
    x = x - E_x[:, None]
    return torch.mm(x, x.T) / (x.size(1) - 1)


@torch.no_grad()
def batch_masked_cov(od_flatten: torch.Tensor, mask_flatten: torch.Tensor) -> torch.Tensor:
    # mask B x num_pixel x 1
    # clamp so avoid 0div in mean and cov computation
    size_masked = mask_flatten.sum(dim=1).clamp_min(2).unsqueeze(-1)
    # B x C
    mean = (od_flatten * mask_flatten).sum(dim=1, keepdim=True) / size_masked
    # B x num_pix x C
    x = (od_flatten - mean) * mask_flatten
    return torch.bmm(x.transpose(1, 2), x) / (size_masked - 1)


def percentile(t: torch.Tensor, q: float, dim: int) -> torch.Tensor:
    """Author: adapted from https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30

    Return the ``q``-th percentile of the flattenepip d input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    Args:
        t:  Input tensor.
        q: Percentile to compute, which must be between 0 and 100 inclusive.
        dim: which dim to operate for function `tensor.kthvalue`.

    Returns:
        Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.shape[dim] - 1))  # interpolation?
    return t.kthvalue(k, dim=dim).values


@torch.no_grad()
def batch_masked_perc(phi: torch.Tensor, mask: torch.Tensor, q: int, dim: int) -> torch.Tensor:
    # fill nan. use nanquantile to ignore the nans (bg)
    # mask = mask.squeeze(-1)
    phi_filled = torch.where(mask.bool(), phi, torch.tensor(torch.inf, device=phi.device))
    # inf at the end. cut off.
    phi_sorted, _ = torch.sort(phi_filled, dim=dim)
    size_masked = mask.sum(dim=dim)
    q_float = q / 100.0
    target_indices = (q_float * (size_masked - 1)).long().clamp(min=0)

    # not friendly to torch.compile
    # torch.nanquantile(phi_masked, q_float, dim=dim, interpolation='nearest')  # B
    return phi_sorted.gather(dim, target_indices.unsqueeze(dim)).squeeze(dim)

def validate_shape(od: torch.Tensor, tissue_mask: torch.Tensor):
    assert isinstance(od, torch.Tensor)
    assert isinstance(tissue_mask, torch.Tensor)
    # batch
    assert tissue_mask.shape[0] == od.shape[0], f"{tissue_mask.shape[0]} vs {od.shape[0]}"
    # spatial
    assert tissue_mask.shape[2:] == od.shape[2:], f"{tissue_mask.shape[2:]} vs {od.shape[2:]}"
    # channel
    assert tissue_mask.shape[1] == 1 or tissue_mask.shape[1] == od.shape[1], \
        f"{tissue_mask.shape[1]} vs. {od.shape[1]}"