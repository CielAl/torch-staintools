import torch
from torch.nn import functional as F


def _avgpool_resize_area(
    x: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:

    B, C, H, W = x.shape

    H2 = ((H + h - 1) // h) * h
    W2 = ((W + w - 1) // w) * w
    pad_h = H2 - H
    pad_w = W2 - W

    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    kH = H2 // h
    kW = W2 // w
    return F.avg_pool2d(x, kernel_size=(kH, kW), stride=(kH, kW))


import torch
import torch.nn.functional as F


def pack_bits_u64(bits: torch.Tensor) -> torch.Tensor:
    """bits * weight may cause overflow in some rare case.

    Args:
        bits:

    Returns:
        torch.Tensor: B-dimensional uint64
    """

    nbits = bits.shape[-1]

    # pad to 64
    # if nbits > 64 let it throw the error here.
    pad_width = 64 - nbits
    bits64 = F.pad(bits, (0, pad_width), mode="constant", value=0)  # (B,64) bool

    device = bits.device
    w0 = (1 << torch.arange(32, device=device, dtype=torch.int64))
    w1 = (1 << torch.arange(32, device=device, dtype=torch.int64))

    lo = (bits64[:, :32].to(torch.int64) * w0).sum(dim=1).to(torch.uint64)
    hi = (bits64[:, 32:].to(torch.int64) * w1).sum(dim=1).to(torch.uint64)
    return (hi << 32) | lo
