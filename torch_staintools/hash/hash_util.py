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
