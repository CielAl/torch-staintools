from typing import Optional, Tuple

import torch

from torch_staintools.functional.compile import lazy_compile
from torch_staintools.hash.hash_util import pack_bits_u64


def _angle_generic(unit_chroma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # more generic version
    # mean-center across channels
    unit_centered = unit_chroma - unit_chroma.mean(dim=1, keepdim=True)  # (B,C,H,W)
    num_channel = unit_chroma.shape[1]

    # fixed deterministic 2D projection weights over channels
    idx = torch.arange(num_channel, device=unit_chroma.device, dtype=torch.float32)
    angles = (2.0 * torch.pi * idx) / float(num_channel)
    p0 = torch.cos(angles).view(1, num_channel, 1, 1)
    p1 = torch.sin(angles).view(1, num_channel, 1, 1)

    # projected 2D coordinates
    x = (unit_centered * p0).sum(dim=1)  # (B,H,W)
    y = (unit_centered * p1).sum(dim=1)  # (B,H,W)

    return x, y


def _angle_c3(unit_chroma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # angle diff for 3-chan input. hardcode differences
    # B x H x W. Assume 3-channel input.
    x = (unit_chroma[:, 0, ...] - unit_chroma[:, 2, ...])  # (B,H,W)
    y = (unit_chroma[:, 1, ...] - unit_chroma[:, 2, ...])  # (B,H,W)
    return x, y


@lazy_compile(dynamic=True)
def _od_hash(od: torch.Tensor,
             mask: torch.Tensor,
             k_bins: int,
             c3: bool,
             ) -> torch.Tensor:
    """Helper of od_angle_hash64

    Args:
        od:
        mask:
        k_bins:
    Returns:

    """
    batch_size = od.shape[0]
    # guard
    mask = mask.to(od.device)
    eps = torch.finfo(torch.float32).eps # fp16's eps
    # mask = (mask > 0).to(od.device, dtype=torch.int32) ## count

    # larger sum --> more absorbed light
    # B x 1 x H x W
    od_sum = od.sum(dim=1, keepdim=True)

    # normalize the vector
    # B x 3 x H x W
    unit_chroma = od / od_sum.clamp_min(eps)

    if c3:
        x, y = _angle_c3(unit_chroma)
    else:
        x, y = _angle_generic(unit_chroma)
    theta = torch.atan2(y, x)  # [-pi, pi]

    # bin index in [0, k_bins-1]
    t = (theta + torch.pi) * (k_bins / (2 * torch.pi))  # [0, k_bins)
    b = torch.clamp(t.to(torch.int64), 0, k_bins - 1)  # (B,H,W)

    # masked histogram
    idx = b.view(batch_size, -1)
    hist_mask = mask.view(batch_size, -1)
    # ones = hist_mask

    hist = torch.zeros(batch_size, k_bins, device=od.device, dtype=torch.int32)
    # scatter_add_ may not be deterministic
    hist.scatter_add_(1, idx, hist_mask)

    hist = hist.float()
    hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(eps)

    # bin->bit via median threshold (balanced bits)
    med = hist.median(dim=1, keepdim=True).values
    bits = (hist > med)  # (B,k_bins)

    # pack LSB-first into uint64
    # breakpoint()
    weights = (1 << torch.arange(k_bins, device=od.device, dtype=torch.int64))
    # result =  (bits * weights).sum(dim=1).to(torch.uint64)
    result = pack_bits_u64(bits)
    return result


@torch.inference_mode()
def od_angle_hash64(
    od: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute the 64bit hash from od angles.

    Args:
        od:
        mask: target masking area for tissue foreground. If None then all regions are considered tissue.
    Returns:
        torch.Tensor: the hash
    """
    # od = od.to(torch.float32) # f32 may be more robust
    assert od.ndim == 4 and od.shape[1] >= 3
    # assert k_bins <= 64
    # k_bins: number of bins. Must be <=64 for 64bit output.
    batch_size, _, height, width = od.shape
    mask = mask if mask is not None else torch.ones(batch_size, 1, height, width, device=od.device,
                                                    dtype=torch.int32)
    mask = (mask > 0).to(device=od.device, dtype=torch.int32)
    c3 = od.shape[1] == 3
    return _od_hash(od, mask, 64, c3)
