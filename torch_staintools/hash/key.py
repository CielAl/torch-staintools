import torch
from typing import List, Tuple

from torch_staintools.hash.color import od_angle_hash64
from torch_staintools.hash.dhash import od_dhash
from torch_staintools.hash.lbp import od_lbp8_hash


def to_uint(h: torch.Tensor) -> List[int]:
    assert h.ndim == 1 or h.numel() == h.shape[0]
    return h.tolist()


def lbp_code_to_bytes(lbp_code: torch.Tensor) -> List[bytes]:
    """
    Args:
        lbp_code: B x R x D. R as number of regions. D can be:
            256: all 8-neighborhood encoding
            59: uniform pattern
            10: rotation invariance

    Returns:
        List of bytes. Each element is a bytes array corresponding to the hash of a data point in the batch.
    """
    assert lbp_code.ndim == 3
    assert lbp_code.dtype is torch.uint8
    batch_size = lbp_code.shape[0]
    lbp_flat = lbp_code.contiguous().view(batch_size, -1).cpu().numpy()
    return [lbp_flat[i].tobytes() for i in range(batch_size)]


def key_from_od(od: torch.Tensor, mask: torch.Tensor) -> List[Tuple[int, int, bytes, int]]:
    dh, dv = od_dhash(od, 8, 8)
    lbp_code = od_lbp8_hash(od, 64, 1, 1, True, 16, True)
    d_color = od_angle_hash64(od, mask)

    dh_hash = to_uint(dh)
    dv_hash = to_uint(dv)
    d_color_hash = to_uint(d_color)
    lbp_code_hash = lbp_code_to_bytes(lbp_code)
    return [(h, v, b, c) for h, v, b, c in zip(dh_hash, dv_hash, lbp_code_hash, d_color_hash)]