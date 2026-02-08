import torch

from torch_staintools.constants import PARAM
from torch_staintools.hash.hash_util import _avgpool_resize_area


def _od_dhash_hv_uint64(
    od: torch.Tensor,
    h: int,
    w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Core for dhash.

    Return the horizontal gradient (d-hash) and the corresponding vertical counterpart.

    Args:
        od: Batchified OD tensor: BxCxHxW.
        h: grid height of dhash. hxw = num bits
        w: grid width of dhash. hxw = num bits

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: dhash and the vertical counterpart.
    """
    B = od.shape[0]
    nbits = h * w

    od_int = od.sum(dim=1, keepdim=True)  # (B,1,H,W)

    small_h = _avgpool_resize_area(od_int, h=h, w=w + 1)
    h_bits = (small_h[:, :, :, 1:] > small_h[:, :, :, :-1]).reshape(B, nbits)

    small_v = _avgpool_resize_area(od_int, h=h + 1, w=w)
    v_bits = (small_v[:, :, 1:, :] > small_v[:, :, :-1, :]).reshape(B, nbits)

    # LSB order
    weights = (1 << torch.arange(nbits, device=od.device, dtype=torch.int64))  # .to(torch.uint64)
    h_hash = (h_bits.to(torch.int64) * weights).sum(dim=1)
    v_hash = (v_bits.to(torch.int64) * weights).sum(dim=1)

    return h_hash.to(torch.uint64), v_hash.to(torch.uint64)


@torch.inference_mode()
def od_dhash(
    od: torch.Tensor,
    out_h: int = PARAM.HASH_DHASH_SIZE,
    out_w: int = PARAM.HASH_DHASH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:

    assert od.ndim == 4, f"od invalid num of dim with shape: {od.shape}"

    nbits = out_h * out_w
    assert nbits <= 64, f"{nbits} exceeds 64 bits (h x w > 64 is not supported)"

    return _od_dhash_hv_uint64(od, out_h, out_w)