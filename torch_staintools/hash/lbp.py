from typing import Optional, Tuple

import torch
from torch.nn import functional as F

from torch_staintools.constants import PARAM
from torch_staintools.functional.compile import lazy_compile

UNIFORM_LENGTH = 59
RIU_LENGTH = 10
LBP_WIDTH = 8
LBP_LENGTH = 2 ** LBP_WIDTH

def _build_lut_uniform() -> torch.Tensor:
    """
    Returns:
        torch.Tensor: int64. lookup table to map {0 ... 255} -> {0 ... 58}
    """
    lut = torch.empty(LBP_LENGTH, dtype=torch.int64)

    next_id = 0
    for code in range(LBP_LENGTH):
        # lsb-first 8bits
        bits = [(code >> i) & 1 for i in range(LBP_WIDTH)]

        # circular transitions
        transitions = 0
        for i in range(LBP_WIDTH):
            if bits[i] != bits[(i + 1) % LBP_WIDTH]:
                transitions += 1

        if transitions <= 2:
            lut[code] = next_id
            next_id += 1
        else:
            lut[code] = UNIFORM_LENGTH - 1

    assert next_id == UNIFORM_LENGTH - 1, f"{next_id} != {UNIFORM_LENGTH - 1}"
    return lut

def _build_lut_riu(uniform_lut59: torch.Tensor) -> torch.Tensor:

    assert uniform_lut59.shape == (256,)
    lut = uniform_lut59.to(torch.int64).cpu()

    u2_to_riu2 = torch.empty(59, dtype=torch.int64)
    u2_to_riu2.fill_(-1)

    for code in range(256):
        u2_id = int(lut[code].item())
        if u2_id == 58:
            continue

        pc = int(code).bit_count()
        u2_to_riu2[u2_id] = pc


    u2_to_riu2[58] = 9

    assert (u2_to_riu2 >= 0).all().item()
    assert (u2_to_riu2 <= 9).all().item()
    return u2_to_riu2


_UNIFORM_LUT = _build_lut_uniform()
_RUI_LUT = _build_lut_riu(_UNIFORM_LUT)


@lazy_compile(dynamic=True)
def _od_lbp8_code(
    od: torch.Tensor,
    thumb_size: int,
) -> torch.Tensor:
    """Core impl for 8-neighbor (radius=1) LBP codes.

    Args:
        od: Input batch image in optical density. BxCxHxW.
        thumb_size: size of the thumbnail.

    Returns:
        torch.Tensor: the code
    """
    # scalar proxy. for now use the sum (same as dhash). may change later?
    s = od.sum(dim=1, keepdim=True)

    s_t = F.interpolate(s, size=(thumb_size, thumb_size), mode="bilinear", align_corners=False)  # (B,1,thumb,thumb)

    # pre-pad for corner and border pixels
    p = F.pad(s_t, (1, 1, 1, 1), mode="replicate")

    # center
    c = p[:, :, 1:-1, 1:-1]  # (B,1,thumb,thumb)

    # 8 directions
    # B x 1 x thumb x thumb
    n0 = p[:, :, 0:-2, 0:-2] >= c  # top-left
    n1 = p[:, :, 0:-2, 1:-1] >= c  # top
    n2 = p[:, :, 0:-2, 2:  ] >= c  # top-right
    n3 = p[:, :, 1:-1, 2:  ] >= c  # right
    n4 = p[:, :, 2:  , 2:  ] >= c  # bottom-right
    n5 = p[:, :, 2:  , 1:-1] >= c  # bottom
    n6 = p[:, :, 2:  , 0:-2] >= c  # bottom-left
    n7 = p[:, :, 1:-1, 0:-2] >= c  # left

    # pack 8-bit pattern
    code = (
        (n0.to(torch.uint8) << 0) |
        (n1.to(torch.uint8) << 1) |
        (n2.to(torch.uint8) << 2) |
        (n3.to(torch.uint8) << 3) |
        (n4.to(torch.uint8) << 4) |
        (n5.to(torch.uint8) << 5) |
        (n6.to(torch.uint8) << 6) |
        (n7.to(torch.uint8) << 7)
    )

    return code


def _hist_helper(code_f: torch.Tensor, lut: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper function for hist_lbp8_256_to_59

    Args:
        code_f: flattened LBP codes. B x num_regions x 256.
        lut: Look-up table. 256-D
    Returns:
        torch.Tensor: histogram. B x num_regions x 59 (uint16).
    """
    bin_idx = lut[code_f]
    # ones = torch.ones_like(bin_idx, dtype=torch.int64)
    batch_size, num_regions, _ = bin_idx.shape
    hist59 = torch.zeros((batch_size, num_regions, 59), device=code_f.device, dtype=torch.int64)
    # hist59.scatter_add_(dim=-1, index=bin_idx, src=ones)

    return hist59, bin_idx


def hist_lbp8_256_to_59(
    code_f: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    """Uniform pattern compression. 256 -> 59

    Args:
        code_f: flattened LBP codes. B x num_regions x 256.
        lut: Look-up table. 256-D
    Returns:
        torch.Tensor: histogram. B x num_regions x 59 (uint16).
    """
    lut = lut.to(code_f.device)
    hist59, bin_idx = _hist_helper(code_f, lut)
    ones = torch.ones_like(bin_idx, dtype=torch.int64)
    hist59.scatter_add_(dim=-1, index=bin_idx, src=ones)

    return hist59.to(torch.uint16)


def u2_hist59_to_riu2_hist10(hist59: torch.Tensor,
                             lut_riu: torch.Tensor) -> torch.Tensor:
    """map the u2 (uniform pattern, 59) to riu (rotation-invariance, 10)

    Args:
        hist59:
        lut_riu:

    Returns:

    """
    assert hist59.shape[-1] == 59
    device = hist59.device
    B, R, _ = hist59.shape

    # (59,) int64 on device for indexing
    map59 = lut_riu.to(device=device, dtype=torch.int64)

    # index tensor for scatter_add: (B,R,59)
    idx = map59.view(1, 1, 59).expand(B, R, 59)

    out = torch.zeros((B, R, RIU_LENGTH), device=device, dtype=torch.int32)
    out.scatter_add_(dim=-1, index=idx, src=hist59.to(torch.int32))
    return out


def _od_lbp8_hist(code: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    B, _, code_h, code_w = code.shape
    region_h = code_h // grid_h
    region_w = code_w // grid_w
    num_region = grid_h * grid_w

    # Reshape into regions:
    # B x 1 x code_h x code_w
    # code_h -> grid_h x region_h
    # B x 1 x grid_h x region_h x grid_w x region_w
    # shuffle to
    # B x grid_h x grid_w x 1 x region_h x region_w
    code_r = code.reshape(B, 1, grid_h, region_h, grid_w, region_w).permute(0, 2, 4, 1, 3, 5)

    # flattened code_r
    # B x (grid_h * grid_w) x (1 * region_h * region_w)
    # B x num_regions x num_pixel_in_region
    code_f = code_r.reshape(B, num_region, -1).to(torch.int64)  # (B,R,P)

    # use all tissue
    # ones = torch.ones_like(code_f, dtype=torch.int64)  # (B,R,P)

    # # Histogram per region.
    # # B x num_region x num_pattern
    # hist = torch.zeros((B, num_region, 256), device=code.device, dtype=torch.int64)
    # # count
    # hist.scatter_add_(dim=-1, index=code_f, src=ones)
    hist = hist_lbp8_256_to_59(code_f, _UNIFORM_LUT.to(code_f.device))
    return hist.to(torch.uint16)


@lazy_compile(dynamic=True)
def _od_lbp_hist_quantize_levels(
    hist: torch.Tensor,
    levels: int,
    use_sqrt: bool,
) -> torch.Tensor:
    """Normalization counts and then quantize to uint8 in [0, lvl - 1]

    Args:
        hist: B x num_regions x D. (256 for LBP, or 59 for Uniform Patterns, 10 for RIU)
        levels: number of quantization levels.
        use_sqrt: apply sqrt after normalization.

    Returns:
        q: B x num_regions x D
    """


    h_float = hist.float()
    eps = torch.finfo(torch.float32).eps
    density = h_float.sum(dim=-1, keepdim=True).clamp_min(eps)
    freq = h_float / density

    if use_sqrt:
        freq = torch.sqrt(freq)
        # Re-normalize
        den2 = freq.sum(dim=-1, keepdim=True).clamp_min(eps)
        freq = freq / den2

    # Quantize to [0..levels-1]
    # +0.5 for rounding to nearest.
    q = torch.clamp((freq * (levels - 1) + 0.5).to(torch.int32), 0, levels - 1).to(torch.uint8)
    return q


@torch.inference_mode()
def od_lbp8_hash(
    od: torch.Tensor,
    thumb_size: int = PARAM.HASH_LBP_THUMB,
    grid_h: int = PARAM.HASH_LBP_GRID,
    grid_w: int = PARAM.HASH_LBP_GRID,
    use_riu: bool = PARAM.HASH_LBP_RIU,
    levels: Optional[int] = PARAM.HASH_LBP_QUANTIZE,
    use_sqrt: bool = PARAM.HASH_LBP_SQRT,

) -> torch.Tensor:
    """

    Args:
        od: Input batch image in optical density. BxCxHxW.
        thumb_size: size of the code thumbnail.
        grid_h: grid height. how many regions along H. (region height = thumb // grid_height)
        grid_w: grid width. how many regions along W. (region width = thumb // grid_width)
        use_riu: whether further cast to RIU (10bytes)
        levels: number of quantization levels. If None, them bypass.
        use_sqrt: whether to further compress the frequency by sqrt.

    Returns:
        torch.Tensor: the histogram. B x (grid_h*grid_w) * D
            grid_h*grid_w as how many grids;
            D = 256 -> original LBP (8bits)
            D = 59 -> Uniform Pattern
    """
    # od = od.to(torch.float32)
    code = _od_lbp8_code(od, thumb_size=thumb_size)
    _, _, code_h, code_w = code.shape

    assert code_h % grid_h ==0 and code_w % grid_w ==0,\
        f"thumb={thumb_size}, grid_h={grid_h}, grid_w={grid_w}"
    hist = _od_lbp8_hist(code, grid_h, grid_w)
    if use_riu:
        hist = u2_hist59_to_riu2_hist10(hist, _RUI_LUT.to(hist.device))
    if levels is None:
        return hist
    return _od_lbp_hist_quantize_levels(hist, levels, use_sqrt)