"""
Add small eps terms to avoid NaN in certain arithmetics (e.g., division and exp)
"""
import torch

_eps_val = torch.finfo(torch.float32).eps


def get_eps(img: torch.Tensor = None) -> torch.Tensor:
    """Get the eps based on the device input tensor.

     Precision is defined by the global variable `_eps_val`.

    Args:
        img: If specified, then the output will be moved to the same device of the img.

    Returns:
        output eps as a tensor. It will be moved to the device of img if img is specified.
    """
    eps = torch.tensor(_eps_val)
    if img is not None:
        eps = eps.to(img.device)
    return eps
