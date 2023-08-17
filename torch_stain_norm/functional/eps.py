import torch

_eps_val = torch.finfo(torch.float32).eps


def get_eps(img: torch.Tensor = None):
    eps = torch.tensor(_eps_val)
    if img is not None:
        eps = eps.to(img.device)
    return eps
