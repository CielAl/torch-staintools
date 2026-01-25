import numpy as np
import random
import torch
# import cv2


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def dummy_from_numpy(hwc: np.ndarray):
    return torch.from_numpy(hwc).permute(-1, 0, 1).unsqueeze(0)


def dummy_to_numpy(bchw: torch.Tensor):
    return bchw.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()


# def get_lab_np(img_ubyte: np.ndarray):
#     lab_result = cv2.cvtColor(img_ubyte, cv2.COLOR_BGR2LAB)
#     lab_result = lab_result.astype(np.float64)
#     lab_result[:, :, 0] *= 100.
#     lab_result[:, :, 0] /= 255.
#     lab_result[:, :, 1: 3] += 128
#     return lab_result


def psnr(true_img: torch.Tensor, pred_img: torch.Tensor, max_pixel: float = 1.0):
    mse = torch.mean((true_img - pred_img) ** 2)
    if mse == 0:
        # Prevent division by zero
        return torch.tensor(float('inf'))
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr_value
