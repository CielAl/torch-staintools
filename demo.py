"""Demo prerequisite:
    tqdm (progress bar)
    staintools (for comparison)
    cv2 (read and process images)
"""
import math
import cv2
import torch
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import make_grid
from torch_staintools.normalizer import NormalizerBuilder
from torch_staintools.augmentor import AugmentorBuilder
from torch_staintools.constants import CONFIG
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import os

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# cpu or gpu
device = torch.device("cuda:0")  # torch.device("cpu")

root_dir = '.'
target = cv2.imread(os.path.join(root_dir, 'test_images/TCGA-33-4547-01Z-00-DX7.'
                                           '91be6f90-d9ab-4345-a3bd-91805d9761b9_8270_5932_0.png'))
# shape: HWC (Height Width Channel)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
norm = cv2.imread(os.path.join(root_dir, 'test_images/TCGA-95-8494-01Z-00-DX1.'
                                         '716299EF-71BB-4095-8F4D-F0C2252CE594_5932_5708_0.png'))
# shape: HWC
norm = cv2.cvtColor(norm, cv2.COLOR_BGR2RGB)


# shape: BCHW (Batch Channel Height Width) - scaled to [0, 1] torch.float32
target_tensor = ToTensor()(target).unsqueeze(0).to(device)

# shape: BCHW - scaled to [0, 1] torch.float32
norm_tensor = ToTensor()(norm).unsqueeze(0).to(device)


# test with multiple smaller regions from the sample image
tile_size = 256
# split the sample images into a batch of patches.
tiles: torch.Tensor = norm_tensor.unfold(2, tile_size, tile_size)\
    .unfold(3, tile_size, tile_size).reshape(1, 3, -1, tile_size, tile_size).squeeze(0).permute(1, 0, 2, 3).contiguous()

# ########## show inputs
plt.imshow(norm)
plt.title("Source - Full size")
plt.show()
plt.imshow(target)
plt.title("Template")
plt.show()


# helper function to convert tensor back to numpy arrays for visualization purposes.
def postprocess(image_tensor):
    return convert_image_dtype(image_tensor, torch.uint8)\
    .squeeze().detach().cpu().permute(1, 2, 0).numpy()

def plot(tiles_to_plot: torch.Tensor, title: str = ""):
    nrow = max(int(math.sqrt(tiles_to_plot.shape[0])), 1)
    grid_np = postprocess(make_grid(tiles_to_plot,
                                    padding = 32,
                                    nrow=nrow, pad_value=1))
    plt.figure(dpi=300, figsize=(8, 8))
    plt.imshow(grid_np)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# We enable the torch.compile (note this is True by default)
CONFIG.ENABLE_COMPILE = True
# ######### Vahadane
normalizer_vahadane = NormalizerBuilder.build('vahadane',
                                              # use fista (fast iterative shrinkage-thresholding algorithm)
                                              # for dictionary learning to
                                              # estimate the stain matrix (sparse constraints)
                                              # alternative: 'cd' (coordinate descent);
                                              # 'ista' (iterative shrinkage-thresholding algorithm)
                                              sparse_stain_solver='fista',
                                              concentration_solver='fista',
                                              # use a smaller max iteration limit for code-step and
                                              # concentration computation since it typically converges early
                                              # but different maxiter will trigger recompilation
                                              maxiter=30,
                                              # whether to cache the stain matrix.
                                              # must pair the input with an identifier (e.g. filename)
                                              # otherwise cache will be ignored.
                                              use_cache=True,
                                              # luminosity threshold for tissue masking.
                                              # bypass by set to None
                                              luminosity_threshold=0.8,
                                              )
normalizer_vahadane = normalizer_vahadane.to(device)
# fit with a single reference image --> will always trigger a recompilation for batch = 1
normalizer_vahadane.fit(target_tensor)

# fully vectorized for batchified inputs.
vahadane_out = normalizer_vahadane(tiles)

# plot the output
plot(vahadane_out, title="Vahadane")

# %timeit normalizer_vahadane(norm_tensor, positive_dict=True)

#   #################### Macenko
# if using cusolver, 'ls' (least square) will fail on single large images.
# try magma backend if 'ls' is still preferred as the concentration estimator (see below)
# torch.backends.cuda.preferred_linalg_library('magma')
# supported concentration solvers: 'qr', 'pinv', 'fista', 'ista', 'ls'
normalizer_macenko = NormalizerBuilder.build('macenko',
                                             use_cache=True,
                                             maxiter=30,
                                             concentration_solver='qr')  # 'ls'
normalizer_macenko = normalizer_macenko.to(device)
normalizer_macenko.fit(target_tensor)

# can pass a custom mask here to override the internal tissue masking
macenko_out = normalizer_macenko(tiles)
plot(macenko_out, title="Macenko")
# # %timeit normalizer_macenko(norm_tensor, algorithm='ista', positive_dict=True,)

# ###################### Reinhard

normalizer_reinhard = NormalizerBuilder.build('reinhard')
normalizer_reinhard = normalizer_reinhard.to(device)
normalizer_reinhard.fit(target_tensor)
reinhard_out = normalizer_reinhard(tiles)
plot(reinhard_out, title="Reinhard")

# %timeit normalizer_reinhard(norm_tensor)

# Augmentation

augmentor = AugmentorBuilder.build('vahadane',
                                   sparse_stain_solver='fista',
                                   concentration_solver='fista',
                                   num_stains=2,
                                   rng=None,  # None if globally managing the seeds
                                   sigma_alpha=0.2,
                                   sigma_beta=0.2,
                                   # for two stains (herein, H&E), augment both H and E.
                                   target_stain_idx=(0, 1),
                                   use_cache=True,
                                   )
# move augmentor to the device
augmentor.to(device)
aug_out = augmentor(tiles)
plot(aug_out, title="Augmentation")

# ##################### StainTool Comparison #####################
# ########## Staintools Vahadane
from staintools.stain_normalizer import StainNormalizer
st_vahadane = StainNormalizer(method='vahadane')
st_vahadane.fit(target)
tiles_np = tiles.permute(0, 2, 3, 1).detach().cpu().contiguous().numpy()

for idx, tile_single in enumerate(tqdm(tiles_np)):
    tile_single: np.ndarray
    tile_single: np.ndarray = (tile_single * 255).astype(np.uint8)
    test_out = st_vahadane.transform(tile_single)
    plt.imshow(test_out)
    plt.title(f"Vahadane StainTools: {idx}")
    plt.show()


# ########## Staintools Vahadane
from staintools.stain_normalizer import StainNormalizer
st_macenko = StainNormalizer(method='macenko')
st_macenko.fit(target)
tiles_np: np.ndarray = tiles.permute(0, 2, 3, 1).detach().cpu().contiguous().numpy()
# timeit st_macenko.transform(norm)
for idx, tile_single in enumerate(tqdm(tiles_np)):
    tile_single: np.ndarray
    tile_single: np.ndarray = (tile_single * 255).astype(np.uint8)
    test_out = st_macenko.transform(tile_single)
    plt.imshow(test_out)
    plt.title(f"Vahadane StainTools: {idx}")
    plt.show()


#  ########## Staintools Reinhard
from staintools.reinhard_color_normalizer import ReinhardColorNormalizer
st_reinhard = ReinhardColorNormalizer()
st_reinhard.fit(target)
tiles_np = tiles.permute(0, 2, 3, 1).detach().cpu().contiguous().numpy()
# %timeit st_reinhard.transform(norm)
for idx, tile_single in enumerate(tqdm(tiles_np)):
    tile_single: np.ndarray
    tile_single: np.ndarray = (tile_single * 255).astype(np.uint8)
    test_out = st_reinhard.transform(tile_single)
    plt.imshow(test_out)
    plt.title(f"Reinhard ST: {idx}")
    plt.show()

# ########## sample generation
images = [norm, target,
          postprocess(normalizer_vahadane(norm_tensor)),
          postprocess(normalizer_macenko(norm_tensor)),
          postprocess(normalizer_reinhard(norm_tensor))
          ]
titles = ["Source", "Template", "Vahadane", "Macenko", "Reinhard"]
assert len(images) == len(titles)
fig, axs = plt.subplots(1, len(images), figsize=(15, 4), dpi=300)
for i, ax in enumerate(axs):
    ax.imshow(images[i])
    ax.set_title(titles[i])
    ax.axis('off')
plt.savefig(os.path.join('.', 'showcases', 'sample_out.png'), bbox_inches='tight')
plt.show()


# stain tool comparison

images = [norm, target,
          st_vahadane.transform(norm),
          st_macenko.transform(norm),
          st_reinhard.transform(norm),
          ]
titles = ["Source", "Template", "Vahadane - StainTools", "Macenko - StainTools", "Reinhard - StainTools"]
assert len(images) == len(titles)
fig, axs = plt.subplots(1, len(images), figsize=(15, 4), dpi=300)
for i, ax in enumerate(axs):
    ax.imshow(images[i])
    ax.set_title(titles[i])
    ax.axis('off')
plt.savefig(os.path.join('.', 'showcases', 'sample_out_staintools.png'), bbox_inches='tight')
plt.show()

algorithms = ['Vahadane', 'Macenko']
num_repeat = 3

# # sample aug output
fig, axs = plt.subplots(2, num_repeat + 1, figsize=(15, 8), dpi=300)
for i, ax_alg in enumerate(axs):
    alg = algorithms[i].lower()
    # noinspection PyTypeChecker
    augmentor = AugmentorBuilder.build(alg,
                                       concentration_solver='ista',
                                       sigma_alpha=0.5,
                                       sigma_beta=0.5,
                                       luminosity_threshold=0.8,
                                       rng=314159, use_cache=True).to(device)
    ax_alg[0].imshow(norm)
    ax_alg[0].set_title("Augmentation Original")
    ax_alg[0].axis('off')
    for j in range(1, len(ax_alg)):
        aug_out = augmentor(norm_tensor, cache_keys=[0])
        ax_alg[j].imshow(postprocess(aug_out))
        ax_alg[j].set_title(f"{alg} :{j}")
        ax_alg[j].axis('off')
plt.savefig(os.path.join('.', 'showcases', 'sample_out_augmentation.png'), bbox_inches='tight')
plt.show()


# #### sample aug output
np.random.seed(314159)
random.seed(314159)
from staintools import StainAugmentor
from staintools.preprocessing.luminosity_standardizer import LuminosityStandardizer
algorithms = ['Vahadane', 'Macenko']
fig, axs = plt.subplots(2, num_repeat + 1, figsize=(15, 8), dpi=300)
for i, ax_alg in enumerate(axs):
    alg = algorithms[i].lower()
    augmentor = StainAugmentor(method=alg, sigma1=0.5, sigma2=0.5, augment_background=False)
    standardized_norm = LuminosityStandardizer.standardize(norm)
    augmentor.fit(standardized_norm)
    ax_alg[0].imshow(standardized_norm)
    ax_alg[0].set_title("Augmentation Original")
    ax_alg[0].axis('off')
    for j in range(1, len(ax_alg)):
        aug_out = augmentor.pop().astype(np.uint8)
        ax_alg[j].imshow(aug_out)
        ax_alg[j].set_title(f"{alg} - StainTools: {j}")
        ax_alg[j].axis('off')
plt.savefig(os.path.join('.', 'showcases', 'sample_out_augmentation_staintools.png'), bbox_inches='tight')
plt.show()


