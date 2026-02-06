# Torch StainTools for Stain Normalization and Augmentation of Histopathological Images

[![Unit Testing](https://github.com/CielAl/torch-staintools/actions/workflows/unittest.yml/badge.svg?branch=main)](https://github.com/CielAl/torch-staintools/actions/workflows/unittest.yml)
[![DOI](https://zenodo.org/badge/679661478.svg)](https://zenodo.org/doi/10.5281/zenodo.10453806)

## Installation

* From Repository:

`pip install git+https://github.com/CielAl/torch-staintools.git`

* From PyPI:

`pip install torch-staintools`

## What's New
* **Version 1.0.6**: full vectorization support and dynamic shape tracking from **Dynamo**.
* Alternative linear concentration solvers: ```'qr'``` (QR Decomposition) and ```'pinv'``` (Moore-Penrose inverse)
## Documentation
Detail documentation regarding the code base can be found in the [GitPages](https://cielal.github.io/torch-staintools/).

## Citation
If this toolkit helps you in your publication, please feel free to cite with the following bibtex entry:
```bibtex
@software{zhou_2024_10496083,
  author       = {Zhou, Yufei},
  title        = {CielAl/torch-staintools: V1.0.4 Release},
  month        = jan,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.4},
  doi          = {10.5281/zenodo.10496083},
  url          = {https://doi.org/10.5281/zenodo.10496083}
}
```

## Description
* **Dynamo** (```torch.compile```)-powered acceleration.
* Stain Normalization (Reinhard, Macenko, and Vahadane) for pytorch with efficient vectorization.
* Stain Augmentation using Macenko and Vahadane as stain extraction.
* Fast normalization/augmentation on GPU with stain matrices caching.
* No SPAMS requirement (which is a dependency in StainTools).

<br />

#### Sample Output of Torch-StainTools Normalization
![Screenshot](https://raw.githubusercontent.com/CielAl/torch-staintools/main/showcases/sample_out.png)

#### Sample Output of StainTools
![Screenshot](https://raw.githubusercontent.com/CielAl/torch-staintools/main/showcases/sample_out_staintools.png)

#### Sample Output of Torch-StainTools Augmentation (Repeat 3 times)
![Screenshot](https://raw.githubusercontent.com/CielAl/torch-staintools/main/showcases/sample_out_augmentation.png)

#### Sample Output of StainTools Augmentation (Repeat 3 times)
![Screenshot](https://raw.githubusercontent.com/CielAl/torch-staintools/main/showcases/sample_out_augmentation_staintools.png)

## Benchmark (No Stain Matrices Caching)
* Use the sample images under ./test_images (size `2500x2500x3`). Mean was computed from 7 runs (1 loop per run) using
timeit. Comparison between torch_stain_tools in CPU/GPU mode, as well as that of the StainTools Implementation.
* For consistency, use ISTA to compute the concentration.
* v1.0.5+ speedup, in part from ```torch.compile```.
### Transformation 

*```torch.compile``` enabled.

| Method   | CPU[s] | GPU[s]       | StainTool[s] |
|:---------|:-------|:-------------|:-------------| 
| Vahadane | 119.00 | ~~7.5~~ 4.60 | 20.90        |  
| Macenko  | 5.57   | 0.48         | 20.70        |
| Reinhard | 0.84   | 0.02         | 0.41         |  

### Fitting
```torch.compile``` enabled.

| Method   | CPU[s] | GPU[s]        | StainTool[s] |
|:---------|:-------|:--------------|:-------------| 
| Vahadane | 132.00 | ~~8.40~~ 5.20 | 19.10        |  
| Macenko  | 6.99   | 0.06          | 20.00        |
| Reinhard | 0.42   | 0.01          | 0.08         |  

### Batchified Concentration Computation
* Split the sample images under ./test_images (size `2500x2500x3`) into 81 non-overlapping `256x256x3` tiles as a batch.
* For the StainTools baseline, a for-loop is implemented to get the individual concentration of each of the numpy array of the 81 tiles.
* ```torch.compile``` enabled. ```cuSolver``` backend is applied.
* * v1.0.6: vectorization support.
* 
| Method                                 | CPU[s] | GPU[s]         | 
|:---------------------------------------|:-------|:---------------| 
| FISTA (`concentration_solver='fista'`) | 1.47   | ~~0.24~~ 0.093 |  
| ISTA (`concentration_solver='ista'`)   | 3.12   | ~~0.31~~ 0.088 |  
| CD   (`concentration_solver='cd'`)     | 29.30s | ~~4.87~~ 0.158 | 
| LS   (`concentration_solver='ls'`)     | 0.22   | **0.097**      |
| QR   (`concentration_solver='qr'`)     | 0.08   | **0.007**      |
| PINV   (`concentration_solver='pinv'`) | 0.08   | **0.004**      |
| StainTools (SPAMS)                     | 16.60  | N/A            |


## Use Cases and Tips
* For details, follow the example in demo.py
* Normalizers are wrapped as `torch.nn.Module`, working similarly to a standalone neural network. This means that for a workflow involving dataloader with multiprocessing, the normalizer
  (Note that CUDA has poor support in multiprocessing, and therefore it may not be the best practice to perform GPU-accelerated on-the-fly stain transformation in pytorch's dataset/dataloader)

* `concentration_solver='ls'` (i.e., `torch.linalg.lstsq`) can be efficient for batches of many smaller input (e.g., `256x256`) in terms of width and height. However, it may fail on GPU for a single larger input image (width and height). This happens with the default ```cusolver``` backend. Try using ```magma``` instead:
   ```python
   torch.backends.cuda.preferred_linalg_library('magma')
   ```

```python
"""Demo prerequisite:
    cv2 (read and process images)
"""
import cv2
import torch
from torchvision.transforms import ToTensor
from torch_staintools.normalizer import NormalizerBuilder
from torch_staintools.augmentor import AugmentorBuilder
from torch_staintools.constants import CONFIG
import os

# Globally fix the random state.
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
                                              # whether to cache the stain matrix.
                                              # must pair the input with an identifier (e.g. filename)
                                              # otherwise cache will be ignored.
                                              use_cache=True
                                              )
normalizer_vahadane = normalizer_vahadane.to(device)
normalizer_vahadane.fit(target_tensor)

# ###### Augmentation
# augment by: alpha * concentration + beta, while alpha is uniformly randomly sampled from (1 - sigma_alpha, 1 + sigma_alpha),
# and beta is uniformly randomly sampled from (-sigma_beta, sigma_beta).
augmentor = AugmentorBuilder.build('vahadane',
                                   # custom generator may cause graph break if torch.compile is enabled.
                                   # (by setting CONFIG.ENABLE_COMPILE = True)
                                   # therefore the random state is globally controlled outside.
                                   # use rng=None here instead.
                                   rng=None,
                                   # the luminosity threshold to find the tissue region to augment
                                   # if set to None means all pixels are treated as tissue
                                   luminosity_threshold=0.8,
                                   # herein we use 'ista' to compute the concentration
                                   concentration_solver='fista',
                                   sigma_alpha=0.2,
                                   sigma_beta=0.2, 
                                   num_stains=2,
                                   # for two stains (herein, H&E), augment both H and E.
                                   target_stain_idx=(0, 1),
                                   # this allows to cache the stain matrix if it's too time-consuming to recompute.
                                   # e.g., if using Vahadane algorithm
                                   use_cache=True,
                                   # size limit of cache. -1 means no limit (stain matrix is often small in size, e.g., 2 x 3)
                                   cache_size_limit=-1,
                                   # if specified, the augmentor will load the cached stain matrices from file system.
                                   load_path=None,
                                   )
# move augmentor to the corresponding device
augmentor = augmentor.to(device)

num_augment = 5
# multiple copies of different random augmentation of the same tile may be generated
for _ in range(num_augment):
  # B x C x H x W
  # use a list of Hashable key (e.g., str) to map the batch input to its corresponding stain matrix in cache.
  # this key should be unique, e.g., using the filename of the input tile.
  # leave it as None if no caching is intended, even if use_cache is enabled.
  # note since the inputs are all batchified, the cache_key are in form of a list, with each element in the 
  # list corresponding to a data point in the batch.
  aug_out = augmentor(norm_tensor, cache_keys=['some unique key'])
  # do anything to the augmentation output

# dump the cache of stain matrices for future usage
augmentor.dump_cache('./cache.pickle')

# fast batch operation
tile_size = 512
tiles: torch.Tensor = norm_tensor.unfold(2, tile_size, tile_size).unfold(3, tile_size, tile_size).reshape(1, 3, -1, tile_size, tile_size).squeeze(0).permute(1, 0, 2, 3).contiguous()
print(tiles.shape)
# use macenko normalization as example
# if using cusolver, 'ls' (least square) will fail on single large images.
# try magma backend if 'ls' is still preferred as the concentration estimator (see below)
torch.backends.cuda.preferred_linalg_library('magma')
normalizer_macenko = NormalizerBuilder.build('macenko', use_cache=True,
                                             # use least square solver, along with cache, to perform
                                             # normalization on-the-fly
                                             concentration_solver='qr')
normalizer_macenko = normalizer_macenko.to(device)
normalizer_macenko.fit(target_tensor)
normalizer_macenko(tiles)

```
## Stain Matrix Caching
As elaborated in the below in the running time benchmark of fitting, computation of stain matrix could be time-consuming.
Therefore, for both `Augmentor` and `Normalizer`, an in-memory (device-specified) cache is implemented to store the previously computed stain matrices (typically with size 2 x 3 in H&E/RGB cases).
To enable the feature, the `use_cache` must be enabled, should you use the factory builders to instantiate the `Normalizer` or `Augmentor`.
Upon the normalization/augmentation procedure, a unique cache_key corresponding to the image input must be defined (e.g., file name).
Since both `Normalizer` and `Augmentor` are designed as `torch.nn.Module` to accept batch inputs (tensors of shape B x C x H x W), a list of cache_keys must be given along with the batch image
inputs during the forward passing:
```
normalizer_vahadane(input_batch, cache_keys=list_of_keys_corresponding_to_input_batch)
augmentor(input_batch, cache_keys=list_of_keys_corresponding_to_input_batch)

```
The next time `Normalizer` or `Augmentor` process the images, the corresponding stain matrices will be queried and fetched from cache if they are stored already, rather than recomputing from scratch.


## Acknowledgments
* Some codes are inspired from [torchvahadane](https://github.com/cwlkr/torchvahadane), [torchstain](https://github.com/EIDOSLAB/torchstain), and [StainTools](https://github.com/Peter554/StainTools)
* Sample images in the demo and ReadMe.md are selected from [The Cancer Genome Atlas Program(TCGA)](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) dataset.
