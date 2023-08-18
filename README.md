# Torch StainTools

## Description
* Stain Normalization (Reinhard, Macenko, and Vahadane) for pytorch. Input tensors (fit and transform) must be in shape of `NxCxHxW`, with value scaled to [0, 1] in format of torch.float32.
* Simulate the workflow in [StainTools library](https://github.com/Peter554/StainTools) but use the Iterative Shrinkage Thresholding Algorithm (ISTA), or optionally, the coordinate descent (CD) to solve the dictionary learning for stain matrix/concentration computation in Vahadane or Macenko (stain concentration only) algorithm. The implementation of ISTA and CD are derived from Cédric Walker's [torchvahadane](https://github.com/cwlkr/torchvahadane)
* No SPAMS requirement (which is a dependency in StainTools).

![Screenshot](showcases/sample_out.png)


## Usecase
* For details, follow the example in demo.py
* Normalizers are wrapped as `torch.nn.Module`, working similarly to a standalone neural network. This means that for a workflow involving dataloader with multiprocessing, the normalizer
  (Note that CUDA has poor support in multiprocessing and therefore it may not be the best practice to perform GPU-accelerated on-the-fly stain transformation in pytorch's dataset/dataloader)


```
import cv2
import torch
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import convert_image_dtype
from torch_staintools.normalizer.factory import NormalizerBuilder
import os
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# cpu or gpu
device = torch.device("cuda:0")

root_dir = '.'
target = cv2.imread(os.path.join(root_dir, 'test_images/TCGA-33-4547-01Z-00-DX7.'
                                           '91be6f90-d9ab-4345-a3bd-91805d9761b9_8270_5932_0.png'))
# shape: HWC
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
norm = cv2.imread(os.path.join(root_dir, 'test_images/TCGA-95-8494-01Z-00-DX1.'
                                         '716299EF-71BB-4095-8F4D-F0C2252CE594_5932_5708_0.png'))
# shape: HWC
norm = cv2.cvtColor(norm, cv2.COLOR_BGR2RGB)


# shape: BCHW - scaled to [0, 1] torch.float32
target_tensor = ToTensor()(target).unsqueeze(0).to(device)

# shape: BCHW - scaled to [0, 1] torch.float32
norm_tensor = ToTensor()(norm).unsqueeze(0).to(device)

# fit
normalizer_vahadane = NormalizerBuilder.build('vahadane', reconst_method='ista')
normalizer_vahadane = normalizer_vahadane.to(device)
normalizer_vahadane.fit(target_tensor)
# transform
# BCHW - scaled to [0, 1] torch.float32
output = normalizer_vahadane(norm_tensor, algorithm='ista', constrained=True, verbose=False)
```

## Installation
`pip install git+https://github.com/CielAl/torch-staintools.git`

## Benchmark
* Use the sample images under ./test_images (size `2500x2500x3`). Mean was computed from 7 runs (1 loop per run) using
timeit. Comparison between torch_stain_tools in CPU/GPU mode, as well as that of the StainTools Implementation.

### Transformation

| Method   | CPU[s] | GPU[s] | StainTool[s] |
|:---------|:-------|:-------|:-------------| 
| Vahadane | 119    | 7.5    | 20.9         |  
| Macenko  | 5.57   | 0.479  | 20.7         |
| Reinhard | 0.840  |0.024   | 0.414        |  

### Fitting
| Method   | CPU[s] | GPU[s] | StainTool[s] |
|:---------|:-------|:-------|:-------------| 
| Vahadane | 132    | 8.40   | 19.1         |  
| Macenko  | 6.99   | 0.064  | 20.0         |
| Reinhard | 0.422  | 0.011  | 0.076        |  


## Acknowledgments
* Some codes are derived from [torchvahadane](https://github.com/cwlkr/torchvahadane), [torchstain](https://github.com/EIDOSLAB/torchstain), and [StainTools](https://github.com/Peter554/StainTools)

