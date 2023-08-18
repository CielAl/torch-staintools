# Torch Stain Tools

## Description
* Stain Normalization (Reinhard, Macenko, and Vahadane) for pytorch. Input tensors (fit and transform) must be in shape of `NxCxHxW`, with value scaled to [0, 1] in format of torch.float32.
* Simulate the workflow in [StainTools library](https://github.com/Peter554/StainTools) but use the Iterative Shrinkage Thresholding Algorithm (ISTA), or optionally, the coordinate descent (CD) to solve the dictionary learning for stain matrix/concentration computation in Vahadane or Macenko (stain concentration only) algorithm. The implementation of ISTA and CD are derived from Cédric Walker's [torchvahadane](https://github.com/cwlkr/torchvahadane)
* No SPAMS requirement (which is a dependency in StainTools).

## Usecase
* For details, follow the example in demo.py
* Normalizers are wrapped as `torch.nn.Module`, working similarly to a standalone neural network. This means that for a workflow involving dataloader with multiprocessing, the normalizer
  (Note that CUDA has poor support in multiprocessing and therefore it may not be the best practice to perform GPU-accelerated on-the-fly stain transformation in pytorch's dataset/dataloader)
 
## Installation
`pip install `

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

