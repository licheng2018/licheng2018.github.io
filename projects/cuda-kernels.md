# CUDA Kernel Optimization and PyTorch Extensions

Designed and optimized high-performance CUDA kernels for ML workloads with hardware-aware memory tuning.

## Operators

- Matmul
- LayerNorm forward and backward
- Numerically stable Softmax

## Work

- Implemented kernels in CUDA C++ and integrated them as PyTorch C++/CUDA extensions.
- Applied block tiling, shared memory reuse, loop unrolling, and register-level optimization.
- Tuned block size and register usage to improve occupancy and reduce warp stalls.
- Added full autograd support for PyTorch integration.

## Outcome

Achieved up to 2-3x speedup over PyTorch eager baselines on selected memory-bound operators such as LayerNorm and Softmax, and around 200x speedup over CPU baselines in controlled benchmarks.

[Back to Home](../index.md)
