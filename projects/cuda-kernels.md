# CUDA Kernel Development and Transformer Operator Optimization

This project implements and analyzes CUDA kernels for core ML and Transformer operators, with a focus on correctness, memory access patterns, tiling, warp-level reductions, shared memory reuse, register-level optimization, profiling, and PyTorch C++/CUDA extension integration.

Source repo: [cuda_projects](https://github.com/licheng2018/cuda_projects/tree/main)

## Contents

| Area | Jump to sections |
|---|---|
| Overview | [Project Overview](#project-overview) · [Visual Interview Map](#visual-interview-map) · [Project Scope](#project-scope) · [Operator Development Goals](#operator-development-goals) · [Repository Structure](#repository-structure) |
| CUDA fundamentals | [Vector Addition](#vector-addition-cuda-programming-foundation) · [Memory Coalescing](#cuda-memory-coalescing-strided-access-benchmark) · [Shared Memory and Bank Conflict](#cuda-shared-memory-and-bank-conflict) · [Warp-Level Reduction](#cuda-warp-level-reduction) · [Loop Unrolling and Register Optimization](#loop-unrolling-and-register-optimization) |
| ML operators | [Matrix Multiplication](#matrix-multiplication) · [GEMM Optimization Case Study](#gemm-optimization-case-study) · [Softmax](#softmax) · [LayerNorm](#layernorm) · [PyTorch Extensions](#pytorch-ccuda-extensions) · [Naive Attention](#naive-attention-and-io-bound-analysis) |
| Analysis | [Profiling Focus](#profiling-focus) · [Main Takeaways](#main-takeaways) · [Experiment Result Analysis](#experiment-result-analysis) |

## Project Overview

Modern deep learning workloads are often limited not only by arithmetic throughput, but also by memory traffic, synchronization overhead, kernel launch overhead, and inefficient mapping of computation onto GPU hardware.

The goal of this project is to study the complete lifecycle of GPU operator development, starting from fundamental CUDA programming and gradually progressing toward production-style implementations of commonly used Transformer operators.

Instead of focusing only on writing functionally correct kernels, this project emphasizes three equally important aspects:

- **Correctness:** ensuring that custom CUDA kernels produce results consistent with CPU or PyTorch reference implementations.
- **Performance:** improving memory access, parallel reduction, data reuse, and kernel organization.
- **Integration:** exposing CUDA kernels as reusable PyTorch C++/CUDA extensions.

The project was designed as an end-to-end learning and engineering exercise for GPU kernel development, machine learning systems, and accelerator-oriented software optimization.

### Visual Interview Map

These diagrams summarize the main CUDA concepts and ML operator case studies in one place, so the project can be explained quickly before drilling into implementation details.

**CUDA execution model and grid-stride loops**

![CUDA vector addition thread mapping and grid-stride loop](../assets/projects/cuda-kernels/vector-add-thread-mapping-grid-stride.png)

**Global memory coalescing**

![CUDA memory coalescing warp access pattern and memory transactions](../assets/projects/cuda-kernels/memory-coalescing-warp-access-pattern.png)

**Shared memory and bank conflicts**

![CUDA shared memory bank conflicts, strided access, and padding](../assets/projects/cuda-kernels/shared-memory-bank-conflict-overview.png)

![CUDA shared memory bank conflicts clean benchmark and padding explanation](../assets/projects/cuda-kernels/shared-memory-bank-conflict-clean-benchmark.png)

**Warp-level reduction**

![CUDA warp-level reduction with shuffle and block reduction](../assets/projects/cuda-kernels/warp-level-reduction-overview.png)

**Loop unrolling and register optimization**

![CUDA loop unrolling and register optimization mechanism, trade-off, and measured results](../assets/projects/cuda-kernels/loop-unrolling-register-optimization.png)

**Matrix multiplication optimization path**

![CUDA matrix multiplication optimization overall pipeline](../assets/projects/cuda-kernels/matmul-optimization-overall-pipeline.png)

**GEMM tile-size trade-off**

![CUDA MatMul tile size trade-off between 16x16 and 32x32](../assets/projects/cuda-kernels/matmul-tile16-vs-tile32-tradeoff.png)

**Tiled MatMul dataflow**

![CUDA shared-memory tiled MatMul dataflow](../assets/projects/cuda-kernels/matmul-tiled-dataflow.png)

**Naive vs tiled MatMul memory traffic**

![CUDA MatMul naive versus tiled memory traffic and data reuse](../assets/projects/cuda-kernels/matmul-naive-vs-tiled-memory-traffic.png)

**Nsight Compute profiling workflow**

![CUDA Nsight Compute profiling workflow from runtime symptom to root cause](../assets/projects/cuda-kernels/nsight-compute-profiling-workflow.png)

**Nsight root-cause analysis for GEMM**

![CUDA MatMul Nsight root-cause analysis for Tile16, Tile32, and Tile32 with padding](../assets/projects/cuda-kernels/nsight-matmul-root-cause-analysis.png)

**Stable Softmax dataflow**

![CUDA stable Softmax and reduction dataflow](../assets/projects/cuda-kernels/softmax-stable-reduction-dataflow.png)

**Softmax mapping strategies**

![CUDA Softmax row mapping strategy comparison across warp-per-row, block-per-row, and multi-warp-per-row](../assets/projects/cuda-kernels/softmax-row-mapping-strategy-comparison.png)

**LayerNorm forward**

![CUDA LayerNorm forward reduction and normalize dataflow](../assets/projects/cuda-kernels/layernorm-forward-reduction-normalize-dataflow.png)

**LayerNorm backward**

![CUDA LayerNorm backward gradient propagation and reduction paths](../assets/projects/cuda-kernels/layernorm-backward-gradient-reduction-paths.png)

![CUDA LayerNorm backward AtomicAdd versus two-pass reduction](../assets/projects/cuda-kernels/layernorm-backward-atomic-vs-two-pass-reduction.png)

## Project Scope

The project spans both low-level GPU programming and higher-level machine learning operator engineering:

- CUDA execution model and kernel launch configuration.
- Thread, warp, block, and grid-level workload mapping.
- Grid-stride loops and scalable kernel design.
- GPU memory hierarchy and memory-access optimization.
- Coalesced global-memory access.
- Shared-memory tiling and data reuse.
- Shared-memory bank conflicts.
- Register reuse and loop unrolling.
- Warp-level and block-level reductions.
- Synchronization and race-condition avoidance.
- Softmax and online Softmax.
- LayerNorm forward and backward.
- Matrix multiplication using shared-memory tiling.
- Kernel fusion, including fused Bias + GELU.
- PyTorch C++/CUDA extension development.
- Correctness and gradient validation.
- Benchmarking and profiler-based bottleneck analysis.

## Operator Development Goals

The goal was to understand how GPU kernel design choices affect the performance of ML operators that appear repeatedly in transformer workloads:

- Matrix multiplication.
- Layer normalization forward and backward.
- Numerically stable softmax.
- Fused elementwise / activation extensions.
- Naive attention score, softmax, and output stages.

The project starts from simple CUDA baselines, then adds progressively more hardware-aware optimizations such as block tiling, shared memory, padding to avoid bank conflicts, warp shuffle reductions, loop unrolling, register reuse, and PyTorch extension wrappers.

## Repository Structure

The repo is organized as a progression from CUDA fundamentals to ML operator kernels:

| Area | Files | Purpose |
|---|---|---|
| CUDA basics | `basic/0_cuda_computation.ipynb`, `1_vector_add.ipynb`, `2_memory_coalescing.ipynb`, `3_shared_memory_and_bank_conflict.ipynb`, `4_warp_level_reduction.ipynb` | Build intuition for execution hierarchy, memory coalescing, shared memory, bank conflicts, and warp reductions. |
| Week 1 operators | `project/week1_matmul_laynorm_softmax/*` | Matrix multiplication, loop unrolling, Nsight profiling, LayerNorm, and Softmax kernels. |
| PyTorch extensions | `project/week2_pytorch_extension/*` | C++/CUDA extension loading, TensorAccessor usage, LayerNorm extension, autograd checks, and fused Bias+GELU. |

## Vector Addition: CUDA Programming Foundation

![CUDA vector addition thread mapping and grid-stride loop](../assets/projects/cuda-kernels/vector-add-thread-mapping-grid-stride.png)

The vector-addition notebook is the introductory experiment in the CUDA project. It uses the simplest possible elementwise operation:

```text
C[i] = A[i] + B[i]
```

The goal is not to prove that vector addition is a complex algorithm. Instead, the notebook establishes the CUDA development workflow used by later kernels such as MatMul, LayerNorm, Softmax, and FlashAttention-style attention.

**In this section:**

- [What This Experiment Teaches](#what-this-experiment-teaches)
- [Basic One-Thread-Per-Element Kernel](#basic-one-thread-per-element-kernel)
- [Grid-Stride Loop Kernel](#grid-stride-loop-kernel)
- [Interview Example: One Thread per Element vs Grid-Stride Loop](#example-one-thread-per-element-vs-grid-stride-loop)
- [Host-Device Workflow](#host-device-workflow)
- [Correctness Result](#correctness-result)
- [Key Takeaways](#key-takeaways)

### What This Experiment Teaches

| Topic | What the notebook demonstrates |
|---|---|
| CUDA execution hierarchy | How threads, blocks, and grids are organized. |
| Thread-to-data mapping | How a global thread index maps one GPU thread to one output element. |
| Launch configuration | How to choose `blockDim`, `gridDim`, and use ceiling division to cover an input. |
| Boundary handling | Why `if (id < N)` is required when the final block has extra threads. |
| Memory management | Difference between unified memory and explicit host/device memory allocation. |
| Grid-stride loop | How fixed-size grids can process arrays larger than the number of launched threads. |
| Correctness workflow | How to compare GPU results against a CPU reference with a tolerance. |
| Error handling | How to check launch errors and asynchronous device execution errors. |

### Basic One-Thread-Per-Element Kernel

The first implementation maps one CUDA thread to one output element:

```cpp
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        c[id] = a[id] + b[id];
    }
}
```

The key indexing expression is:

```cpp
int id = blockIdx.x * blockDim.x + threadIdx.x;
```

This converts a serial CPU loop into a parallel GPU mapping:

| CPU version | CUDA version |
|---|---|
| One loop processes all elements serially. | Many GPU threads process elements in parallel. |
| `for (int i = 0; i < N; i++)` | `id = blockIdx.x * blockDim.x + threadIdx.x` |
| `C[i] = A[i] + B[i]` | `C[id] = A[id] + B[id]` |

For the basic experiment:

| Configuration | Value |
|---|---:|
| `N` | 1024 elements |
| Threads per block | 256 |
| Blocks | 4 |
| Total launched threads | 1024 |
| Memory model | Unified memory via `cudaMallocManaged` |
| Example output | `0 3 6 9 12 15 18 21 24 27` |

The first ten GPU outputs match the CPU reference outputs.

### Grid-Stride Loop Kernel

The second implementation introduces a more reusable CUDA pattern:

```cpp
__global__ void vectorAddGridStride(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}
```

A grid-stride loop decouples the input size from the number of launched threads. Each thread starts from its global thread ID and advances by the total grid size:

```text
Thread 0 -> 0, stride, 2*stride, ...
Thread 1 -> 1, stride + 1, 2*stride + 1, ...
Thread 2 -> 2, stride + 2, 2*stride + 2, ...
```

This pattern is important because it preserves coalesced access within each loop iteration. Adjacent threads in the same warp still access adjacent elements such as `A[0]`, `A[1]`, ..., `A[31]`, then `A[stride]`, `A[stride + 1]`, and so on.

### Example: One Thread per Element vs Grid-Stride Loop

This small example is useful for explaining the idea quickly in an interview.

Assume:

```text
N = 16 elements
blockDim.x = 4 threads per block
gridDim.x = 2 blocks
```

The total number of launched threads is:

```text
2 blocks * 4 threads/block = 8 threads
```

With the basic one-thread-per-element kernel:

```cpp
int id = blockIdx.x * blockDim.x + threadIdx.x;

if (id < N) {
    C[id] = A[id] + B[id];
}
```

the eight launched threads process only the first eight elements:

```text
Thread 0 -> element 0
Thread 1 -> element 1
Thread 2 -> element 2
Thread 3 -> element 3
Thread 4 -> element 4
Thread 5 -> element 5
Thread 6 -> element 6
Thread 7 -> element 7
```

Elements `8` through `15` are not processed because no launched thread has a global ID larger than `7`. To process all 16 elements with this approach, the launch configuration must create at least 16 threads.

With the grid-stride loop:

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (int i = tid; i < N; i += stride) {
    C[i] = A[i] + B[i];
}
```

the stride is:

```text
stride = 4 * 2 = 8
```

Each thread processes one element in the first pass and one more element in the second pass:

```text
Thread 0 -> elements 0 and 8
Thread 1 -> elements 1 and 9
Thread 2 -> elements 2 and 10
Thread 3 -> elements 3 and 11
Thread 4 -> elements 4 and 12
Thread 5 -> elements 5 and 13
Thread 6 -> elements 6 and 14
Thread 7 -> elements 7 and 15
```

All 16 elements are covered even though only 8 threads were launched. The key point is:

| Mapping style | Behavior |
|---|---|
| One thread per element | One thread computes exactly one output element; the launch must provide enough threads for the input size. |
| Grid-stride loop | One thread can process multiple elements separated by the grid-wide stride; the launch configuration can be reused for larger inputs. |

This does not automatically make the grid-stride version faster. Its value is scalability, flexibility, and a cleaner reusable CUDA kernel pattern.

### Host-Device Workflow

The more complete version uses explicit memory management rather than unified memory:

```text
Allocate host memory
        ↓
Initialize input arrays
        ↓
Allocate device memory with cudaMalloc
        ↓
Copy A and B from host to device
        ↓
Launch CUDA kernel
        ↓
Check launch error with cudaGetLastError
        ↓
Synchronize with cudaDeviceSynchronize
        ↓
Copy C from device to host
        ↓
Compute CPU reference output
        ↓
Validate GPU output against CPU output
        ↓
Free host and device memory
```

The notebook also uses a CUDA error-checking macro around CUDA API calls and checks both:

- `cudaGetLastError()` for immediate launch-related errors.
- `cudaDeviceSynchronize()` for asynchronous execution errors such as illegal memory access.

### Correctness Result

The grid-stride experiment uses a larger input:

| Item | Value |
|---|---:|
| `N` | `2^24 = 16,777,216` elements |
| Approximate bytes per array | 64 MiB |
| Device arrays | `dA`, `dB`, `dC` |
| Approximate device-array memory | 192 MiB |
| CPU reference | Serial vector addition |
| Tolerance | `1e-6` |
| Result | `Correctness: PASS` |

This validates that the GPU grid-stride kernel matched the CPU reference for approximately 16.7 million floating-point elements.

### What This Experiment Does Not Claim

This notebook should be described as a correctness and CUDA-programming-pattern experiment, not as a performance-optimization result.

It does not yet report:

- GPU kernel latency.
- CPU latency.
- Host-to-device or device-to-host transfer time.
- Effective memory bandwidth.
- Block-size sweeps.
- CPU-GPU speedup.
- Nsight Compute memory-transaction metrics.

The grid-stride loop is therefore not claimed to be faster than the basic kernel. Its main contribution is scalability, reusable launch configuration, and a production-style indexing pattern.

One experimental nuance is that the current grid-stride configuration still launches enough threads to cover the whole input:

```cpp
int blockSize = 256;
int gridSize = (N + blockSize - 1) / blockSize;
```

Because `gridSize * blockSize >= N`, most threads execute the loop only once. A clearer demonstration of thread reuse would fix the grid size, for example:

```cpp
int blockSize = 256;
int gridSize = 256;
```

Then only 65,536 threads would process 16,777,216 elements, and each thread would process roughly 256 elements through the grid-stride loop.

### Key Takeaways

| Takeaway | Why it matters later |
|---|---|
| Global thread indexing | Foundation for elementwise kernels, activation functions, residual adds, and tensor scaling. |
| Boundary checks | Required for safe kernels when input sizes do not align with block sizes. |
| Grid-stride loops | Reusable pattern for scalable CUDA kernels and fixed launch configurations. |
| Coalesced access | Introduces the memory-access reasoning later used in Softmax, LayerNorm, and attention kernels. |
| Correctness before performance | Establishes the validation workflow reused for more complex ML operators. |
| Explicit memory management | Prepares for measuring H2D, kernel, and D2H costs independently. |

## CUDA Memory Coalescing: Strided Access Benchmark

![CUDA memory coalescing warp access pattern and memory transactions](../assets/projects/cuda-kernels/memory-coalescing-warp-access-pattern.png)

This notebook extends the previous Vector Addition project. The vector-addition notebook mainly answers:

```text
How do we map computation onto CUDA threads?
```

The memory-coalescing notebook asks the next performance question:

```text
Even if we launch many GPU threads, why can a kernel still be slow?
```

One of the main answers is the global-memory access pattern. GPUs care not only about how much data a kernel accesses, but also whether threads in the same warp access contiguous and well-aligned memory addresses.

This project modifies vector addition from:

```cpp
C[i] = A[i] + B[i];
```

to a strided memory-access pattern:

```cpp
int idx = i * stride;
C[idx] = A[idx] + B[idx];
```

By comparing `stride = 1`, `2`, and `4`, the notebook experimentally shows the performance chain:

```text
worse coalescing -> more memory transactions -> lower effective bandwidth -> longer kernel runtime
```

**In this section:**

- [Motivation: Parallelism Is Not Enough](#motivation-parallelism-is-not-enough)
- [Memory Coalescing Intuition](#memory-coalescing-intuition)
- [Strided Vector Addition Kernel](#strided-vector-addition-kernel)
- [Concrete Warp-Level Example](#concrete-warp-level-example)
- [Why Larger Memory Stride Hurts](#why-larger-memory-stride-hurts)
- [Benchmark Setup](#benchmark-setup)
- [Experimental Results](#experimental-results)
- [Correctness Verification and Caveat](#correctness-verification-and-caveat)
- [Key Takeaways](#key-takeaways-1)

### Motivation: Parallelism Is Not Enough

A common beginner assumption is that moving computation to the GPU and launching many threads automatically makes the program fast. In practice, GPU performance is also affected by:

- Global-memory access pattern.
- Memory coalescing.
- Cache-line and memory-sector utilization.
- Number of memory transactions.
- Arithmetic intensity.
- Warp stalls.
- Synchronization overhead.

This experiment demonstrates that two kernels can have the same arithmetic work, same number of threads, and same launch configuration, yet differ by several times in runtime simply because their memory addresses are accessed differently.

This is especially important for ML kernels. Operators such as LayerNorm, Softmax, attention, embedding lookup, tensor transpose, and gather/scatter-heavy workloads often spend a large fraction of time waiting on global memory.

### Memory Coalescing Intuition

NVIDIA GPUs execute threads in warps. A warp usually contains 32 threads. Suppose each thread reads one `float`, and each `float` is 4 bytes.

If a warp accesses:

```text
Thread 0  -> A[0]
Thread 1  -> A[1]
Thread 2  -> A[2]
...
Thread 31 -> A[31]
```

then the warp accesses a contiguous 128-byte region:

```text
32 threads * 4 bytes/thread = 128 bytes
```

This is a coalesced access pattern. It can be served with a small number of memory transactions, depending on architecture and address alignment.

If the warp instead accesses:

```text
Thread 0  -> A[0]
Thread 1  -> A[4]
Thread 2  -> A[8]
...
Thread 31 -> A[124]
```

then each thread still reads only one `float`, but the addresses are spread across a much larger memory range. The GPU must issue more memory transactions, and many bytes fetched by those transactions are not useful to the current warp. The useful data has not changed, but transaction count and wasted memory traffic increase.

### Project Scope

The notebook contains four main parts:

| Part | Purpose |
|---|---|
| Configurable strided vector-add kernel | Use the same arithmetic operation while changing the physical memory access pattern. |
| Stride sweep | Run `stride = 1`, `2`, and `4` to compare coalesced and less-coalesced access. |
| CUDA-event benchmark | Measure average GPU kernel latency using CUDA events. |
| Effective-bandwidth calculation and correctness check | Compute useful bandwidth from `N * 3 * sizeof(float)` and compare GPU output with a CPU reference. |

The notebook description also planned to use Nsight Systems to analyze kernel duration, GPU utilization, memory throughput, CPU-GPU synchronization, and idle periods. However, the current saved notebook does not include `nsys profile` commands, `.nsys-rep` files, SQLite reports, timeline screenshots, or Nsight statistics. Therefore, the completed result should be described as a CUDA-event benchmark, not a full Nsight Systems profiling study.

### Strided Vector Addition Kernel

The core kernel is:

```cpp
__global__ void vectorAddStrided(
    const float* A,
    const float* B,
    float* C,
    int N,
    int stride
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    if (tid < N) {
        for (int i = tid; i < N; i += gridStride) {
            int idx = i * stride;
            C[idx] = A[idx] + B[idx];
        }
    }
}
```

There are two different meanings of stride in this kernel, and they should not be confused.

| Concept | Code | Meaning |
|---|---|---|
| Grid stride | `gridStride = blockDim.x * gridDim.x` | Controls which logical work items are assigned to the same thread across loop iterations. |
| Memory stride | `idx = i * stride` | Controls the physical distance between memory addresses accessed by neighboring threads. |

The grid-stride loop comes from the previous Vector Addition project:

```cpp
for (int i = tid; i < N; i += gridStride)
```

Its purpose is to let one thread process multiple logical elements and decouple the launch configuration from the input size.

The memory stride is the subject of this experiment:

```cpp
int idx = i * stride;
```

It intentionally changes the physical addresses accessed by neighboring threads:

```text
stride = 1: tid 0 -> A[0], tid 1 -> A[1]
stride = 2: tid 0 -> A[0], tid 1 -> A[2]
stride = 4: tid 0 -> A[0], tid 1 -> A[4]
```

In one sentence:

```text
gridStride determines which logical work items are assigned to the same thread,
while memory stride determines the physical distance between addresses accessed
by neighboring threads.
```

### Concrete Warp-Level Example

This example is useful for explaining memory coalescing in an interview. To keep the example small, consider only 8 threads, where each thread reads one 4-byte `float`.

#### Stride = 1

```cpp
idx = tid;
```

Access pattern:

```text
Thread 0 -> A[0]
Thread 1 -> A[1]
Thread 2 -> A[2]
Thread 3 -> A[3]
Thread 4 -> A[4]
Thread 5 -> A[5]
Thread 6 -> A[6]
Thread 7 -> A[7]
```

The byte addresses are:

```text
0, 4, 8, 12, 16, 20, 24, 28
```

The 8 threads read 32 bytes of useful data, and those 32 bytes are contiguous. For a full warp:

```text
Thread 0-31 -> A[0-31]
```

The access span is:

```text
32 * 4 = 128 bytes
```

This is the ideal access pattern.

#### Stride = 2

```cpp
idx = tid * 2;
```

Access pattern:

```text
Thread 0 -> A[0]
Thread 1 -> A[2]
Thread 2 -> A[4]
Thread 3 -> A[6]
...
Thread 7 -> A[14]
```

The byte addresses are:

```text
0, 8, 16, 24, 32, 40, 48, 56
```

The 8 threads still need only 32 bytes of useful data, but the access range expands to roughly 60 bytes. For a full warp:

```text
Thread 0-31 -> A[0], A[2], ..., A[62]
```

The byte-address span is approximately:

```text
63 * 4 = 252 bytes
```

The GPU must touch more sectors or cache lines, and roughly half of the fetched data is not used by the current warp.

#### Stride = 4

```cpp
idx = tid * 4;
```

Access pattern:

```text
Thread 0 -> A[0]
Thread 1 -> A[4]
Thread 2 -> A[8]
Thread 3 -> A[12]
...
Thread 7 -> A[28]
```

The byte addresses are:

```text
0, 16, 32, 48, 64, 80, 96, 112
```

For a full warp:

```text
Thread 0-31 -> A[0], A[4], ..., A[124]
```

The address span is close to:

```text
125 * 4 = 500 bytes
```

The warp still requests only:

```text
32 * 4 = 128 bytes of useful data
```

but that useful data is scattered across a much larger address range, so more memory transactions are required. This is why the arithmetic work remains identical while kernel runtime increases sharply.

### Why Larger Memory Stride Hurts

Vector addition performs one floating-point operation per output element:

```cpp
C[idx] = A[idx] + B[idx];
```

Each logical output element requires:

| Operation | Useful bytes |
|---|---:|
| Load `A[idx]` | 4 bytes |
| Load `B[idx]` | 4 bytes |
| Store `C[idx]` | 4 bytes |
| Floating-point add | 1 FLOP |

The arithmetic intensity is approximately:

```text
1 FLOP / 12 bytes = 0.083 FLOP/byte
```

This is very low, so vector addition is a memory-bound kernel. When memory stride increases:

- Useful computation does not increase.
- Useful bytes do not increase.
- Memory transactions increase.
- Cache-line and memory-sector utilization decrease.
- Effective bandwidth decreases.
- Warps spend more time waiting on memory.
- Kernel runtime increases.

The causal chain is:

```text
larger stride -> less coalesced access -> more memory transactions
              -> more memory stalls -> longer runtime
```

### Benchmark Setup

The notebook uses:

```cpp
const int N = 1 << 24;
const int strides[] = {1, 2, 4};
```

So the logical output size is:

```text
N = 2^24 = 16,777,216 elements
```

Because the largest accessed index is approximately `(N - 1) * 4`, the code allocates:

```cpp
allocCount = N * maxStride;
```

That is:

```text
16,777,216 * 4 = 67,108,864 floats
```

Each array is approximately:

```text
67,108,864 floats * 4 bytes = 268,435,456 bytes = 256 MiB
```

The device allocation contains three arrays:

```text
dA, dB, dC -> about 768 MiB total
```

The host allocation contains four arrays:

```text
hA, hB, hC_gpu, hC_cpu -> about 1 GiB total
```

The launch configuration is:

```cpp
blockSize = 256;
gridSize = 65536;
```

This launches:

```text
65,536 blocks * 256 threads/block = 16,777,216 threads
```

So in the current experiment, there is roughly one thread per logical element. Although the kernel is written as a grid-stride loop, most threads typically execute the loop body only once under this launch configuration.

### Benchmark Method

The benchmark uses 5 warmup iterations:

```cpp
warmupIters = 5;
```

Warmup reduces the effect of:

- Initial CUDA context behavior.
- Cold instruction/cache effects.
- GPU clock ramp-up.
- First-launch overhead.

The timed benchmark uses 20 iterations:

```cpp
iters = 20;
```

It measures GPU kernel time using CUDA events:

```cpp
cudaEventRecord(start);

for (int i = 0; i < iters; ++i) {
    vectorAddStrided<<<gridSize, blockSize>>>(...);
}

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

The reported latency is:

```text
average kernel latency = total CUDA-event time / iters
```

CUDA events are more appropriate than CPU timers for this measurement because they record time on the GPU execution timeline.

### Effective Bandwidth Calculation

The notebook computes:

```cpp
double bytes_total = (double)N * 3.0 * sizeof(float);
```

Each logical element contributes:

```text
read A: 4 bytes
read B: 4 bytes
write C: 4 bytes
```

Therefore:

```text
useful bytes = N * 12
```

The bandwidth reported in the notebook is effective bandwidth:

```text
effective bandwidth = useful bytes / kernel time
```

This is not necessarily the actual DRAM traffic. For `stride = 2` and `stride = 4`, poor transaction utilization means the GPU may fetch or transfer significantly more physical data than `N * 12` bytes. The falling effective bandwidth captures the fact that the same useful work takes longer to complete.

### Experimental Results

Notebook output:

```text
N = 16,777,216
allocCount = 67,108,864
blockSize = 256
gridSize = 65,536
```

| Memory stride | Kernel time | Effective bandwidth | Correctness |
|---:|---:|---:|---|
| 1 | 0.7965 ms | 252.78 GB/s | PASS |
| 2 | 1.9959 ms | 100.87 GB/s | PASS |
| 4 | 4.1621 ms | 48.37 GB/s | PASS |

The result shows a large performance gap caused only by memory-access pattern:

| Comparison | Latency increase | Bandwidth change | Interpretation |
|---|---:|---:|---|
| `stride 1 -> stride 2` | 2.51x slower | 252.78 -> 100.87 GB/s, about 60.1% lower | Skipping every other float reduces cache-line and memory-sector utilization. |
| `stride 1 -> stride 4` | 5.23x slower | 252.78 -> 48.37 GB/s, about 80.9% lower | The same useful data is spread across a much larger memory range, causing many more memory transactions. |

`stride = 4` is not exactly four times slower than `stride = 1`. Runtime is affected by memory-sector granularity, cache-line size, address alignment, L1/L2 behavior, request merging, store behavior, GPU clocks, outstanding memory requests, latency hiding, and measurement noise. Therefore, stride and runtime should not be treated as a strictly linear relationship.

### Correctness Verification and Caveat

The CPU reference uses the same memory stride:

```cpp
for (int i = 0; i < N; ++i) {
    int idx = i * stride;
    C[idx] = A[idx] + B[idx];
}
```

The GPU and CPU outputs are compared with an absolute tolerance of `1e-6`, and the notebook reports `PASS` for all three strides.

However, there is an important validation detail. For `stride = 4`, the kernel writes only:

```text
C[0], C[4], C[8], C[12], ...
```

but the current code uses:

```cpp
int checkCount = N;
```

and compares:

```text
gpu[0 ... N-1] vs cpu[0 ... N-1]
```

This can be weaker than intended because:

- `dC` is cleared with `cudaMemset` before each test.
- `hC_cpu` is allocated with `malloc`.
- `hC_cpu` is not necessarily cleared before each stride test.
- The CPU reference only writes positions `i * stride`.
- Many unwritten positions may contain uninitialized values or previous results.

Two cleaner validation methods would be:

```cpp
// Method 1: compare only positions actually written by the strided kernel.
for (int i = 0; i < N; ++i) {
    int idx = i * stride;
    if (fabsf(hC_gpu[idx] - hC_cpu[idx]) > tol) {
        ...
    }
}
```

or:

```cpp
// Method 2: clear both CPU and GPU output buffers before each test.
std::memset(hC_cpu, 0, bytes);
CUDA_CHECK(cudaMemset(dC, 0, bytes));

checkClose(hC_gpu, hC_cpu, allocCount, tol);
```

The accurate interview explanation is:

```text
The measured outputs passed the notebook's current correctness check, although
the validation could be improved by comparing only the strided indices or by
initializing the full CPU output buffer before each run.
```

This is a useful code-review point and shows the difference between a quick benchmark and a fully rigorous validation harness.

### Profiling Caveat

The notebook motivation mentions Nsight Systems, but the saved implementation currently reports CUDA-event timing and effective bandwidth only.

It is more accurate to say:

```text
The notebook planned an Nsight Systems analysis, but the recorded implementation
currently reports CUDA-event timing and effective bandwidth only.
```

Also, Nsight Systems is most useful for timeline-level analysis such as kernel duration, memory copies, synchronization, CPU-GPU interaction, and idle GPU periods. To directly prove that uncoalesced access increases memory transactions, Nsight Compute would be the better tool because it can expose global load/store efficiency, memory sectors, throughput, and warp stall reasons.

### Key Takeaways

| Lesson | Explanation |
|---|---|
| Coalescing is a warp-level concept | It is determined by the addresses accessed by neighboring threads in the same warp at the same memory instruction, not by one thread's consecutive accesses over time. |
| Grid stride and memory stride are different | Grid stride assigns logical work items to a thread; memory stride changes the physical distance between addresses accessed by neighboring threads. |
| Vector addition is memory-bound | With about `0.083 FLOP/byte`, performance is dominated by global-memory bandwidth and transaction efficiency. |
| More threads cannot fix bad access patterns | Even millions of threads cannot fully compensate for fragmented memory requests and low transaction utilization. |
| Effective bandwidth is a useful symptom | Lower effective bandwidth indicates that the same useful bytes require more time, often due to wasted memory traffic. |
| Profiling should match the question | CUDA events measure kernel time; Nsight Systems shows timelines; Nsight Compute is better for transaction-level memory diagnostics. |

## CUDA Shared Memory and Bank Conflict

![CUDA shared memory bank conflicts, strided access, and padding](../assets/projects/cuda-kernels/shared-memory-bank-conflict-overview.png)

![CUDA shared memory bank conflicts clean benchmark and padding explanation](../assets/projects/cuda-kernels/shared-memory-bank-conflict-clean-benchmark.png)

This notebook is the next step after the Vector Addition and Memory Coalescing notebooks:

```text
Vector Addition        -> thread/block/grid mapping and basic kernel workflow
Memory Coalescing      -> global-memory access pattern
Shared Memory + Banks  -> on-chip data reuse, thread cooperation, and shared-memory layout
```

It answers two practical CUDA questions:

```text
Why can shared memory accelerate a CUDA kernel?
Why can shared memory still become very slow because of bank conflicts?
```

The notebook contains three levels of experiments:

| Experiment | Purpose |
|---|---|
| Shared-memory tile copy | Practice the workflow of loading from global memory into shared memory, synchronizing, and writing back. |
| Initial bank-conflict benchmark | Explore baseline, strided, and padded access modes across multiple strides. |
| Clean 32x32 tile benchmark | Compare row access, conflicted column access with `pitch = 32`, and padded column access with `pitch = 33`. |

The clean benchmark is the most important result:

| Access pattern | Runtime |
|---|---:|
| Row access, mostly conflict-free | 0.2050 ms |
| Column access, `pitch = 32` | 12.6755 ms |
| Column access, `pitch = 33` with padding | 0.2935 ms |

The conflicted column access is about `61.8x` slower than the row-access baseline. Adding one column of padding reduces runtime from `12.6755 ms` to `0.2935 ms`, bringing performance close to the conflict-free baseline.

**In this section:**

- [Motivation: Why Shared Memory Matters](#motivation-why-shared-memory-matters)
- [Shared-Memory Tile Copy](#shared-memory-tile-copy)
- [What Is a Shared-Memory Bank?](#what-is-a-shared-memory-bank)
- [Why 32x32 Column Access Conflicts](#why-32x32-column-access-conflicts)
- [Why Padding Removes the Conflict](#why-padding-removes-the-conflict)
- [Clean Bank-Conflict Benchmark](#clean-bank-conflict-benchmark)
- [Clean Benchmark Results](#clean-benchmark-results)
- [Shared-Memory Bank Conflicts vs Global-Memory Coalescing](#shared-memory-bank-conflicts-vs-global-memory-coalescing)
- [Relationship to MatMul Tiling](#relationship-to-matmul-tiling)
- [Key Takeaways](#key-takeaways-2)

### Motivation: Why Shared Memory Matters

Global memory is large and visible to all blocks, but it has much higher latency than registers and on-chip storage. When a kernel repeatedly loads the same values from global memory, such as in matrix multiplication:

```text
C[i, j] = sum_k A[i, k] * B[k, j]
```

the same `A` and `B` elements may be reused by multiple threads. Without tiling, this can cause:

- Excessive global-memory traffic.
- Higher memory latency.
- Lower arithmetic intensity.
- Stronger memory-bound behavior.

Shared memory helps by loading a tile from global memory into on-chip memory once, then allowing threads in the same block to reuse that tile. The key value of shared memory is therefore not simply "copying data into a faster memory", but enabling **data reuse**.

Shared memory has several important properties:

- It is located on chip.
- It is visible to all threads in the same block.
- It is not directly shared across different blocks.
- Its lifetime is tied to the thread block.
- It is explicitly managed by the programmer.
- Its capacity is limited.
- Its latency is much lower than global memory when accessed efficiently.

It is commonly used in MatMul tiling, reductions, convolution, LayerNorm, Softmax, attention, transpose, and prefix-sum kernels.

### Shared-Memory Tile Copy

The first experiment implements a simple tile-copy kernel:

```text
Global memory
      ↓
Shared-memory tile
      ↓
Block synchronization
      ↓
Global memory
```

The core kernel is:

```cpp
__global__ void tileCopyShared(
    const float* __restrict__ A,
    float* __restrict__ B,
    int N
) {
    extern __shared__ float tile[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (tid < N) {
        tile[tid] = A[gid];
    }

    __syncthreads();

    if (gid < N) {
        B[gid] = tile[tid];
    }
}
```

Here:

| Variable | Meaning |
|---|---|
| `gid` | Global index across the whole grid. |
| `tid` | Local thread index within the block. |
| `tile[tid]` | Block-local shared-memory location. |

The kernel uses dynamic shared memory:

```cpp
extern __shared__ float tile[];
```

The shared-memory size is provided at kernel launch:

```cpp
size_t sharedMemBytes = blockSize * sizeof(float);

tileCopyShared<<<gridSize, blockSize, sharedMemBytes>>>(dA, dB, N);
```

With `blockSize = 256`, each block uses:

```text
256 floats * 4 bytes = 1024 bytes = 1 KiB
```

### Why `__syncthreads()` Is Used

`__syncthreads()` is a block-level barrier. It guarantees that:

- All threads in the block reach the barrier.
- Shared-memory writes before the barrier are visible to other threads in the block.
- No thread continues past the barrier until all threads in the block arrive.

In this tile-copy kernel, each thread writes `tile[tid]` and later reads the same `tile[tid]`. There is no cross-thread reuse in this exact mapping, so the barrier is not essential for this specific data dependency.

However, the experiment includes `__syncthreads()` to practice the more general shared-memory pattern used in kernels such as transpose:

```cpp
tile[threadIdx.y][threadIdx.x] = input[...];

__syncthreads();

output[...] = tile[threadIdx.x][threadIdx.y];
```

In that case, a thread may read data written by another thread, so synchronization is required.

### Boundary Check Caveat in the Tile-Copy Kernel

The tile-copy kernel currently uses:

```cpp
if (tid < N) {
    tile[tid] = A[gid];
}
```

This condition is not the right boundary check. `tid` is only the local thread index:

```text
0 <= tid < blockDim.x
```

With `blockDim.x = 256` and `N = 2^24`, `tid < N` is always true. The load should be guarded by the global index:

```cpp
if (gid < N) {
    tile[tid] = A[gid];
}

__syncthreads();

if (gid < N) {
    B[gid] = tile[tid];
}
```

The current experiment still prints `Correctness: PASS` because `N = 2^24` is divisible by `256`, so the final block has no extra out-of-range threads. This is still an important code-review point:

```text
The kernel passes for the current input size, but the load-side boundary check
should use the global index rather than the local thread index.
```

### Is Shared-Memory Copy Faster Than Direct Copy?

Not necessarily. A direct copy is:

```cpp
B[gid] = A[gid];
```

It requires one global load and one global store. The shared-memory copy requires:

- One global load.
- One shared-memory store.
- One barrier.
- One shared-memory load.
- One global store.

It does not reduce global-memory traffic and does not create data reuse. Therefore, in this experiment shared memory is mainly a teaching tool, not a performance optimization.

The accurate lesson is:

```text
Shared memory improves performance when data loaded from global memory is reused
by multiple operations or threads. Using shared memory only as a pass-through
buffer usually adds overhead instead of improving performance.
```

### What Is a Shared-Memory Bank?

Shared memory is fast, but it is not a single uniform memory array with unlimited ports. On a typical NVIDIA GPU, shared memory is divided into 32 banks, and a warp usually contains 32 threads.

If the 32 threads in a warp access 32 different banks, the accesses can proceed in parallel. If multiple threads access the same bank but different addresses, a **bank conflict** occurs. These accesses must be split or serialized, increasing latency and reducing effective shared-memory bandwidth.

![Shared-memory bank mapping](../assets/projects/cuda-kernels/shared-memory-bank-mapping.png)

For 4-byte `float` data, a common simplified model is:

```text
bank = (byte_address / 4) mod 32
```

If the array starts at an aligned address, this is approximately:

```text
bank = index mod 32
```

Examples:

```text
shared[0]  -> bank 0
shared[1]  -> bank 1
shared[2]  -> bank 2
...
shared[31] -> bank 31
shared[32] -> bank 0
shared[33] -> bank 1
```

Conflict-free access:

```text
Thread 0  -> shared[0]  -> bank 0
Thread 1  -> shared[1]  -> bank 1
...
Thread 31 -> shared[31] -> bank 31
```

Each thread accesses a different bank, so there is no bank conflict.

With stride 2:

```text
bank = (2 * t) mod 32
```

threads `0` and `16` map to the same bank, threads `1` and `17` map to the same bank, and so on. This commonly forms a 2-way bank conflict.

With stride 4:

```text
bank = (4 * t) mod 32
```

threads `0`, `8`, `16`, and `24` map to the same bank, creating a 4-way conflict.

With stride 32:

```text
bank = (32 * t) mod 32 = 0
```

all threads map to bank 0. If they access different addresses in that bank, this forms the most severe 32-way bank conflict.

One important exception is broadcast behavior: if multiple threads read the exact same shared-memory address, the hardware may broadcast the value instead of treating it as a normal conflict. Bank-conflict analysis must consider bank mapping, whether addresses are identical or different, read vs write behavior, and the GPU architecture.

### Why 32x32 Column Access Conflicts

The clean experiment uses a 2D shared-memory tile. Without padding:

```cpp
pitch = 32;
idx = row * pitch + col;
```

For row access:

```cpp
idx = ty * pitch + tx;
```

Within a warp, `ty` is usually fixed and `tx = 0...31`, so the accessed indices are:

```text
row * 32 + 0
row * 32 + 1
...
row * 32 + 31
```

The bank mapping is:

```text
bank = (row * 32 + tx) mod 32 = tx
```

The 32 threads access 32 different banks, so the access is mostly conflict-free.

For column access:

```cpp
idx = tx * pitch + col;
```

With `pitch = 32`:

```text
idx = tx * 32 + col
bank = (tx * 32 + col) mod 32
```

Since `tx * 32 mod 32 = 0`, every thread maps to:

```text
bank = col
```

The warp accesses:

```text
Thread 0  -> shared[0][col]
Thread 1  -> shared[1][col]
Thread 2  -> shared[2][col]
...
Thread 31 -> shared[31][col]
```

These are different addresses, but they all map to the same bank, forming a classic 32-way bank conflict.

### Why Padding Removes the Conflict

The padded version changes the pitch from 32 to 33:

```cpp
float tile[32][33];
```

or, with dynamic shared memory:

```cpp
pitch = TILE + 1;
```

Now column access is:

```text
idx = tx * 33 + col
bank = (tx * 33 + col) mod 32
```

Because `33 mod 32 = 1`:

```text
bank = (tx + col) mod 32
```

As `tx` goes from `0` to `31`, the access spreads across all 32 banks:

```text
Thread 0 -> bank col
Thread 1 -> bank col + 1
Thread 2 -> bank col + 2
...
```

Padding changes the row pitch so that successive rows no longer start in the same bank. It does not change the algorithm or reduce the number of shared-memory accesses; it changes the layout so that the same warp-level access pattern maps to different banks.

### Clean Bank-Conflict Benchmark

The final clean benchmark uses:

```cpp
dim3 block(32, 8);
dim3 grid(256);
```

Each block has:

```text
32 * 8 = 256 threads
```

The largest shared-memory tile is:

```text
32 * 33 floats = 4224 bytes = about 4.12 KiB
```

Each kernel repeats the shared-memory access:

```cpp
iters = 4096;
```

The benchmark uses:

```text
warmup = 5
runs = 20
```

and measures average kernel latency with CUDA events.

The benchmark compares three modes:

| Mode | Access pattern | Expected behavior |
|---|---|---|
| Mode 0 | Row access: `idx = ty * pitch + tx` | Threads in a warp access consecutive elements and different banks. |
| Mode 1 | Column access, `pitch = 32`: `idx = tx * 32 + col` | Threads in a warp map to the same bank, causing severe 32-way conflict. |
| Mode 2 | Column access, `pitch = 33`: `idx = tx * 33 + col` | Padding shifts row starts across banks and removes the main conflict. |

### Clean Benchmark Results

Notebook output:

| Mode | Runtime | Interpretation |
|---|---:|---|
| Row access, no major conflict | 0.2050 ms | Conflict-free baseline. |
| Column access, `pitch = 32` | 12.6755 ms | Severe 32-way bank conflict. |
| Column access, `pitch = 33` | 0.2935 ms | Padding removes the catastrophic conflict penalty. |

The conflicted column access is:

```text
12.6755 / 0.2050 = 61.83x slower
```

This is larger than a simple 32-way serialization estimate, which shows that runtime is affected by more than a simplified conflict degree. Other factors can include instruction scheduling, loop dependencies, compiler-generated instructions, warp scheduling, pipeline utilization, shared-memory replay behavior, address calculation, and the measurement configuration.

The padded version is:

```text
12.6755 / 0.2935 = 43.19x faster than the conflicted version
```

Compared with the row baseline:

```text
0.2935 / 0.2050 = 1.43x slower
```

This is reasonable because row access and padded column access still have different index calculations and instruction paths. The conclusion is not that padded access is exactly as fast as row access. The correct conclusion is:

```text
Padding removes the catastrophic bank-conflict penalty and restores performance
close to the conflict-free baseline.
```

### Why the Initial Bank-Conflict Benchmark Was Less Clean

The notebook also contains an earlier bank-conflict benchmark with outputs such as:

```text
stride 1 baseline: 1.4614 ms
stride 1 strided:  1.5553 ms
stride 1 padded:   5.3495 ms
```

That experiment is useful for exploration, but its modes do not perform equivalent work. For example, the padded mode uses different branch behavior, address calculation, active-thread count, data layout, and per-block work. Because the work is not identical across modes, the absolute runtimes cannot cleanly prove that padding restores performance.

The later `bank_conflict_clean.cu` experiment provides the more reliable comparison and should be the main result used in an interview.

### Shared-Memory Bank Conflicts vs Global-Memory Coalescing

These two ideas are related but different:

| Concept | What it optimizes | Main question |
|---|---|---|
| Global-memory coalescing | How warp requests are grouped into global-memory transactions. | Are neighboring threads accessing contiguous and aligned global addresses? |
| Shared-memory bank-conflict avoidance | How warp requests are distributed across shared-memory banks. | Are neighboring threads mapping to different shared-memory banks? |

In one sentence:

```text
Coalescing optimizes how warp requests are grouped for global memory, while
bank-conflict avoidance optimizes how warp requests are distributed across
shared-memory banks.
```

### Relationship to Matrix Transpose

Bank conflict is classically illustrated by tiled matrix transpose:

```cpp
__shared__ float tile[32][32];
```

Global-memory load:

```cpp
tile[threadIdx.y][threadIdx.x] = input[...];
```

This is usually row-wise:

- Global load is coalesced.
- Shared-memory write is conflict-free.

But transpose read:

```cpp
tile[threadIdx.x][threadIdx.y]
```

becomes column access. With row pitch 32, the access can map all threads to the same bank. The standard fix is:

```cpp
__shared__ float tile[32][33];
```

This is exactly what the clean benchmark simulates.

### Relationship to MatMul Tiling

Shared memory improves matrix multiplication when data is reused by multiple threads, but the layout matters. A tile size such as 32 can create bank conflicts when consecutive threads access shared-memory addresses separated by 32 banks. Padding the shared-memory tile changes the stride and avoids conflict-heavy access.

![MatMul bank conflict](../assets/projects/cuda-kernels/matmul-bank-conflict.png)

### Shared Memory and Occupancy Trade-off

Shared memory is not always better, and using more shared memory is not always better. Each SM has limited shared-memory capacity. If each block uses more shared memory, fewer blocks may be resident on the same SM.

For example, if an SM has 64 KiB available for blocks:

| Shared memory per block | Possible effect |
|---:|---|
| 8 KiB | More blocks can be resident. |
| 32 KiB | Fewer blocks can be resident. |
| 64 KiB | Potentially only one block can be resident. |

Shared-memory tiling must balance:

- Tile size.
- Data reuse.
- Shared-memory usage.
- Registers per thread.
- Resident blocks per SM.
- Active warps.
- Occupancy.
- Synchronization overhead.

The accurate trade-off is:

```text
Larger tiles may improve data reuse, but they also consume more shared memory
and can reduce occupancy.
```

### Key Takeaways

| Lesson | Explanation |
|---|---|
| Shared memory helps through reuse | It is useful when data loaded from global memory is reused by multiple operations or threads. |
| Shared memory is not automatically faster | A pass-through shared-memory copy can add overhead without reducing global-memory traffic. |
| Bank conflicts are warp-level layout problems | They depend on how threads in the same warp map their shared-memory addresses to banks. |
| Padding changes bank mapping | Changing pitch from 32 to 33 shifts row starts across banks and removes the classic column-access conflict. |
| Clean microbenchmarks matter | To prove a layout optimization, each benchmark mode should perform comparable work. |
| Occupancy still matters | Larger shared-memory tiles can improve reuse but reduce resident blocks and active warps. |

## CUDA Warp-Level Reduction

![CUDA warp-level reduction with shuffle and block reduction](../assets/projects/cuda-kernels/warp-level-reduction-overview.png)

This notebook is the next CUDA fundamentals experiment after Vector Addition, Memory Coalescing, and Shared Memory Bank Conflict:

```text
Vector Addition              -> thread/block/grid mapping and basic kernel workflow
Memory Coalescing            -> global-memory access pattern
Shared Memory + Bank Conflict -> block-level data sharing and shared-memory layout
Warp-Level Reduction         -> warp-level thread communication and efficient parallel reduction
```

It answers two core questions:

```text
How can a large array be reduced into one result in parallel?
Why is __shfl_down_sync often more efficient than a reduction implemented entirely through shared memory?
```

The notebook implements a hierarchical sum reduction:

```text
Each thread accumulates several input elements
                    ↓
Each warp performs a shuffle reduction
                    ↓
Lane 0 of each warp writes one partial sum to shared memory
                    ↓
The first warp reduces all warp partial sums
                    ↓
Each block writes one partial sum
                    ↓
CPU performs the final reduction across blocks
```

![Reduction tree](../assets/projects/cuda-kernels/warp-level-reduction-tree.png)

The experiment processes:

```text
N = 2^20 = 1,048,576 float elements
```

and reports:

```text
GPU sum = 523641.625000
CPU sum = 523641.625000
Correctness: PASS
```

This validates that the hierarchical warp/block reduction matches the CPU reference for the tested input.

**In this section:**

- [Motivation: Reduction Is a Core ML Operation](#motivation-reduction-is-a-core-ml-operation)
- [Why Not Let Every Thread Write the Same Output?](#why-not-let-every-thread-write-the-same-output)
- [Why Use Warp-Level Shuffle?](#why-use-warp-level-shuffle)
- [What Is a Warp?](#what-is-a-warp)
- [What `__shfl_down_sync` Does](#what-__shfl_down_sync-does)
- [Warp Reduction Function](#warp-reduction-function)
- [Interview-Friendly 8-Lane Example](#interview-friendly-8-lane-example)
- [Kernel Layer 1: Thread-Local Accumulation](#kernel-layer-1-thread-local-accumulation)
- [Kernel Layer 2: Warp-Level Reduction](#kernel-layer-2-warp-level-reduction)
- [Kernel Layer 3: First Warp Performs Block Reduction](#kernel-layer-3-first-warp-performs-block-reduction)
- [Complete Reduction Hierarchy](#complete-reduction-hierarchy)
- [Experiment Configuration](#experiment-configuration)
- [Correctness Verification](#correctness-verification)
- [Key Takeaways](#key-takeaways-3)

### Motivation: Reduction Is a Core ML Operation

Reduction combines many values into fewer values, often one final result. Common reductions include:

- Sum.
- Maximum.
- Minimum.
- Mean.
- Variance.
- Logical AND/OR.

For a sum reduction:

```text
S = sum_i x[i]
```

Reduction appears throughout ML workloads:

| Workload component | Reduction pattern |
|---|---|
| Loss calculation | Sum or mean over examples/tokens. |
| Dot product | Sum of elementwise products. |
| LayerNorm | Mean and variance reductions. |
| BatchNorm | Batch statistics. |
| Softmax | Maximum reduction and exponential-sum reduction. |
| Attention | Score normalization and row-wise reductions. |
| Gradient clipping | Gradient norm reduction. |
| Global pooling | Reduce spatial or sequence dimensions. |

Reduction is therefore not an isolated CUDA trick; it is a building block for many high-performance ML kernels.

### Why Not Let Every Thread Write the Same Output?

The most direct but incorrect idea is:

```cpp
out[0] += in[idx];
```

If many threads execute this at the same time, a data race occurs:

- Multiple threads read the old value.
- Multiple threads compute their own updates.
- Multiple threads write to the same address.
- Some updates are lost.

One possible fix is:

```cpp
atomicAdd(out, value);
```

However, if every thread performs an atomic add to the same global-memory address, the kernel suffers heavy contention and serialization. A more efficient reduction usually uses a hierarchy:

1. Each thread accumulates several elements in registers.
2. Each warp reduces its thread-local values.
3. Each block combines the warp results.
4. The grid combines block-level partial sums.

This greatly reduces global atomic operations and global-memory writes.

### Why Use Warp-Level Shuffle?

A traditional block reduction often uses shared memory:

```cpp
shared[tid] = value;
__syncthreads();

shared[tid] += shared[tid + offset];
__syncthreads();
```

This works, but it requires:

- Shared-memory stores.
- Shared-memory loads.
- Multiple `__syncthreads()` barriers.
- Repeated shared-memory indexing.

Warp-level shuffle allows one thread to directly read another lane's register value inside the same warp:

```cpp
__shfl_down_sync(mask, value, offset);
```

The advantages are:

- Intermediate values do not need to be written to shared memory at every step.
- Block-level synchronization is not needed inside each warp-level reduction step.
- Data moves through the warp shuffle network between registers.
- A 32-lane warp reduction takes only `log2(32) = 5` shuffle steps.

The core motivation is:

```text
Replace repeated shared-memory communication with register-level warp communication.
```

### What Is a Warp?

On NVIDIA GPUs, a warp usually contains 32 threads. The threads in a warp are called lanes:

```text
lane 0, lane 1, lane 2, ..., lane 31
```

A 256-thread block contains:

```text
256 / 32 = 8 warps
```

The code computes lane and warp IDs with:

```cpp
int lane = tid & 31;
int warp = tid >> 5;
```

These bit operations are efficient forms of:

```text
lane = tid % 32
warp = floor(tid / 32)
```

because 32 is a power of two.

### What `__shfl_down_sync` Does

The core primitive is:

```cpp
__shfl_down_sync(mask, val, offset);
```

It lets the current lane read the `val` register from:

```text
lane + offset
```

For example:

```cpp
float other = __shfl_down_sync(mask, val, 16);
```

means:

```text
Lane 0 reads lane 16
Lane 1 reads lane 17
Lane 2 reads lane 18
...
Lane 15 reads lane 31
```

For lanes 16-31, the source lane would be outside the warp, so those returned values should not be used as valid reduction results. In a downward shuffle reduction, only lower lanes keep meaningful partial sums, and the complete result is eventually guaranteed only in lane 0.

### The Mask Argument

The notebook uses:

```cpp
unsigned mask = __activemask();
```

The mask indicates which lanes are active participants in the warp primitive. A full warp is often represented as:

```cpp
0xffffffff
```

or 32 bits set to 1.

`__activemask()` returns the active-lane mask for threads executing the current instruction. The mask is not just metadata; it defines which lanes must participate together in the shuffle operation. Threads named in the mask should execute the same warp primitive consistently.

In this notebook, the block size is 256 threads, so all warps are full 32-thread warps and mask handling is relatively simple.

### Warp Reduction Function

The core device function is:

```cpp
__device__ __forceinline__
float warpReduceSum(float val) {
    unsigned mask = __activemask();

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    return val;
}
```

The offsets are:

```text
16, 8, 4, 2, 1
```

This forms a tree reduction.

### Interview-Friendly 8-Lane Example

For an easier explanation, imagine a warp has only 8 lanes:

```text
Lane:   0  1  2  3  4  5  6  7
Value:  1  2  3  4  5  6  7  8
```

The goal is:

```text
1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36
```

Step 1, `offset = 4`:

```text
Lane 0: 1 + 5 = 6
Lane 1: 2 + 6 = 8
Lane 2: 3 + 7 = 10
Lane 3: 4 + 8 = 12
```

Effective state:

```text
6, 8, 10, 12
```

Step 2, `offset = 2`:

```text
Lane 0: 6 + 10 = 16
Lane 1: 8 + 12 = 20
```

Effective state:

```text
16, 20
```

Step 3, `offset = 1`:

```text
Lane 0: 16 + 20 = 36
```

Only lane 0 has the complete reduction result. An 8-lane reduction needs `log2(8) = 3` steps; a real 32-lane warp reduction needs `log2(32) = 5` steps.

### Kernel Layer 1: Thread-Local Accumulation

The kernel first computes:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
```

and then performs a grid-stride loop:

```cpp
for (int i = idx; i < N; i += stride) {
    sum += in[i];
}
```

Each thread accumulates several input elements into a register variable `sum`. This has several benefits:

- One thread can process multiple input elements.
- The launch configuration can be decoupled from the input size.
- The reduction later needs to combine fewer values.
- Thread-local partial sums live in registers.
- Neighboring threads still access neighboring addresses during each loop iteration, preserving coalesced global loads.

In the current notebook configuration:

```text
N = 1,048,576
blockSize = 256
gridSize = 4096
total threads = 4096 * 256 = 1,048,576
```

The total thread count equals `N`, so each thread mostly reads one element. The code supports grid-stride reuse, but the current launch configuration does not strongly demonstrate multi-element accumulation per thread.

A clearer demonstration would use a fixed grid:

```cpp
gridSize = 256;
blockSize = 256;
```

Then:

```text
total threads = 256 * 256 = 65,536
average elements per thread = 1,048,576 / 65,536 = 16
```

### Kernel Layer 2: Warp-Level Reduction

After each thread has a local sum, the kernel runs:

```cpp
sum = warpReduceSum(sum);
```

This combines the 32 thread-local sums inside each warp. The full warp sum ends up in lane 0 of that warp.

The code then writes one partial sum per warp:

```cpp
if (lane == 0) {
    warpSums[warp] = sum;
}
```

For a 256-thread block:

```text
256 / 32 = 8 warps
```

so the block writes only:

```text
warpSums[0], warpSums[1], ..., warpSums[7]
```

This is much less shared-memory traffic than writing every thread's value to shared memory at each reduction stage.

### Why Shared Memory Is Still Needed

Warp shuffle works only inside one warp. It cannot directly let warp 0 read a register from warp 1. Cross-warp reduction therefore still needs an intermediate communication mechanism.

The kernel uses:

```cpp
__shared__ float warpSums[32];
```

Each warp's lane 0 writes:

```cpp
warpSums[warp]
```

The block-level design is therefore:

```text
Warp registers
       ↓
Shared-memory partial sums
       ↓
First warp
```

In other words:

```text
within a warp: use shuffle
across warps: use a small shared-memory buffer
```

### Why `__syncthreads()` Is Required Here

After lane 0 of each warp writes its partial sum:

```cpp
if (lane == 0) {
    warpSums[warp] = sum;
}

__syncthreads();
```

the barrier guarantees:

- All warp lane-0 threads have written to `warpSums`.
- The first warp reads the partial sums only after all writes are visible.

This is a real cross-warp communication point. Warp-level shuffle cannot synchronize the whole block, and warp 0 may reach the read step earlier than other warps. Without `__syncthreads()`, warp 0 could read uninitialized, old, or incomplete partial sums.

### Kernel Layer 3: First Warp Performs Block Reduction

The first warp reduces the warp partial sums:

```cpp
float blockSum = 0.0f;

if (warp == 0) {
    int numWarps = (blockDim.x + 31) / 32;

    blockSum = (lane < numWarps)
        ? warpSums[lane]
        : 0.0f;

    blockSum = warpReduceSum(blockSum);

    if (lane == 0) {
        out[blockIdx.x] = blockSum;
    }
}
```

For a 256-thread block, `numWarps = 8`. Therefore:

```text
Lane 0 reads warpSums[0]
Lane 1 reads warpSums[1]
...
Lane 7 reads warpSums[7]
Lane 8-31 use 0
```

The first warp performs another `warpReduceSum`, and `warp 0, lane 0` writes the block result:

```cpp
out[blockIdx.x] = blockSum;
```

### Complete Reduction Hierarchy

With a 256-thread block:

```text
Input elements
      ↓
256 thread-local sums
      ↓
8 warp-level sums
      ↓
8 shared-memory values
      ↓
1 block-level sum
      ↓
out[blockIdx.x]
```

The grid has 4096 blocks, so the GPU outputs 4096 block partial sums. The CPU then computes the final global sum:

```cpp
float gpuSum = 0.0f;

for (size_t i = 0; i < outCount; ++i) {
    gpuSum += hOut[i];
}
```

### Why the Final Step Runs on CPU in This Notebook

The notebook's main goal is to study warp-level and block-level reduction, so the GPU performs only the first-stage reduction. It compresses 1,048,576 input elements into 4096 block partial sums, then the CPU performs the final reduction.

This keeps the code simple, verifies the first GPU reduction stage, avoids recursive kernel launches, and keeps the focus on warp shuffle.

In a production implementation, the final stage would usually remain on GPU, using one of these strategies:

| Strategy | Idea | Trade-off |
|---|---|---|
| Multi-stage GPU reduction | Repeatedly launch kernels until only one partial sum remains. | Simple and scalable, but adds kernel launches. |
| Block-level atomic | Each block calls `atomicAdd(globalSum, blockSum)`. | Reduces atomics from `N` to number of blocks, but still has atomic contention. |
| Cooperative groups / single-kernel strategy | Use grid-wide synchronization with cooperative launch. | More complex and has launch/hardware constraints. |

### Experiment Configuration

| Item | Value |
|---|---:|
| GPU | NVIDIA Tesla T4 |
| CUDA architecture | `sm_75` |
| Input size | `N = 1 << 20 = 1,048,576` |
| Input values | `hIn[i] = 0.001f * (i % 1000)` |
| Block size | 256 threads |
| Grid size | 4096 blocks |
| Output count | 4096 block partial sums |

The input values repeat periodically:

```text
0.000, 0.001, 0.002, ..., 0.999, then repeat
```

### Correctness Verification

The CPU reference uses a double accumulator:

```cpp
static float cpuSum(const float* a, int N) {
    double acc = 0.0;

    for (int i = 0; i < N; ++i) {
        acc += a[i];
    }

    return static_cast<float>(acc);
}
```

Using `double` reduces the CPU reference accumulation error. The notebook reports:

| Metric | Value |
|---|---:|
| GPU sum | 523641.625000 |
| CPU sum | 523641.625000 |
| Tolerance | `1e-2` |
| Correctness | PASS |

The printed results match exactly in this run.

### Floating-Point Reduction Caveat

Floating-point addition is not strictly associative:

```text
(a + b) + c != a + (b + c)
```

A CPU serial reduction usually accumulates values in one sequential order:

```text
(((x0 + x1) + x2) + x3) ...
```

A GPU tree reduction combines values in a different order:

```text
(x0 + x16), (x1 + x17), ...
```

and then continues merging partial sums. Because rounding happens in a different order, the GPU and CPU results may differ slightly. Reduction validation should therefore use a tolerance:

```cpp
fabs(gpu - cpu) < tol
```

For larger reductions, a combined absolute and relative tolerance is often more robust:

```cpp
diff <= atol + rtol * fabs(cpu)
```

### Key Takeaways

| Lesson | Explanation |
|---|---|
| Reduction is foundational for ML kernels | Softmax, LayerNorm, attention, loss functions, and normalization all depend on reductions. |
| Avoid one global atomic per element | Direct atomic accumulation into one global address causes heavy serialization. |
| Warp shuffle reduces shared-memory traffic | Values move between lanes through registers instead of repeated shared-memory stores/loads. |
| The full warp result is in lane 0 | After a downward shuffle reduction, only lane 0 should write the warp result. |
| Shared memory is still needed across warps | Warp shuffle cannot communicate directly between different warps. |
| `__syncthreads()` is required for cross-warp exchange | The first warp must not read `warpSums` before other warps have written them. |
| Final GPU reductions need another stage | A complete production reduction would finish remaining block partial sums on GPU. |
| Floating-point tolerance is required | CPU and GPU reduction orders differ, so exact equality is not a safe validation rule. |

## Loop Unrolling and Register Optimization

![CUDA loop unrolling and register optimization mechanism, trade-off, and measured results](../assets/projects/cuda-kernels/loop-unrolling-register-optimization.png)

This CUDA fundamentals experiment studies how loop unrolling, register-resident temporary variables, and constant-memory broadcast affect a dot-product style workload.

The computation is:

```text
out[i] = sum_{k=0}^{K-1} A[i, k] * B[k]
```

Each CUDA thread owns one row of `A` and computes a dot product against the shared vector `B`.

The experiment compares three versions:

1. Baseline loop.
2. Unroll x4 with `B` in global memory.
3. Unroll x4 with `B` in constant memory.

The experiment runs on an NVIDIA A100:

```text
M = 1,048,576
K = 1024
UNROLL = 4
```

All three kernels pass the correctness check.

**In this section:**

- [Motivation: Loop Overhead and Register Reuse](#motivation-loop-overhead-and-register-reuse)
- [Concrete Unrolling Example](#concrete-unrolling-example)
- [Baseline Dot-Product Kernel](#baseline-dot-product-kernel)
- [Unroll + Register Kernel](#unroll-register-kernel)
- [Constant-Memory Version](#constant-memory-version)
- [Loop Unrolling Benchmark Results](#loop-unrolling-benchmark-results)
- [Why Constant Memory Did Not Improve Further](#why-constant-memory-did-not-improve-further)

### Motivation: Loop Overhead and Register Reuse

A normal inner loop looks like this:

```cpp
for (int k = 0; k < K; ++k) {
    acc += Ai[k] * B[k];
}
```

Besides the useful multiply-add work, each iteration also requires:

- Updating the loop counter.
- Checking the loop condition.
- Executing the branch.
- Computing the next address.

Loop unrolling processes multiple elements per loop iteration. This reduces loop-control overhead and gives the compiler more room for instruction scheduling.

Register optimization keeps thread-local values such as:

```cpp
float acc;
float a;
float b;
```

resident in registers where possible, avoiding unnecessary memory traffic for temporary values.

### Concrete Unrolling Example

Assume:

```text
K = 8
UNROLL = 4
```

The baseline loop executes eight iterations:

```text
k = 0
k = 1
k = 2
...
k = 7
```

After unrolling by 4, the outer loop executes only two iterations:

```text
First iteration:  k = 0, 1, 2, 3
Second iteration: k = 4, 5, 6, 7
```

The expanded inner work is roughly:

```cpp
acc += Ai[k]     * B[k];
acc += Ai[k + 1] * B[k + 1];
acc += Ai[k + 2] * B[k + 2];
acc += Ai[k + 3] * B[k + 3];
```

The loop condition, counter update, and branch frequency are reduced to about one quarter of the baseline loop.

### Baseline Dot-Product Kernel

The baseline kernel uses a simple loop:

```cpp
for (int k = 0; k < K; ++k) {
    acc += Ai[k] * B[k];
}
```

Each thread:

- Reads one full row of `A`.
- Repeatedly reads the common vector `B`.
- Uses one accumulator for the dot product.
- Writes one output element.

This baseline is easy to understand and provides the reference for measuring the impact of loop unrolling.

### Unroll + Register Kernel

The optimized version uses:

```cpp
#define UNROLL 4
```

The core loop is:

```cpp
for (; k + UNROLL - 1 < K; k += UNROLL) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        float a = Ai[k + u];
        float b = B[k + u];
        acc += a * b;
    }
}
```

It also includes a tail loop:

```cpp
for (; k < K; ++k) {
    acc += Ai[k] * B[k];
}
```

The tail loop handles cases where `K` is not divisible by `UNROLL`. For example:

```text
K = 10
UNROLL = 4
```

The unrolled loop processes the first eight elements, and the tail loop processes the final two elements.

### Constant-Memory Version

The constant-memory version stores `B` as:

```cpp
__constant__ float Bc[MAX_K];
```

and copies the host vector with:

```cpp
cudaMemcpyToSymbol(Bc, hB, bytesB);
```

This workload appears suitable for constant memory because, at the same `k` iteration, all threads in a warp read the same `B` element:

```text
Thread 0  -> B[k]
Thread 1  -> B[k]
...
Thread 31 -> B[k]
```

That access pattern can use constant-cache broadcast.

### Loop Unrolling Benchmark Results

Benchmark setting from notebook: NVIDIA A100, dot-product style workload with `M=1,048,576`, `K=1024`, and `UNROLL=4`.

| Kernel | Correctness | Time | GFLOP/s | Speedup vs Baseline |
|---|---|---:|---:|---:|
| Baseline | PASS | 8.4628 ms | 253.76 | 1.00x |
| Unroll + registers, global B | PASS | 7.3327 ms | 292.86 | 1.15x |
| Unroll + registers, constant B | PASS | 7.3741 ms | 291.22 | 1.15x |

The unrolled global-memory version improves runtime by:

```text
8.4628 / 7.3327 ~= 1.15x
```

or about 15%.

The constant-memory version performs almost the same as the global-memory unrolled version and does not provide an additional speedup in this benchmark.

### Why Constant Memory Did Not Improve Further

The vector `B` is small:

```text
1024 * 4 = 4096 bytes = 4 KiB
```

It can already fit easily in cache.

By contrast, `A` is much larger:

```text
1,048,576 * 1024 * 4 ~= 4 GiB
```

The dominant memory traffic therefore comes from reading `A`, not from reading `B`. Even if `B` access is further optimized through constant-memory broadcast, the total runtime changes little because `B` is not the main bandwidth cost.

The main lesson is:

```text
Loop unrolling helps by reducing loop overhead and improving instruction
scheduling, but constant memory only helps when the optimized operand is a
meaningful part of the total bottleneck.
```

## Matrix Multiplication

![CUDA matrix multiplication optimization overall pipeline](../assets/projects/cuda-kernels/matmul-optimization-overall-pipeline.png)

Matrix multiplication computes:

```text
C = A * B
A: M x K
B: K x N
C: M x N
```

The CPU baseline has `O(M*N*K)` work and becomes too slow for large matrices. The CUDA version maps output elements or output tiles to GPU threads/blocks.

![GPU matmul mapping](../assets/projects/cuda-kernels/matmul-gpu-mapping.png)

### GEMM Optimization Case Study

![CUDA MatMul tile size trade-off between 16x16 and 32x32](../assets/projects/cuda-kernels/matmul-tile16-vs-tile32-tradeoff.png)

**In this section:**

- [GEMM Optimization Roadmap](#gemm-optimization-roadmap)
- [GEMM Benchmark Results](#gemm-benchmark-results)
- [Motivation: Why GEMM Matters](#motivation-why-gemm-matters)
- [Naive GEMM and Repeated Global-Memory Access](#naive-gemm-and-repeated-global-memory-access)
- [Project Scope: Five Kernels](#project-scope-five-kernels)
- [Matrix Layout and Indexing](#matrix-layout-and-indexing)
- [Concrete Tile Mapping Example](#concrete-tile-mapping-example)
- [Kernel 1: Naive Matrix Multiplication](#kernel-1-naive-matrix-multiplication)
- [Kernel 2: Block Tiling with Global Loads](#kernel-2-block-tiling-with-global-loads)
- [Kernel 3: Shared-Memory Tiled MatMul, Tile 16](#kernel-3-shared-memory-tiled-matmul-tile-16)
- [Kernel 4: Tile 32 with Bank Conflict](#kernel-4-tile-32-with-bank-conflict)
- [Kernel 5: Tile 32 with Padding](#kernel-5-tile-32-with-padding)
- [Benchmarking and Correctness](#benchmarking-and-correctness)
- [Profiler Metrics Snapshot](#profiler-metrics-snapshot)
- [Boundary and Synchronization Caveat](#boundary-and-synchronization-caveat)
- [Tile Size 16 vs 32 Trade-off](#tile-size-16-vs-32-trade-off)
- [Distance from cuBLAS](#distance-from-cublas)

This GEMM notebook starts from the most basic matrix multiplication kernel and progressively implements five CUDA versions:

1. Naive MatMul.
2. Block-tiled MatMul that still reads directly from global memory.
3. Shared-memory tiled MatMul with `TILE = 16`.
4. Shared-memory `TILE = 32` with an intentionally constructed bank conflict.
5. Shared-memory `TILE = 32` with padding to remove the main bank conflict.

The experiment uses:

```text
M = N = K = 1024
C = A x B
GPU = NVIDIA Tesla T4
CUDA = CUDA 12.x toolchain
```

All five kernels pass the correctness check.

### GEMM Optimization Roadmap

The GEMM path is intentionally staged so that each version isolates one optimization idea:

1. **Naive kernel:** each thread computes one output element and repeatedly reads from global memory.
2. **Block tiling:** each block computes a tile of `C`, establishing the tile-based work partitioning.
3. **Shared-memory tiling:** tiles of `A` and `B` are loaded into shared memory and reused by threads in the block.
4. **Tile-size tuning:** compare `TILE = 16` and `TILE = 32`.
5. **Bank-conflict experiment:** intentionally expose how a bad shared-memory layout can make a larger tile slower.
6. **Padding:** change the shared-memory pitch to avoid conflict-heavy access.

This path matters because it separates two ideas that are often mixed together:

```text
Tiling organizes the computation.
Shared memory turns that organization into explicit data reuse.
```

### GEMM Benchmark Results

Environment from notebook: Tesla T4, CUDA 12.x toolchain. Matrix size: `M=N=K=1024`.

| Kernel | Correctness | Time | GFLOP/s | Speedup vs Naive |
|---|---|---:|---:|---:|
| Naive MatMul baseline | PASS | 9.1753 ms | 234.05 | 1.00x |
| Block tiled, still global loads | PASS | 4.0592 ms | 529.04 | 2.26x |
| Shared-memory tiled, Tile 16 | PASS | 2.4577 ms | 873.76 | 3.73x |
| Shared-memory tiled, Tile 32 with bank conflict | PASS | 10.4073 ms | 206.34 | 0.88x |
| Shared-memory Tile 32 with padding | PASS | 2.0924 ms | 1026.32 | 4.39x |

The best GEMM kernel in this benchmark is the padded Tile 32 shared-memory version, reaching about `1026` GFLOP/s and `4.39x` speedup over the naive baseline.

The most important conclusion is:

```text
Tiling creates the computation structure, but shared-memory data reuse is what
actually reduces repeated global-memory traffic. If the shared-memory layout
creates severe bank conflicts, a larger tile can be slower than the naive kernel.
```

### Motivation: Why GEMM Matters

Matrix multiplication is the computational core of many ML and scientific workloads:

- Transformer linear layers.
- Q, K, and V projections.
- Attention score calculation.
- MLP layers.
- Some convolution implementations.
- Scientific computing kernels.
- cuBLAS GEMM.
- Triton matmul.
- Tensor Core workloads.

The mathematical definition is:

```text
C[i, j] = sum_{k=0}^{K-1} A[i, k] * B[k, j]
```

Because GEMM combines high arithmetic work with repeated memory access, it is one of the most representative projects for learning GPU performance optimization.

### Naive GEMM and Repeated Global-Memory Access

In the naive kernel, each thread computes one output element:

```cpp
for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
}
```

For each output element, one thread performs `K` multiply-add steps and reads:

```text
K A elements + K B elements
```

Ignoring cache effects, the full output matrix therefore creates roughly:

```text
2 * M * N * K global loads
```

The key issue is not that every individual memory access is necessarily uncoalesced. For example, neighboring threads often read consecutive `B[k, col]` values, which can be coalesced. The more accurate problem is:

```text
The naive kernel repeatedly loads A and B values from global memory without
explicitly organizing block-level data reuse.
```

The optimization goal is to load A and B tiles from global memory once per block, place them in shared memory, and reuse them across the threads computing the output tile.

### Project Scope: Five Kernels

The notebook implements five kernels:

| Kernel | Purpose |
|---|---|
| Naive MatMul | One thread computes one `C` element and reads operands directly from global memory. |
| Block tiling structure | One block maps to a `16x16` tile of `C`; the `K` loop is split into `16`-wide tiles, but operands are still loaded from global memory. |
| Shared-memory Tile 16 | Each block loads `16x16` A and B tiles into shared memory and reuses them. |
| Shared-memory Tile 32 with bank conflict | Uses `32x32` blocks and transposed B shared-memory layout to intentionally create severe bank conflicts. |
| Shared-memory Tile 32 with padding | Changes the B shared-memory pitch from `32` to `33` to remove the main bank conflict. |

The project also includes:

- CPU GEMM reference.
- Elementwise correctness checking.
- CUDA-event timing.
- Warmup and repeated benchmark iterations.
- GFLOP/s calculation.
- CUDA error checking.

### Matrix Layout and Indexing

All matrices use row-major layout:

```text
A: M x K
B: K x N
C: M x N
```

The one-dimensional addresses are:

```text
A[row][k]   -> A[row * K + k]
B[k][col]   -> B[k * N + col]
C[row][col] -> C[row * N + col]
```

Each CUDA thread maps to one output coordinate:

```cpp
int row = blockIdx.y * TILE + threadIdx.y;
int col = blockIdx.x * TILE + threadIdx.x;
```

Here:

| Index | Meaning |
|---|---|
| `blockIdx.y` | Selects the output tile row. |
| `blockIdx.x` | Selects the output tile column. |
| `threadIdx.y` | Selects the row inside the tile. |
| `threadIdx.x` | Selects the column inside the tile. |

### Concrete Tile Mapping Example

Assume:

```text
TILE = 4
M = N = K = 8
```

Then `C` is divided into four `4x4` tiles:

```text
C tile (0,0) | C tile (0,1)
-------------+-------------
C tile (1,0) | C tile (1,1)
```

For `blockIdx = (1, 0)`, the block computes:

```text
rows    0-3
columns 4-7
```

For `threadIdx = (2, 1)`:

```text
row = 0 * 4 + 1 = 1
col = 1 * 4 + 2 = 6
```

so that thread computes:

```text
C[1][6] = A[1][0]B[0][6] + ... + A[1][7]B[7][6]
```

The `K` dimension is split into two tiles:

```text
K tile 0: k = 0-3
K tile 1: k = 4-7
```

In the first round, the block loads:

```text
A rows 0-3, columns 0-3
B rows 0-3, columns 4-7
```

In the second round, it loads:

```text
A rows 0-3, columns 4-7
B rows 4-7, columns 4-7
```

Each round computes a partial dot product, and the thread accumulates those partial results into the final `C[1][6]`.

The block-level mapping below shows the same idea at the full CUDA launch level: a block owns a tile of the output matrix, while threads inside the block map to elements within that tile.

![MatMul block tiling](../assets/projects/cuda-kernels/matmul-block-tiling.png)

### Kernel 1: Naive Matrix Multiplication

The naive kernel is:

```cpp
__global__ void matmulNaive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * TILE16 + threadIdx.y;
    int col = blockIdx.x * TILE16 + threadIdx.x;

    float acc = 0.0f;

    if (row < M && col < N) {
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = acc;
    }
}
```

Each thread computes one output element. For `M=N=K=1024`, the total floating-point work is:

```text
2 * 1024^3 = 2,147,483,648 FLOPs
```

or about `2.147` GFLOPs of work, counting one multiplication and one addition per inner-loop step.

Naive result:

| Metric | Value |
|---|---:|
| Time | 9.1753 ms |
| GFLOP/s | 234.05 |
| Correctness | PASS |

This version is the baseline for the later optimization stages.

### Kernel 2: Block Tiling with Global Loads

![CUDA shared-memory tiled MatMul dataflow](../assets/projects/cuda-kernels/matmul-tiled-dataflow.png)

![CUDA MatMul naive versus tiled memory traffic and data reuse](../assets/projects/cuda-kernels/matmul-naive-vs-tiled-memory-traffic.png)

The second version changes the loop structure:

```cpp
for (int k_tile = 0; k_tile < K; k_tile += TILE16) {
    for (int k_in_tile = 0; k_in_tile < TILE16; ++k_in_tile) {
        int k = k_tile + k_in_tile;

        if (k < K) {
            acc += A[row * K + k] * B[k * N + col];
        }
    }
}
```

Mathematically, it still computes the same dot product. The main difference is that it:

- Explicitly introduces the block-tile concept.
- Splits the `K` loop into smaller tiles.
- Uses an inner tile loop that can be unrolled.
- Prepares the code structure for shared-memory tiling.

However, this kernel still reads:

```cpp
A[row * K + k]
B[k * N + col]
```

directly from global memory. Therefore, it does not explicitly reduce global-memory traffic. It reorganizes the loop but does not cache A/B tiles in shared memory.

Result:

| Kernel | Time | Speedup vs Naive |
|---|---:|---:|
| Naive | 9.1753 ms | 1.00x |
| Block tiled, still global loads | 4.0592 ms | 2.26x |

The improvement is likely due to compiler and loop-structure effects, such as better unrolling, instruction scheduling, cache behavior, and instruction-level parallelism. The notebook does not include SASS/PTX or Nsight counters proving the exact cause, so the careful interpretation is:

```text
The block-tiled loop structure improved runtime in this experiment, but it did
not explicitly reduce global-memory traffic because operands were still loaded
directly from global memory.
```

### Kernel 3: Shared-Memory Tiled MatMul, Tile 16

The Tile 16 version is the classical shared-memory GEMM optimization:

```cpp
__shared__ float As[16][16];
__shared__ float Bs[16][16];
```

Each block uses:

```text
2 * 16 * 16 * 4 = 2048 bytes = 2 KiB shared memory
```

For each `K` tile, the block performs:

1. Load an A tile into shared memory.
2. Load a B tile into shared memory.
3. Synchronize the block.
4. Compute using shared-memory values.
5. Synchronize again before overwriting the shared-memory tiles.

The figure below shows why the shared-memory version is different from mere block tiling: A and B tiles are loaded once from global memory into on-chip shared memory, then reused across the multiply-accumulate loop.

![MatMul shared memory tiling](../assets/projects/cuda-kernels/matmul-shared-memory.png)

The tile-load pattern is:

```cpp
int aRow = row;
int aCol = k_tile + threadIdx.x;

As[threadIdx.y][threadIdx.x] =
    (aRow < M && aCol < K)
    ? A[aRow * K + aCol]
    : 0.0f;

int bRow = k_tile + threadIdx.y;
int bCol = col;

Bs[threadIdx.y][threadIdx.x] =
    (bRow < K && bCol < N)
    ? B[bRow * N + bCol]
    : 0.0f;

__syncthreads();

for (int kk = 0; kk < TILE16; ++kk) {
    acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
}

__syncthreads();
```

For one `16x16` output tile and one `K` tile of length 16:

| Version | Global reads |
|---|---:|
| Naive/global version | `256 threads * 32 reads = 8192 floats` |
| Shared-memory version | `256 A values + 256 B values = 512 floats` |

The ideal source-code-level reuse factor is:

```text
8192 / 512 = 16x
```

Actual DRAM traffic depends on cache, broadcast behavior, and compiler code generation, but the shared-memory algorithm clearly expresses block-level data reuse.

Tile 16 result:

| Metric | Value |
|---|---:|
| Time | 2.4577 ms |
| GFLOP/s | 873.76 |
| Speedup vs Naive | 3.73x |
| Speedup vs block-tiled global-load version | 1.65x |
| Correctness | PASS |

The timing results are consistent with reduced global-memory traffic through shared-memory reuse.

### Kernel 4: Tile 32 with Bank Conflict

The Tile 32 conflict version uses:

```cpp
dim3 block32(32, 32);
```

That is:

```text
32 * 32 = 1024 threads per block
```

which reaches the common CUDA maximum number of threads per block.

Shared memory:

```cpp
__shared__ float As[32][32];
__shared__ float BsT[32][32];
```

Total shared memory:

```text
2 * 32 * 32 * 4 = 8192 bytes = 8 KiB
```

The kernel stores B in a transposed shared-memory layout:

```cpp
float bval = B[bRow * N + col];
BsT[threadIdx.x][threadIdx.y] = bval;
```

and later computes:

```cpp
acc += As[threadIdx.y][kk] * BsT[threadIdx.x][kk];
```

This still computes the correct mathematical result, but it intentionally creates column-like shared-memory access.

For `BsT`, the row pitch is 32. The shared-memory index is:

```text
index = tx * 32 + kk
```

For a warp where `tx = 0, 1, 2, ..., 31` and `kk` is fixed:

```text
bank = (tx * 32 + kk) mod 32 = kk
```

So the 32 threads access 32 different addresses that all map to the same bank:

```text
32-way bank conflict
```

The transposed store can also conflict:

```cpp
BsT[threadIdx.x][threadIdx.y] = bval;
```

For fixed `ty`, the address is:

```text
tx * 32 + ty
```

which also maps every thread to bank `ty`.

Result:

| Kernel | Time | GFLOP/s | Relative to Naive |
|---|---:|---:|---:|
| Naive | 9.1753 ms | 234.05 | 1.00x |
| Tile 32 with bank conflict | 10.4073 ms | 206.34 | 0.88x |

The conflicted Tile 32 kernel is about 13% slower than the naive kernel. The causes include severe shared-memory bank conflicts, `1024` threads per block reducing block residency flexibility, extra synchronization, and additional transposed-indexing overhead.

The lesson is:

```text
Shared memory is only fast when its access layout is efficient.
```

### Kernel 5: Tile 32 with Padding

The padding version changes:

```cpp
BsT[32][32]
```

to:

```cpp
BsT[32][33]
```

The pitch changes from 32 to 33. The bank mapping becomes:

```text
index = tx * 33 + kk
bank  = (tx * 33 + kk) mod 32
      = (tx + kk) mod 32
```

As `tx` goes from 0 to 31, the accesses are spread across different banks.

The padding cost is small:

| Layout | Shared memory for B tile |
|---|---:|
| `32x32` | 4096 bytes |
| `32x33` | 4224 bytes |
| Extra cost | 128 bytes |

Total shared memory increases from `8192` bytes to `8320` bytes, but the bank mapping changes dramatically.

Padding result:

| Comparison | Result |
|---|---:|
| Tile 32 conflict | 10.4073 ms, 206.34 GFLOP/s |
| Tile 32 padded | 2.0924 ms, 1026.32 GFLOP/s |
| Speedup vs conflict version | 4.97x |
| Speedup vs naive | 4.39x |
| Speedup vs Tile 16 | 1.17x |

The padded Tile 32 kernel is the fastest version in the notebook.

### Benchmarking and Correctness

GFLOP/s is computed as:

```cpp
double flops = 2.0 * M * N * K;
GFLOPS = flops / seconds / 1e9;
```

For `M=N=K=1024`, the total work is:

```text
2 * 1024^3 = 2.147 x 10^9 FLOPs
```

Each kernel is benchmarked with:

- 5 warmup launches.
- 20 timed iterations.
- CUDA events for GPU elapsed time.
- Average kernel latency as the reported result.

This measures kernel execution time only. It does not include host allocation, device allocation, host-to-device copy, device-to-host copy, or CPU reference time.

Correctness is checked against a CPU reference that uses a `double` accumulator:

```cpp
double acc = 0.0;

for (int k = 0; k < K; ++k) {
    acc += static_cast<double>(A[i * K + k]) *
           static_cast<double>(B[k * N + j]);
}

C[i * N + j] = static_cast<float>(acc);
```

The GPU output is compared with:

```cpp
fabsf(gpu[i] - ref[i]) <= 1e-3
```

All five kernels pass.

The notebook also runs Nsight Compute profiling. Profiler runs may print CUDA-event times that are hundreds or thousands of milliseconds because profiling performs replay passes, instrumentation, and repeated metric collection. Those profiler-time prints should not be interpreted as normal benchmark latency. The normal benchmark numbers are:

```text
9.1753 ms
4.0592 ms
2.4577 ms
10.4073 ms
2.0924 ms
```

### Profiler Metrics Snapshot

The interview review notes also record one representative Nsight Compute metrics snapshot for the five GEMM kernels:

| Kernel | Mem GB/s | Max BW | L1 hit | L2 hit |
|---|---:|---:|---:|---:|
| Naive | 22.15 | 62.50% | 87.37% | 82.09% |
| Structural Tile16 | 20.87 | 62.79% | 87.31% | 83.37% |
| Shared Tile16 | 32.41 | 74.48% | 6.21% | 81.67% |
| Conflicted Tile32 | 5.39 | 8.02% | 0.00% | 63.72% |
| Padded Tile32 | 27.16 | 74.53% | 0.00% | 63.54% |

These metrics should be interpreted carefully. The table is useful for comparing the custom kernels under one recorded T4 profiling run, but it does not by itself prove general superiority over cuBLAS, PyTorch, Tensor Core GEMM, or production kernels.

The most important signal is the contrast between the conflicted and padded Tile32 kernels:

```text
Conflicted Tile32: 5.39 GB/s, 8.02% max bandwidth
Padded Tile32:    27.16 GB/s, 74.53% max bandwidth
```

The padded version restores much higher effective memory throughput after removing the severe shared-memory bank-conflict pattern. This supports the timing result where padded Tile32 is about `4.97x` faster than the deliberately conflicted Tile32 kernel.

The L1 hit-rate values are not a simple "higher is always better" score. Shared-memory tiling changes what is loaded through caches, what is reused through shared memory, and how the profiler attributes traffic. For this project, the profiler table is best used together with the source-level explanation of shared-memory reuse and bank-conflict avoidance.

### Boundary and Synchronization Caveat

The shared-memory kernels contain an important code-review issue:

```cpp
if (row >= M || col >= N) return;
```

appears before code that later executes:

```cpp
__syncthreads();
```

If `M` or `N` is not divisible by the tile size, the final block may contain:

- Some threads that return early.
- Other threads that reach `__syncthreads()`.

This is unsafe and can cause undefined behavior or block deadlock.

The current experiment does not expose the issue because `M=N=1024`, and 1024 is divisible by both 16 and 32. A safer pattern is:

```cpp
bool validOutput = row < M && col < N;

As[ty][tx] =
    (row < M && aCol < K)
    ? A[row * K + aCol]
    : 0.0f;

Bs[ty][tx] =
    (bRow < K && col < N)
    ? B[bRow * N + col]
    : 0.0f;

__syncthreads();

if (validOutput) {
    C[row * N + col] = acc;
}
```

The interview-ready rule is:

```text
Never place an early return before a block-wide barrier when only part of a
block may take that return.
```

The naive kernel also uses:

```cpp
int row = blockIdx.y * TILE16 + threadIdx.y;
int col = blockIdx.x * TILE16 + threadIdx.x;
```

which assumes the launch block is always `16x16`. A more general version is:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

For shared-memory tiled kernels, compile-time tile sizes are still reasonable because the shared-memory array shape and loop unrolling depend on the tile size.

### Tile Size 16 vs 32 Trade-off

Tile 16 uses:

```text
16 * 16 = 256 threads per block
```

Advantages:

- Moderate block size.
- More flexible block residency.
- Less shared memory per block.
- Finer boundary granularity.

Costs:

- Tile-level reuse factor is about 16.
- More `K` tiles.
- More block barriers.

For `K=1024`:

```text
1024 / 16 = 64 K tiles
64 * 2 = 128 block barriers
```

Tile 32 uses:

```text
32 * 32 = 1024 threads per block
```

Advantages:

- Tile-level reuse factor is about 32.
- `K` tile count is cut in half.
- Barrier count is cut in half.

For `K=1024`:

```text
1024 / 32 = 32 K tiles
32 * 2 = 64 block barriers
```

Costs:

- Maximum threads per block.
- Lower scheduling flexibility.
- Occupancy may be limited by block size, registers, or shared memory.
- Bad layout can produce severe bank conflicts.
- It may not be ideal for small or irregular matrices.

The padded Tile 32 version is faster than Tile 16 in this specific experiment:

| Kernel | Time |
|---|---:|
| Tile 16 | 2.4577 ms |
| Tile 32 padded | 2.0924 ms |

This is likely due to higher reuse, fewer `K` tile iterations, fewer `__syncthreads()` calls, and removal of the main bank-conflict penalty. However, the result should not be generalized as "Tile 32 is always better." The best tile size depends on thread count, register usage, shared memory usage, occupancy, tensor dimensions, bank conflicts, and GPU architecture.

### Distance from cuBLAS

The fastest notebook kernel reaches:

```text
1026.32 GFLOP/s ~= 1.03 TFLOP/s
```

This is a useful teaching-level shared-memory GEMM, but it is not a cuBLAS-competitive GEMM implementation. It does not implement:

- Register tiling.
- Multiple output elements per thread.
- Vectorized loads.
- Double buffering.
- Asynchronous copy.
- Software pipelining.
- Warp-level tiling.
- Tensor Core MMA.
- WMMA.
- Mixed precision.
- Split-K.
- Epilogue fusion.
- Architecture-specific tuning.

The current implementation follows:

```text
one thread -> one C element
```

More advanced GEMM kernels usually let each thread compute a small output fragment and keep multiple accumulators in registers. The right positioning for this notebook is:

```text
A progression from naive GEMM to classical shared-memory tiling, not a
cuBLAS-competitive GEMM implementation.
```

## Softmax

![CUDA stable Softmax and reduction dataflow](../assets/projects/cuda-kernels/softmax-stable-reduction-dataflow.png)

![CUDA Softmax row mapping strategy comparison across warp-per-row, block-per-row, and multi-warp-per-row](../assets/projects/cuda-kernels/softmax-row-mapping-strategy-comparison.png)

This experiment implements and compares three CUDA row-wise Softmax mapping strategies:

- **Warp-per-row:** one warp handles one row.
- **Block-per-row:** one block handles one row.
- **Multi-warp-per-row:** a configurable number of warps cooperate on one row.

The input is a 2D matrix:

```text
X in R^{B x D}
```

Softmax is applied independently to each row:

```text
y_i = exp(x_i - m) / sum_j exp(x_j - m)
m = max_j x_j
```

The experiment runs on a Tesla T4 and tests different row lengths `D` to study how mapping strategy affects performance.

**In this section:**

- [Why Softmax Subtracts the Maximum](#why-softmax-subtracts-the-maximum)
- [Concrete Softmax Example](#concrete-softmax-example)
- [Warp-Level Reduction for Softmax](#warp-level-reduction-for-softmax)
- [Strategy 1: Warp-per-row](#strategy-1-warp-per-row)
- [Strategy 2: Block-per-row](#strategy-2-block-per-row)
- [Strategy 3: Multi-warp-per-row](#strategy-3-multi-warp-per-row)
- [Softmax Benchmark Results](#softmax-benchmark-results)
- [Main Softmax Mapping Conclusion](#main-softmax-mapping-conclusion)
- [Three-pass Softmax](#three-pass-softmax)
- [Toward a More Efficient Two-pass Softmax](#toward-a-more-efficient-two-pass-softmax)

### Why Softmax Subtracts the Maximum

Softmax is central to attention and must be numerically stable. Directly computing:

```text
exp(x_i)
```

can overflow. For example:

```text
x = [1000, 1001, 1002]
```

`exp(1000)` is already outside the safe floating-point range. The stable implementation subtracts the row maximum first:

```text
m = 1002
x - m = [-2, -1, 0]
```

Then:

```text
exp(-2), exp(-1), exp(0)
```

are all safe to compute, and the Softmax output is unchanged:

```text
exp(x_i - m) / sum_j exp(x_j - m)
= exp(x_i) / sum_j exp(x_j)
```

Therefore the kernel needs three conceptual stages:

```text
Pass 1: find the row maximum
Pass 2: compute the sum of exp(x - max)
Pass 3: compute the normalized output
```

![Softmax definition](../assets/projects/cuda-kernels/softmax-definition.png)

### Concrete Softmax Example

Assume one input row is:

```text
x = [1, 2, 3, 4]
```

First find:

```text
m = 4
```

Then compute:

```text
exp(1 - 4) = exp(-3) ~= 0.0498
exp(2 - 4) = exp(-2) ~= 0.1353
exp(3 - 4) = exp(-1) ~= 0.3679
exp(4 - 4) = exp(0)  = 1
```

The sum is:

```text
s ~= 1.553
```

The final Softmax output is approximately:

```text
[0.0321, 0.0871, 0.2369, 0.6439]
```

In CUDA, this is not computed by one thread serially. Multiple threads process different elements of the row, then use reductions to combine the row maximum and exponential sum.

### Warp-Level Reduction for Softmax

The project implements:

```cpp
warpReduceMax(...)
warpReduceSum(...)
```

using:

```cpp
__shfl_down_sync
```

with offsets:

```text
16, 8, 4, 2, 1
```

This reduces 32 thread-local values inside one warp down to lane 0:

```text
Reduce:
32 local values -> lane 0
```

Then the result is broadcast from lane 0 to all lanes:

```cpp
__shfl_sync(mask, result, 0)
```

```text
Broadcast:
lane 0 result -> all 32 lanes
```

![Warp-level reduction](../assets/projects/cuda-kernels/softmax-warp-reduction.png)

### Strategy 1: Warp-per-row

In the warp-per-row strategy, one warp processes one full row. Each lane handles:

```cpp
for (int j = lane; j < D; j += 32)
```

For example, when `D = 128`:

```text
Lane 0  handles j = 0, 32, 64, 96
Lane 1  handles j = 1, 33, 65, 97
...
Lane 31 handles j = 31, 63, 95, 127
```

The flow is:

1. Each lane computes a local maximum.
2. Warp reduction computes the row maximum.
3. Each lane computes a local exponential sum.
4. Warp reduction computes the row sum.
5. Each lane writes its output elements.

Advantages:

- No shared memory is needed.
- No `__syncthreads()` is needed.
- Reductions use warp shuffle only.
- One block can process multiple rows.

This strategy is best suited for small `D` and large batch/row count `B`.

### Strategy 2: Block-per-row

In the block-per-row strategy, one block processes one row. Each thread visits:

```cpp
for (int j = threadIdx.x; j < D; j += blockDim.x)
```

The block reduction uses two levels:

```text
Each warp performs shuffle reduction
              ↓
Lane 0 of each warp writes to shared memory
              ↓
The first warp combines warp partial results
```

This is better for longer rows, where 32 threads in one warp may not provide enough row-level parallelism.

The costs are:

- Shared memory is required.
- Cross-warp synchronization is required.
- One full block is consumed per row.
- Short rows may waste many threads.

### Strategy 3: Multi-warp-per-row

The multi-warp-per-row version also uses one block per row, but the number of warps per row is configurable:

```text
2 warps  = 64 threads
4 warps  = 128 threads
8 warps  = 256 threads
16 warps = 512 threads
```

The kernel is conceptually similar to block-per-row. The main difference is the chosen `blockDim.x` at launch.

The experiment looks for the balance between:

```text
too few threads  -> each thread processes too many elements
too many threads -> more reduction, synchronization, and resource overhead
```

More warps per row are not always better.

![Softmax implementation strategies](../assets/projects/cuda-kernels/softmax-strategies.png)

### Softmax Benchmark Results

Environment from notebook: Tesla T4. The benchmark reports average launch time and approximate memory bandwidth.

| Shape | Strategy | Time (ms) | Approx. GB/s | Notes |
|---|---|---:|---:|---|
| `B=65536, D=128` | Warp-per-row | 0.2871 | 233.75 | Best for small row width |
| `B=65536, D=128` | Block-per-row | 0.9043 | 74.21 | Slower due to overhead for small rows |
| `B=65536, D=128` | Multiwarp-per-row | 0.7211 | 93.06 | Also slower for small rows |
| `B=16384, D=512` | Warp-per-row | 0.3279 | 204.12 | Slightly best in this run |
| `B=16384, D=512` | Block-per-row | 0.3533 | 189.95 | Competitive |
| `B=16384, D=512` | Multiwarp-per-row | 0.3503 | 191.59 | Competitive |
| `B=16384, D=512` | Multiwarp-per-row, 16 warps | 0.4480 | Lower | Slower from extra overhead |
| `B=4096, D=4096` | Warp-per-row | 1.2903 | 104.02 | Too little cooperation for long rows |
| `B=4096, D=4096` | Block-per-row, 512 threads | 0.9278 | 144.66 | Better for long rows |
| `B=4096, D=4096` | Multiwarp-per-row | 0.7547 | 177.84 | Best for long rows |

For `D = 128`, warp-per-row is fastest:

```text
0.9043 / 0.2871 ~= 3.15x faster than block-per-row
```

The row is short enough that one warp is sufficient, and a full block adds unnecessary reduction and synchronization overhead.

For `D = 512`, warp-per-row remains slightly fastest in this run. This is important because it differs from the simple rule that multi-warp is always best for medium rows. The result shows that medium row lengths require benchmarking.

For `D = 4096`, multi-warp-per-row is fastest:

```text
1.2903 / 0.7547 ~= 1.71x faster than warp-per-row
```

With one warp, each lane processes about:

```text
4096 / 32 = 128 elements
```

With 256 threads, each thread processes about:

```text
4096 / 256 = 16 elements
```

The longer row benefits from higher row-level parallelism.

### Main Softmax Mapping Conclusion

The project shows that Softmax mapping should be selected based on row length:

| Row length | Better strategy |
|---|---|
| Small `D` | Warp-per-row |
| Medium `D` | Benchmark; warp or a small multi-warp configuration may win |
| Large `D` | Multi-warp or block-per-row |

The core trade-off is:

```text
row-level parallelism vs reduction and synchronization overhead
```

There is no single Softmax kernel that is optimal for all shapes.

### Three-pass Softmax

All three kernels implement a 3-pass Softmax pattern over the input row:

```cpp
// Pass 1
local_max = max(local_max, row_x[j]);

// Pass 2
local_sum += expf(row_x[j] - m);

// Pass 3
row_y[j] = expf(row_x[j] - m) / s;
```

Each element is approximately:

- Read from global memory three times.
- Used in `expf` twice.
- Written once to the output.

This creates repeated memory traffic and repeated exponential computation.

### Toward a More Efficient Two-pass Softmax

A more optimized design can compute and cache exponentials during the second pass:

```text
Pass 1:
find maximum

Pass 2:
compute exp(x - max)
store exponentials
reduce sum

Final:
normalize stored exponentials
```

Strictly speaking, the final normalization still needs to read the stored exponentials and write the output, so the code may still have three phases. The difference is:

```text
3-pass version:
third phase rereads x and recomputes exp

cached version:
third phase reads stored exp values and avoids recomputing exp
```

The trade-off is:

```text
less global loading and fewer expf computations
                  vs
more shared-memory/register usage
```

This optimization is most valuable when `D` is small enough that intermediate values can fit in registers or shared memory.

## LayerNorm

![CUDA LayerNorm forward reduction and normalize dataflow](../assets/projects/cuda-kernels/layernorm-forward-reduction-normalize-dataflow.png)

![CUDA LayerNorm backward gradient propagation and reduction paths](../assets/projects/cuda-kernels/layernorm-backward-gradient-reduction-paths.png)

![CUDA LayerNorm backward AtomicAdd versus two-pass reduction](../assets/projects/cuda-kernels/layernorm-backward-atomic-vs-two-pass-reduction.png)

This LayerNorm notebook contains two major backward experiments:

1. A single-kernel backward implementation that directly uses `atomicAdd` to accumulate `dgamma` and `dbeta`.
2. A hierarchical two-pass reduction implementation that avoids heavy atomic contention by computing partial `dgamma` / `dbeta` results first, then reducing those partials.

The default input shape is:

```text
B = 4096
D = 1024
```

That is, 4096 rows and 1024 features per row. The experiment runs on a Tesla T4 with CUDA 12.x and includes CPU references for validating CUDA forward and backward outputs.

**In this section:**

- [LayerNorm Forward Formula](#layernorm-forward-formula)
- [Forward Kernel: One Block per Row](#forward-kernel-one-block-per-row)
- [Backward dx: Two Row-wise Reductions](#backward-dx-two-row-wise-reductions)
- [Backward Version 1: Direct Atomic Accumulation](#backward-version-1-direct-atomic-accumulation)
- [Backward Version 2: Two-pass dgamma/dbeta Reduction](#backward-version-2-two-pass-dgammadbeta-reduction)
- [Direct Atomic vs Two-pass Reduction](#direct-atomic-vs-two-pass-reduction)
- [LayerNorm Benchmark and Validation Results](#layernorm-benchmark-and-validation-results)
- [Correctness Verification](#correctness-verification-1)
- [Main LayerNorm Takeaways](#main-layernorm-takeaways)

### LayerNorm Forward Formula

LayerNorm forward computes per-row statistics:

```text
mu_b = (1 / D) * sum_i x_{b,i}
var_b = (1 / D) * sum_i (x_{b,i} - mu_b)^2
inv_std_b = 1 / sqrt(var_b + eps)
```

Then it normalizes and applies affine parameters:

```text
y_{b,i} = gamma_i * (x_{b,i} - mu_b) * inv_std_b + beta_i
```

Backward computes:

```text
dx
dgamma
dbeta
```

The important systems point is that LayerNorm has reductions along two different dimensions:

- Row-wise reductions for `mean`, `variance`, and `dx`.
- Batch-wise reductions across rows for `dgamma` and `dbeta`.

### Forward Kernel: One Block per Row

The forward kernel uses:

```text
one block per row
blockIdx.x = row index
```

Threads inside the block read the row with a block-stride loop:

```cpp
for (int i = tid; i < D; i += blockDim.x) {
    ...
}
```

In each loop iteration, neighboring threads access neighboring feature positions, so the global-memory access pattern is coalesced.

The forward reduction flow is:

```text
thread-local sum and squared sum
        ↓
warp-level reduction with __shfl_down_sync
        ↓
lane 0 of each warp writes partials to shared memory
        ↓
__syncthreads()
        ↓
warp 0 reduces warp partials
        ↓
thread 0 computes mean, variance, and inv_std
        ↓
mean / inv_std are broadcast through shared memory
        ↓
all threads normalize, apply gamma/beta, and write output
```

This experiment practices the standard block-reduction pattern:

```text
thread-local accumulation
-> warp shuffle reduction
-> shared-memory block reduction
-> broadcast
-> normalization
```

The forward implementation is not Welford. It reduces:

```text
sum(x)
sum(x^2)
```

and computes:

```cpp
variance = sum_x2 / D - mean * mean;
```

This is simple and fast, but less numerically stable than Welford for difficult inputs.

### Backward dx: Two Row-wise Reductions

LayerNorm backward defines:

```text
xhat_i = (x_i - mean) * inv_std
g_i = dout_i * gamma_i
```

Each row needs two reductions:

```text
S1 = sum_i g_i
S2 = sum_i g_i * xhat_i
```

Then each element computes:

```text
dx_i = inv_std * (g_i - S1 / D - xhat_i * S2 / D)
```

The CUDA kernel reduces `S1` and `S2` together:

```cpp
s1 = warpReduceSum(s1);
s2 = warpReduceSum(s2);
```

and then merges warp partials through shared memory before each thread finalizes its own `dx_i`.

This is a useful interview point:

```text
LayerNorm backward is reduction-heavy. Computing dx requires two row-wise
reductions before each element can be finalized.
```

### Backward Version 1: Direct Atomic Accumulation

The first backward version directly accumulates `dgamma` and `dbeta` inside the same backward kernel:

```cpp
atomicAdd(&dbeta[i], dyi);
atomicAdd(&dgamma[i], dyi * xhat);
```

The math is:

```text
dbeta_i  = sum_b dout_{b,i}
dgamma_i = sum_b dout_{b,i} * xhat_{b,i}
```

For each feature `i`, all 4096 rows update the same `dbeta[i]` and `dgamma[i]` addresses. With `B=4096` and `D=1024`, this creates roughly:

```text
4096 * 1024 * 2 = 8.39 million atomic additions
```

More importantly, those atomics contend for only 1024 `dgamma` addresses and 1024 `dbeta` addresses.

The profiled result at the default scale is:

| Kernel | Time |
|---|---:|
| Forward | 0.2128 ms |
| Backward with direct atomics | 13.0604 ms |

The backward is about:

```text
13.0604 / 0.2128 ~= 61.4x slower than forward
```

This does not mean that the LayerNorm backward formula is intrinsically 61x more expensive than forward. The dominant issue is global-memory atomic contention on `dgamma` and `dbeta`.

The key lesson is:

```text
A mathematically correct GPU implementation may still perform extremely poorly
if many thread blocks atomically update the same global-memory locations.
```

### Backward Version 2: Two-pass dgamma/dbeta Reduction

The second backward design splits the work into three kernels:

| Kernel | Purpose |
|---|---|
| Kernel 1 | Compute `dx` with one block per row. |
| Kernel 2 | Compute partial `dgamma` and `dbeta` over batch tiles. |
| Kernel 3 | Reduce partial results into final `dgamma` and `dbeta`. |

Kernel 1 computes `S1`, `S2`, and `dx` per row. It does not require cross-batch reduction and does not need atomics.

Kernel 2 tiles the batch and feature dimensions. The notebook uses:

```cpp
TILE_D = 256;
TILE_N = 128;
```

For:

```text
B = 4096
```

the number of partial groups is:

```text
P = ceil(4096 / 128) = 32
```

Instead of 4096 rows atomically updating one final value, every 128 rows first produce partial outputs:

```text
dgamma_partial[p][i]
dbeta_partial[p][i]
```

There are 32 groups of partial results. Kernel 3 then reduces, for each feature:

```text
32 partial values -> final dgamma[i], dbeta[i]
```

The direct atomic reduction is replaced by:

```text
parallel local accumulation
-> partial global results
-> final reduction
```

This is the standard hierarchical reduction idea applied to LayerNorm parameter gradients.

### Direct Atomic vs Two-pass Reduction

| Design | Advantage | Problem |
|---|---|---|
| Direct atomic | Simple implementation; one backward kernel. | Severe contention when many rows update the same small set of feature-gradient addresses. |
| Two-pass reduction | Greatly reduces contention and scales better with larger batch size. | Requires multiple kernels, an additional partial buffer, and extra global-memory passes. |

The lesson is not "atomics are always bad." A direct atomic approach can be acceptable when contention is low, such as very small batch sizes. But when thousands of blocks update the same small output vector, atomics become a major bottleneck.

### LayerNorm Benchmark and Validation Results

Benchmark setting from notebook: Tesla T4, `B=4096`, `D=1024`.

| LayerNorm kernel | Correctness | Time | Notes |
|---|---|---:|---|
| Forward with direct-atomic backward experiment | PASS | 0.2128 ms | One block per row, row-wise reductions. |
| Backward with direct atomic `dgamma/dbeta` | Profiled | 13.0604 ms | Dominated by atomic contention. |
| Warp-based forward | PASS | 0.2119 ms | One optimized forward path. |
| Warp-based backward | PASS | 0.2916 ms | Backward gradients pass correctness checks in the recorded run. |
| Two-pass forward | PASS | 0.2118 ms | Similar forward speed. |
| Two-pass backward total | FAIL in notebook | 0.6136 ms | Shows the complexity of implementing correct multi-kernel backward reductions. |
| Mixed precision, FP32 reduction | Accuracy checked | 0.1811 ms | Faster and more accurate than low-precision reduction. |
| Pure low-precision reduction | Accuracy checked | 0.2149 ms | Higher error and slower in this run. |

The direct-atomic result explains why a single-kernel implementation can be dramatically slower. The later optimized/variant results show that faster backward paths are possible, but also that multi-kernel backward reductions are easier to get wrong and require careful validation.

The mixed-precision experiments show why numerical precision matters. FP32 reductions with FP16 IO gave lower error and faster runtime than pure low-precision reduction in the tested configuration.

### Correctness Verification

The CPU reference is compared against CUDA outputs for:

- Forward output `y`.
- Row `mean`.
- Row `inv_std`.
- Backward `dx`.
- `dgamma`.
- `dbeta`.

Typical tolerances:

| Quantity | Tolerance |
|---|---:|
| Forward `y` | `1e-3` |
| `mean` / `inv_std` | `1e-4` |
| Backward outputs | `1e-2` |

The two-pass version executes:

```text
ln_forward_warp
ln_backward_dx_warp
ln_dg_db_partials
ln_reduce_partials
```

and then compares results with the CPU reference.

### Main LayerNorm Takeaways

| Takeaway | Explanation |
|---|---|
| Forward maps naturally to one block per row | Each row's mean and variance are independent, so reductions can stay inside a block. |
| Warp shuffle plus shared memory is the standard block-reduction pattern | Warp-local reductions use `__shfl_down_sync`; cross-warp partials go through shared memory and `__syncthreads()`. |
| Backward is more complex than forward | `dx` requires row-wise reductions, while `dgamma/dbeta` require reductions across the batch dimension. |
| Direct atomics can be catastrophically slow | At `B=4096, D=1024`, millions of atomic updates contend for only 1024 feature-gradient addresses. |
| Hierarchical reduction is the scalable pattern | Partial `dgamma/dbeta` buffers reduce contention, then a final kernel reduces partials. |
| Numerical precision matters | FP32 reductions with FP16 IO can improve both accuracy and runtime compared with lower-precision reductions. |

## PyTorch C++/CUDA Extensions

The project also integrates CUDA kernels with PyTorch through C++/CUDA extensions:

- TensorAccessor-based extension loading.
- Forward-only extension path.
- Forward/backward LayerNorm extension.
- Autograd wrapper validation.
- Fused Bias+GELU extension.

Selected validation results:

| Extension | Test | Result |
|---|---|---|
| TensorAccessor extension | `B=256, D=1024` | `allclose=True`, max absolute error `0.0` |
| TensorAccessor benchmark | `B=4096, D=1024`, 200 iterations | `0.156652 ms/iter` |
| LayerNorm extension gradients | `dx`, `dgamma`, `dbeta` | allclose true, max abs errors around `4.77e-07` to `9.54e-07` |
| Fused Bias+GELU, FP32 | `B=256, D=1024` | max error `4.73e-04` |
| Fused Bias+GELU, FP16 | `B=256, D=1024` | max error `3.91e-03` |

## Naive Attention and IO-Bound Analysis

The attention notes explain the full attention pipeline:

```text
Q = X Wq
K = X Wk
V = X Wv
Scores = Q K^T
P = softmax(Scores)
O = P V
```

![Attention QKV construction](../assets/projects/cuda-kernels/attention-qkv.png)

![Attention output computation](../assets/projects/cuda-kernels/attention-output.png)

For a naive attention implementation, the project uses a block-per-row tiling strategy where each program/block computes one row or block of rows of the attention score matrix.

![Attention tiling](../assets/projects/cuda-kernels/attention-tiling.png)

The notes derive why naive attention can be IO-bound. For `n=2048`, `d=64`, FP16:

- Attention score computation is roughly `4 * n^2 * d`, around `1.07e9` FLOPs.
- Materializing score/probability matrices creates roughly `4 * n^2` FP16 elements of S/P traffic, about 33.5 MB just for those intermediate matrices.
- Arithmetic intensity is about 32 FLOPs/byte.
- On a T4-style roofline with FP16 peak around 65 TFLOP/s and memory bandwidth around 320 GB/s, the ridge point is about 203 FLOPs/byte.

Since `32 << 203`, naive attention lies on the memory-bound side of the roofline.

![Attention roofline analysis](../assets/projects/cuda-kernels/attention-roofline.png)

## Profiling Focus

![CUDA Nsight Compute profiling workflow from runtime symptom to root cause](../assets/projects/cuda-kernels/nsight-compute-profiling-workflow.png)

![CUDA MatMul Nsight root-cause analysis for Tile16, Tile32, and Tile32 with padding](../assets/projects/cuda-kernels/nsight-matmul-root-cause-analysis.png)

The project uses Nsight Compute concepts to connect code changes with GPU behavior:

- Memory workload analysis.
- Source counters.
- Warp state statistics.
- Scheduler statistics.
- DRAM throughput.
- SM utilization.
- Memory dependency stalls.

For IO-bound kernels, the expected profiling signature is high DRAM activity, lower achieved FLOPs relative to peak, and stalls related to memory dependencies.

## Main Takeaways

- Shared memory tiling gives a large MatMul speedup only when access patterns avoid bank conflicts.
- Padding can turn a slow Tile32 shared-memory kernel into the best MatMul variant.
- Loop unrolling and register reuse provide moderate gains by reducing loop overhead and improving instruction scheduling.
- Softmax strategy depends on row width: warp-per-row is strong for small rows, while multiwarp-per-row is better for long rows.
- LayerNorm performance depends on both reduction strategy and numerical precision.
- PyTorch C++/CUDA extensions require both kernel correctness and integration correctness through autograd checks.
- Naive attention is memory-bound because it materializes large score/probability matrices with low arithmetic intensity relative to GPU roofline.

## Experiment Result Analysis

The MatMul results show the clearest optimization ladder. Moving from naive global-memory access to block tiling gives a 2.26x speedup. Adding shared-memory tiling with `TILE=16` increases speedup to 3.73x. However, `TILE=32` without padding becomes slower than the naive baseline because bank conflicts and unfavorable memory access patterns outweigh the theoretical reuse benefit. Once padding is added, the Tile32 kernel becomes the best result at 4.38x speedup.

The Softmax results show that there is no single best mapping for every shape. Small rows benefit from warp-per-row because a single warp can reduce the row efficiently with low coordination overhead. Large rows benefit from multiwarp-per-row because one warp is not enough parallelism for the row width, and multiple cooperating warps improve memory throughput.

LayerNorm demonstrates the interaction between performance and numerical correctness. A fast kernel is not useful unless the backward pass and reductions are correct. The mixed-precision experiment is especially important: FP32 reductions with FP16 IO achieved both better accuracy and better runtime than pure low-precision reduction in this run.

The attention analysis connects the operator-level work to a roofline model. Naive attention performs substantial compute, but it also materializes large intermediate score and probability matrices. The arithmetic-intensity estimate places it on the memory-bound side of the T4 roofline, explaining why IO-aware attention kernels such as FlashAttention avoid materializing the full attention matrix and instead tile Q/K/V through faster memory.

Overall, this project shows how CUDA optimization is not one trick. Performance comes from matching the kernel strategy to the operator shape, memory hierarchy, numerical constraints, and profiler evidence.

[Back to Home](../index.md)
