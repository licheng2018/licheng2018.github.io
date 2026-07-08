# PyTorch Compiler and AI Compiler Pipeline Project

This project studies how modern ML compiler stacks turn PyTorch programs into optimized GPU execution, and why compilation does not always make a workload faster. I built a compiler/runtime benchmarking project and a toy AI compiler pipeline to connect frontend graph transformations with backend GPU performance.

The project also includes a compiler explainer that walks from eager-mode execution to graph capture, TorchDynamo, FX graphs, graph breaks, guards, caching, TorchInductor, and Triton/CUDA code generation.

[Download the original project explainer PDF](../assets/projects/compiler.pdf)

## Project Goal

The goal was to understand the full path from Python-level PyTorch code to generated GPU kernels:

- How one line of PyTorch can become several GPU kernels in eager mode.
- How `torch.compile` captures and transforms model execution.
- How TorchDynamo, FX, AOTAutograd, and TorchInductor interact.
- When compiler optimizations such as fusion reduce latency and memory traffic.
- Why graph breaks, dynamic shapes, Python control flow, or small workloads can erase the expected speedup.
- How a simplified compiler pipeline can lower graph patterns into Triton kernels.

## Motivation: One Line of PyTorch, Many GPU Kernels

At the Python level, a statement can look like one simple expression:

```python
y = torch.relu(x @ w + b)
```

In eager mode, PyTorch may execute this as multiple operations:

1. A matrix multiplication kernel.
2. An add kernel.
3. A ReLU kernel.

Each step may read from and write back to GPU global memory. This is flexible and easy to debug, but it can create extra kernel launches, intermediate tensors, and repeated high-bandwidth memory traffic.

## What Happens in Eager Mode

Eager mode executes immediately and optimizes locally. For:

```python
y = torch.relu(x @ w + b)
```

execution may look like:

1. **Matmul**
   - Read `x` and `w`.
   - Compute `x @ w`.
   - Write an intermediate result to HBM.
2. **Add**
   - Read the intermediate result from HBM.
   - Read `b`.
   - Compute `x @ w + b`.
   - Write another intermediate result to HBM.
3. **ReLU**
   - Read the intermediate result from HBM.
   - Compute `relu`.
   - Write the final output to HBM.

The computation is correct, but there can be too much HBM reading and writing.

## Why This Is a Problem

The GPU is fast, but memory traffic is expensive. Eager mode can introduce several sources of overhead:

- Kernel launch overhead.
- Python dispatch overhead.
- Intermediate tensor materialization.
- Repeated HBM reads and writes.
- Limited cross-operator optimization.

For small operations, launch overhead can dominate. For large tensors, HBM traffic can dominate.

## The Bridge to Graph Capture

Instead of executing each operation immediately:

```text
add kernel
mul kernel
sigmoid kernel
relu kernel
add kernel
```

the compiler wants to first capture the whole computation:

```text
x
-> add
-> mul
-> sigmoid
-> add with relu(x)
-> output
```

Once the computation is visible as a graph, the compiler can attempt to turn several operations into one fused kernel.

## Graph Capture

Example Python code:

```python
def f(x):
    y = x + 1
    y = y * 2
    y = torch.relu(y)
    return y
```

Captured graph:

```text
x -> add -> multiply -> relu -> output
```

Once the compiler has a graph, it can apply:

- Operator fusion.
- Memory planning.
- Decomposition.
- Lowering.
- Code generation.

## PyTorch Compiler Pipeline

The project studies the PyTorch 2.x compiler path:

```text
PyTorch eager code
-> TorchDynamo
-> FX Graph
-> AOTAutograd
-> TorchInductor
-> Triton / CUDA / external kernels
-> optimized execution
```

Each component has a distinct role:

- **TorchDynamo** captures Python/PyTorch code into a graph.
- **FX Graph** represents the captured computation.
- **AOTAutograd** handles forward and backward graphs for training.
- **TorchInductor** optimizes and lowers the graph.
- **Triton/CUDA/external kernels** execute on hardware.

## TorchDynamo as the Frontend

TorchDynamo is the frontend of `torch.compile`. It intercepts Python execution and extracts PyTorch tensor operations into FX graphs.

It works at the Python frame and bytecode level, which allows normal PyTorch code to stay mostly unchanged:

```python
compiled_model = torch.compile(model)
```

## TorchDynamo Is Not Ordinary Tracing

Ordinary tracing usually runs the function once and records what happened:

```python
def f(x, flag):
    if flag:
        return torch.relu(x)
    else:
        return torch.sigmoid(x)
```

A normal trace may only record the branch taken by the example input. TorchDynamo works at the Python frame and bytecode level, so it can inspect Python execution more carefully, insert guards, and handle graph breaks.

## The Compiler Wants a Continuous Graph

For a simple tensor-only function:

```python
def f(x):
    a = x + 1
    b = a * 2
    c = torch.relu(b)
    return c
```

TorchDynamo can capture the whole function as one graph:

```text
x -> add -> multiply -> relu -> output
```

This is good for the compiler because the whole computation is visible. Once the full region is captured, TorchInductor can fuse operations, reduce memory traffic, and generate optimized code.

## Graph Breaks

A graph break happens when TorchDynamo cannot safely continue capturing Python code into the same FX graph.

Instead of one continuous graph:

```text
x -> add -> multiply -> relu -> output
```

Dynamo may split execution into:

```text
Graph 1: x -> add
Python eager execution
Graph 2: multiply -> relu -> output
```

The program can still run correctly, but the compiler no longer sees the whole computation at once.

## Example: `.item()` Breaks the Tensor World

Consider this function:

```python
def f(x):
    if x.sum().item() > 0:
        return x * 2
    else:
        return x - 2
```

`x.sum()` is a tensor operation, but `.item()` converts the tensor value into a Python scalar. Then the Python `if` statement depends on runtime tensor data. This is difficult to capture as a static graph.

When code contains:

```python
if x.sum().item() > 0:
```

three things happen:

- A GPU tensor value must be converted into a Python scalar.
- Python control flow depends on that scalar.
- The graph structure may depend on runtime data.

This may also force synchronization:

```text
GPU computes x.sum()
-> CPU/Python waits for the value
-> Python decides which branch to run
```

That breaks the compiler's ability to optimize the whole computation as one graph.

## Avoiding This Graph Break

Instead of using Python control flow based on `.item()`:

```python
def f(x):
    if x.sum().item() > 0:
        return x * 2
    else:
        return x - 2
```

the computation can be expressed with tensor operations:

```python
def f(x):
    cond = x.sum() > 0
    return torch.where(cond, x * 2, x - 2)
```

This can be represented as a graph:

```text
x -> sum -> compare
x -> multiply by 2
x -> subtract 2
-> torch.where
-> output
```

## Other Common Causes of Graph Breaks

Graph breaks are often caused by Python behavior that cannot be safely represented inside an FX graph:

- Python I/O or logging inside the compiled region, such as `print(x.shape)`.
- Python side effects and mutation, such as `cache.append(x)`.
- Loop counts that depend on runtime tensor data:

```python
n = int(x[0].item())
for _ in range(n):
    x = x + 1
```

- External libraries that Dynamo cannot capture safely:

```python
some_external_library(x)
```

## Performance Impact of Graph Breaks

Graph breaks can hurt performance because they reduce the compiler's optimization scope.

Without a graph break:

```text
add -> multiply -> sigmoid -> relu -> add
```

TorchInductor may fuse the whole chain into one kernel.

With a graph break:

```text
Graph 1: add -> multiply
Python eager region
Graph 2: sigmoid -> relu -> add
```

Consequences include:

- Less operator fusion.
- More intermediate tensor materialization.
- More kernel launches.
- More Python dispatch overhead.
- Possible CPU-GPU synchronization.
- More fragmented execution.

## Guards

When Dynamo compiles a graph, it makes assumptions about the inputs and Python state:

- Input shape.
- Input dtype.
- Input device.
- Tensor layout and stride.
- Module attributes.
- Python object state.

Example guard assumptions:

```text
x.shape == [32, 768]
x.dtype == float16
x.device == cuda
x is contiguous
```

Guards check whether these assumptions still hold on the next call.

## Cache and Recompilation

Compilation is expensive, so Dynamo and the compiler stack try to avoid compiling the same graph again.

On the first call:

```text
capture graph
-> compile graph
-> generate code
-> store compiled version in cache
-> execute
```

On a later call:

```text
check guards
-> guards pass
-> reuse cached compiled version
-> execute
```

If guards fail:

```text
recompile
-> create a new cached version
```

This is why dynamic shapes, changing layouts, or changing Python object state can trigger recompilation and reduce the practical benefit of compilation.

## TorchInductor

TorchDynamo answers:

```text
Can we capture this Python/PyTorch program as a graph?
```

TorchInductor answers:

```text
How do we make this graph run faster?
```

TorchInductor performs:

- Fusion.
- Decomposition.
- Memory planning.
- Lowering.
- Scheduling.
- Code generation.

On GPU, TorchInductor often generates Triton kernels or calls external libraries such as cuBLAS.

## Benchmark Workloads

I evaluated representative ML and attention workloads where compiler behavior can be inspected clearly:

- MLP blocks.
- LayerNorm.
- Elementwise operator chains.
- Naive attention.
- SDPA attention.
- Fused elementwise patterns.
- MLP-style kernels such as Linear + ReLU.

For each workload, I compared eager execution and compiled execution across cold-start and steady-state settings.

## What I Measured

- Cold-start compile time.
- Steady-state latency.
- Compilation overhead.
- Kernel launch count.
- Operator fusion opportunities.
- Optimized library dispatch behavior.
- Memory traffic and intermediate tensor materialization.
- Small-batch overhead and cases where eager execution remains competitive.

## TorchInductor Code and Runtime Analysis

I inspected TorchInductor-generated code and profiling results to connect compiler decisions with runtime behavior. The analysis focused on:

- Which operators were fused into generated kernels.
- Which operations still dispatched to existing optimized libraries.
- How many GPU kernels were launched before and after compilation.
- Whether intermediate tensors were materialized in global memory.
- How generated Triton/CUDA kernels affected memory movement and launch overhead.

This made the compiler pipeline more concrete: graph-level transformations could be tied directly to GPU runtime effects such as kernel launches, memory traffic, and fusion benefits.

## Toy AI Compiler Pipeline

To understand the compiler stack from first principles, I also built a simplified AI compiler pipeline with:

- Frontend IR construction.
- Graph-level pattern matching.
- Operator fusion passes.
- Lowering from graph representation to backend code.
- Triton code generation for fused elementwise and MLP-style kernels.

Example fusion targets included elementwise chains and Linear + ReLU style patterns. The toy pipeline demonstrated how a compiler can identify repeated graph structures, fuse them, and generate a lower-level GPU implementation.


## Experiment Results

The complete experiments come from two notebooks:

- [torch_compile_colab_skeleton.ipynb](https://github.com/licheng2018/AI-compiler/blob/main/torch_compile_colab_skeleton.ipynb)
- [Mini_PyTorch_to_Triton_Compiler.ipynb](https://github.com/licheng2018/AI-compiler/blob/main/Mini_PyTorch_to_Triton_Compiler.ipynb)

Both notebooks were run on a Tesla T4 GPU with CUDA 12.8 and PyTorch 2.11.0+cu128. The mini compiler notebook used Triton 3.6.0.

### MLP: Eager vs. `torch.compile`

Workload: MLP with `D=768`, `DFF=3072`, `float16`.

| Batch | Eager first run (ms) | Eager steady avg (ms) | Compile first run (ms) | Compile steady avg (ms) | Steady speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 321.220 | 0.106 | 8828.449 | 0.208 | 0.51x |
| 8 | 14.804 | 0.123 | 987.963 | 0.243 | 0.50x |
| 32 | 2.677 | 0.139 | 0.439 | 0.241 | 0.57x |
| 128 | 41.330 | 0.221 | 0.442 | 0.287 | 0.77x |
| 512 | 0.575 | 0.462 | 0.797 | 0.523 | 0.88x |

Dynamo inspection for the MLP produced one graph with zero graph breaks and three captured ops: `linear`, `gelu`, and `linear`. Even with clean graph capture, compiled execution was slower in steady state for this GEMM-heavy workload because eager PyTorch already dispatches to highly optimized library kernels.

### Elementwise Fusion

Workload: elementwise chain in `float16`.

| N | Eager first run (ms) | Eager steady avg (ms) | Compile first run (ms) | Compile steady avg (ms) | Steady speedup |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 108.165 | 0.075 | 385.398 | 0.088 | 0.85x |
| 1,048,576 | 0.170 | 0.099 | 395.395 | 0.105 | 0.94x |
| 16,777,216 | 1.741 | 1.528 | 0.621 | 0.460 | 3.32x |

The large elementwise case shows the clearest compiler win: at `N=16,777,216`, compiled execution reached a 3.32x steady-state speedup. This matches the expected benefit of fusion: fewer kernel launches and less repeated global memory traffic.

### Attention: Naive Attention, Compiled Attention, and SDPA

Workload: `B=1`, `H=8`, `D=64`, `float16`.

| Sequence N | Naive eager avg (ms) | Naive compile avg (ms) | Naive compile speedup | SDPA eager avg (ms) | SDPA compile avg (ms) | Best steady path |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 0.172 | 0.125 | 1.38x | 0.037 | 0.124 | SDPA eager |
| 256 | 0.152 | 0.150 | 1.01x | 0.043 | 0.138 | SDPA eager |
| 512 | 0.172 | 0.161 | 1.07x | 0.075 | 0.151 | SDPA eager |
| 1024 | 0.585 | 0.300 | 1.95x | 0.218 | 0.296 | SDPA eager |

Compiling naive attention helped most at longer sequence length, especially `N=1024`, where the naive compiled path was nearly 2x faster than naive eager. However, PyTorch SDPA eager remained the fastest steady-state path across all tested sequence lengths, showing that calling the right optimized primitive can beat compiling a naive implementation.

### Graph Break and Full-Graph Debugging

| Function / case | Graph count | Graph break count | Op count | Result |
|---|---:|---:|---:|---|
| Tensor-only function: add -> ReLU -> multiply | 1 | 0 | 3 | Captured as one continuous graph. |
| `.item()` data-dependent branch | 2 | 1 | 1 | Broke graph capture because `Tensor.item()` converted tensor data into Python control flow. |
| `torch.where` rewrite | 1 | 0 | 4 | Fixed the graph break by keeping the branch as tensor computation. |
| `fullgraph=True` with `.item()` branch | Failed | Failed | N/A | Compilation failed because Dynamo could not guard on a data-dependent expression. |
| `fullgraph=True` with tensor-friendly rewrite | 1 | 0 | N/A | Compiled successfully. |

The graph-break experiments show that correctness and compilability are different. A function can run correctly in eager mode while still being a poor candidate for compilation because Python control flow fragments the graph.

### Dynamic Shape and Recompilation

Batch-size variation for shape `(B, 128, 256)`:

| Compile mode | Shape sequence | Observed elapsed times (ms) | Interpretation |
|---|---|---|---|
| Static compile | `(1,128,256)`, `(2,128,256)`, `(4,128,256)`, `(8,128,256)`, `(1,128,256)` | 569.467, 534.745, 543.342, 530.246, 0.394 | New shapes triggered expensive compilation/recompilation; the repeated shape reused cache. |
| Dynamic compile | `(1,128,256)`, `(2,128,256)`, `(4,128,256)`, `(8,128,256)`, `(1,128,256)` | 902.859, 777.453, 0.468, 0.460, 0.335 | Higher initial compilation cost, then broad reuse across later batch sizes. |

Sequence-length variation for shape `(1, S, 256)`:

| Compile mode | Shape sequence | Observed elapsed times (ms) | Interpretation |
|---|---|---|---|
| Static compile | `(1,64,256)`, `(1,128,256)`, `(1,256,256)`, `(1,512,256)`, `(1,128,256)` | 576.871, 0.540, 640.394, 10.249, 0.402 | Shape changes triggered recompilation; Dynamo also hit the recompile limit due to sequence-length mismatch. |
| Dynamic compile | `(1,64,256)`, `(1,128,256)`, `(1,256,256)`, `(1,512,256)`, `(1,128,256)` | 0.421, 0.464, 0.376, 0.308, 0.369 | Dynamic compile reused the compiled path across sequence lengths after setup. |

### Backend Comparison

| Backend | First run (ms) | Steady avg (ms) | Steady p50 (ms) | Steady p95 (ms) |
|---|---:|---:|---:|---:|
| Raw eager | 99.820 | 0.184 | 0.180 | 0.262 |
| `torch.compile`, backend=`eager` | 68.763 | 0.185 | 0.177 | 0.223 |
| `torch.compile`, backend=`aot_eager` | 83.465 | 0.284 | 0.251 | 0.354 |
| `torch.compile`, backend=`inductor` | 935.247 | 0.212 | 0.206 | 0.260 |

This comparison separates graph-capture overhead from backend code-generation benefit. In this test, Inductor had the largest first-run cost and did not beat raw eager in steady-state latency.

### Mini PyTorch-to-Triton Compiler

The mini compiler notebook implemented a small end-to-end compiler flow:

```text
PyTorch function
-> FX graph capture
-> custom mini IR
-> add + ReLU pattern fusion
-> Triton kernel lowering
-> runtime wrapper
-> correctness and performance benchmark
```

FX graph for the frontend function:

```text
x, bias -> operator.add -> torch.relu -> output
```

Mini IR before fusion:

```text
placeholder(x)
placeholder(bias)
add(x, bias)
relu(add)
output(relu)
```

Mini IR after fusion:

```text
placeholder(x)
placeholder(bias)
fused_add_relu(x, bias)
output(relu)
```

Correctness check: maximum difference was `0.0`.

| N | Eager first run (ms) | Eager steady avg (ms) | Triton first run (ms) | Triton steady avg (ms) | Speedup over eager |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.101 | 0.039 | 0.101 | 0.049 | 0.78x |
| 1,048,576 | 0.103 | 0.103 | 0.139 | 0.089 | 1.16x |
| 16,777,216 | 1.409 | 1.402 | 0.932 | 0.861 | 1.63x |

For reference, `torch.compile` with Inductor on the same `N=16,777,216` add+ReLU workload reported:

| Method | First run (ms) | Steady avg (ms) | Steady p50 (ms) | Steady p95 (ms) |
|---|---:|---:|---:|---:|
| `torch.compile` Inductor | 3110.247 | 0.195 | 0.190 | 0.213 |

## Main Takeaways

- One line of PyTorch can become multiple GPU kernels in eager mode.
- Eager execution is flexible, but repeated HBM traffic and kernel launches can be expensive.
- Graph capture gives the compiler visibility across operations.
- Operator fusion can reduce intermediate tensor materialization, global memory movement, and kernel launch overhead.
- Compilation benefits are workload-dependent; graph capture alone does not guarantee faster execution.
- Graph breaks, dynamic-shape guards, Python-side control flow, and `.item()` can prevent useful fusion or trigger recompilation.
- Guards and caching are essential for reusing compiled graphs, but guard failures can cause recompilation.
- Inspecting generated code and profiler traces is essential for explaining compiler behavior, not just measuring latency.

## Skills Demonstrated

- ML compiler/runtime analysis.
- PyTorch 2.x compiler workflow.
- TorchDynamo and FX graph inspection.
- AOTAutograd and forward/backward graph reasoning.
- TorchInductor generated-code analysis.
- Triton code generation.
- GPU profiling and performance debugging.
- Benchmark design for cold-start and steady-state latency.
- Operator fusion and memory-traffic reasoning.
- Explaining compiler behavior from both frontend graph capture and backend GPU execution.

## Experiment Result Analysis

The results show that `torch.compile` is not a universal speed button. For GEMM-heavy MLP workloads, eager PyTorch already reaches optimized cuBLAS-backed paths, so compilation added overhead and produced steady-state slowdowns from 0.50x to 0.88x. Clean graph capture alone was not enough; the workload also needed optimization opportunities that Inductor could exploit beyond existing library dispatch.

Elementwise workloads behaved differently. Small and medium elementwise cases did not benefit much, but the large `N=16,777,216` case achieved a 3.32x speedup. This is the clearest evidence for fusion: when many simple operations would otherwise materialize intermediate tensors and repeatedly touch HBM, compiled fusion can reduce memory traffic and kernel launch overhead.

Attention results showed a more nuanced story. Compiling naive attention improved the long-sequence case, reaching about 1.95x speedup at `N=1024`, but SDPA eager was still faster than both naive eager and naive compiled paths. This suggests the best systems decision is not always to compile arbitrary Python code; often it is better to route the workload to a specialized primitive that already uses an optimized backend.

The graph-break experiments explain why compiler-friendly code matters. `.item()` moves tensor data into Python control flow, causing a graph break and making `fullgraph=True` fail. Rewriting the branch with `torch.where` kept the computation in tensor form and restored graph capture. This is important for production ML systems because small Python-side decisions can determine whether the compiler sees one optimizable region or several fragmented regions.

Dynamic-shape experiments show the trade-off between specialization and reuse. Static compilation produced expensive recompiles when shape assumptions changed, while dynamic compilation paid higher initial cost but reused compiled paths across later shape changes. For serving systems with variable batch size or sequence length, dynamic-shape behavior and guard failures can dominate real-world latency.

The mini compiler results validate the compiler pipeline at a smaller scale. The custom FX-to-IR-to-Triton path produced numerically correct output and achieved speedups on large tensors, but it was slower for small tensors. This mirrors the broader lesson from `torch.compile`: compiler-generated kernels pay off when the workload is large enough and memory traffic or fusion opportunities dominate overhead.

[Back to Home](../index.md)
