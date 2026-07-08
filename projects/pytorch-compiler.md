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

[Back to Home](../index.md)
