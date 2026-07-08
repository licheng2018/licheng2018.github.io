# PyTorch Compiler and AI Compiler Pipeline Project

This project studies how modern ML compiler stacks turn PyTorch programs into optimized GPU execution, and why compilation does not always make a workload faster. I built a compiler/runtime benchmarking project and a toy AI compiler pipeline to connect frontend graph transformations with backend GPU performance.

## Project Goal

The goal was to understand the full path from Python-level PyTorch code to generated GPU kernels:

- How `torch.compile` captures and transforms model execution.
- How TorchDynamo, FX, AOTAutograd, and TorchInductor interact.
- When compiler optimizations such as fusion reduce latency and memory traffic.
- Why graph breaks, dynamic shapes, Python control flow, or small workloads can erase the expected speedup.
- How a simplified compiler pipeline can lower graph patterns into Triton kernels.

## Compiler Stack Studied

- **PyTorch eager mode** as the baseline execution path.
- **`torch.compile`** for end-to-end compiled execution.
- **TorchDynamo** for Python frame evaluation and graph capture.
- **FX graph representation** for inspecting captured operator graphs.
- **AOTAutograd** for forward/backward graph handling.
- **TorchInductor** for generated Triton/CUDA kernel paths.
- **SDPA and attention dispatch** for understanding when PyTorch routes work to optimized kernels.

## Benchmark Workloads

I evaluated representative ML and attention workloads where compiler behavior can be inspected clearly:

- MLP blocks
- LayerNorm
- Elementwise operator chains
- Naive attention
- SDPA attention
- Fused elementwise patterns
- MLP-style kernels such as Linear + ReLU

For each workload, I compared eager execution and compiled execution across cold-start and steady-state settings.

## What I Measured

- Cold-start compile time
- Steady-state latency
- Compilation overhead
- Kernel launch count
- Operator fusion opportunities
- Optimized library dispatch behavior
- Memory traffic and intermediate tensor materialization
- Small-batch overhead and cases where eager execution remains competitive

## Key Investigation: Why Compiled Execution Is Not Always Faster

A major part of the project was debugging cases where `torch.compile` did not produce a speedup. I investigated:

- Graph breaks caused by Python control flow
- Tensor-dependent branching
- `.item()` usage
- Dynamic-shape guards
- Recompilation triggers
- Small workloads where compile overhead dominates
- Cases where eager mode already calls optimized kernels
- Cases where fusion opportunities are limited by graph boundaries

This helped separate two different questions: whether the compiler can capture a graph, and whether the captured graph has enough optimization opportunity to improve GPU runtime.

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

- Frontend IR construction
- Graph-level pattern matching
- Operator fusion passes
- Lowering from graph representation to backend code
- Triton code generation for fused elementwise and MLP-style kernels

Example fusion targets included elementwise chains and Linear + ReLU style patterns. The toy pipeline demonstrated how a compiler can identify repeated graph structures, fuse them, and generate a lower-level GPU implementation.

## Main Takeaways

- Operator fusion can reduce intermediate tensor materialization, global memory movement, and kernel launch overhead.
- Compilation benefits are workload-dependent; graph capture alone does not guarantee faster execution.
- Graph breaks, dynamic-shape guards, Python-side control flow, and `.item()` can prevent useful fusion or trigger recompilation.
- Small workloads can be dominated by compile overhead or launch overhead, making eager execution competitive.
- Inspecting generated code and profiler traces is essential for explaining compiler behavior, not just measuring latency.

## Skills Demonstrated

- ML compiler/runtime analysis
- PyTorch 2.x compiler workflow
- TorchDynamo and FX graph inspection
- TorchInductor generated-code analysis
- Triton code generation
- GPU profiling and performance debugging
- Benchmark design for cold-start and steady-state latency
- Operator fusion and memory-traffic reasoning

[Back to Home](../index.md)
