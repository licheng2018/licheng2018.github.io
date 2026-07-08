# PyTorch Compiler and AI Compiler Pipeline

Built a compiler/runtime benchmarking project and toy AI compiler pipeline to study how PyTorch 2.x compilation, graph capture, operator fusion, and Triton code generation affect GPU performance for ML and attention workloads.

## Focus Areas

- PyTorch eager mode vs. `torch.compile`
- TorchDynamo graph capture and FX graph representation
- AOTAutograd and TorchInductor-generated Triton/CUDA kernels
- Graph breaks, dynamic-shape guards, recompilation triggers, and Python control-flow limits
- Operator fusion, kernel launch overhead, memory traffic, and GPU runtime behavior

## Work

- Benchmarked MLP, LayerNorm, elementwise, naive attention, and SDPA attention workloads.
- Compared compilation overhead, cold-start compile time, steady-state latency, kernel fusion benefits, and optimized library dispatch behavior.
- Investigated `.item()`, tensor-dependent branching, dynamic shapes, and small-batch overhead to explain when compiled execution is not faster than eager execution.
- Inspected TorchInductor-generated code and profiling results to connect compiler decisions with GPU kernel launches, memory movement, and fusion opportunities.
- Built a toy AI compiler pipeline with frontend IR construction, graph-level pattern fusion, lowering, and Triton code generation.

## Outcome

Demonstrated how operator fusion can reduce intermediate tensor materialization, global memory movement, and kernel launch overhead, linking frontend graph transformations to backend GPU performance.

[Back to Home](../index.md)
