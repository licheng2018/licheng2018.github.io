# Licheng Zheng

<img src="assets/profile/licheng-profile.jpg" alt="Licheng Zheng" width="190" align="right">

Ph.D. Candidate in Electrical Engineering at École de technologie supérieure (ÉTS), Montreal.

I work at the intersection of **machine learning systems**, **AI infrastructure**, **performance engineering**, **GPU computing**, and **distributed optimization**. My recent projects span PyTorch compiler/runtime analysis, CUDA and Triton kernel optimization, LLM inference serving, distributed training, quantization, and communication-aware system benchmarking. I enjoy building practical systems, measuring them rigorously, and turning performance bottlenecks into clear engineering decisions across ML infrastructure, backend systems, and applied AI workloads.

[Resume](resume.md) · [Publications](publications.md) · [GitHub](https://github.com/licheng2018) · [LinkedIn](https://www.linkedin.com/in/licheng-zheng-589807178/) · musicsir at outlook dot com

## At a Glance

| Focus | What I Build |
|---|---|
| ML systems and AI infrastructure | Compiler/runtime experiments, inference benchmarks, distributed training workflows, and deployment-oriented performance studies |
| GPU performance engineering | CUDA kernels, Triton kernels, FlashAttention-style attention, profiling, memory traffic analysis, and operator-level optimization |
| Distributed and communication systems | NCCL collective benchmarks, FSDP/ZeRO analysis, DP/TP/PP experiments, and communication-aware scheduling |
| Applied optimization | Multi-agent RL and resource allocation for wireless, satellite, and infrastructure systems |

## Technical Strengths

`PyTorch` · `torch.compile` · `TorchDynamo` · `FX` · `TorchInductor` · `CUDA C++` · `Triton` · `NCCL` · `FSDP` · `ZeRO` · `LLM Serving` · `Quantization` · `Nsight` · `PyTorch Profiler` · `Python` · `C++`

## Featured Projects

| Area | Project | What It Demonstrates | Links |
|---|---|---|---|
| ML Compiler | **[PyTorch Compiler and AI Compiler Pipeline](projects/pytorch-compiler.md)** | Benchmarked eager vs `torch.compile`, graph capture, graph breaks, TorchInductor/Triton code generation, and built a toy compiler pipeline with IR, fusion, lowering, and Triton kernels. | [github](https://github.com/licheng2018/AI-compiler) |
| LLM Serving | **[LLM Inference Serving Benchmark](projects/llm-serving.md)** | Measured TTFT, TPOT, throughput, p95 latency, prefill/decode behavior, KV-cache pressure, and concurrency trade-offs on controlled serving workloads. | [github](https://github.com/licheng2018/serving-benchmark) |
| CUDA Kernels | **[CUDA Kernel Optimization for ML Operators](projects/cuda-kernels.md)** | Optimized Matmul, LayerNorm, Softmax, and attention-style operators using shared memory, tiling, loop unrolling, PyTorch extensions, and GPU profiling. | [github](https://github.com/licheng2018/cuda_projects) |
| Triton / Attention | **[IO-Aware Attention System with Triton and FlashAttention-style Optimization](projects/triton-flashattention.md)** | Implemented tiled Q/K/V loading, online softmax, SRAM reuse, fused kernels, and FlashAttention-style memory traffic reduction in Triton. | [github](https://github.com/licheng2018/triton) |
| Distributed Training | **[Distributed Training Optimization with FSDP, ZeRO, and Activation Checkpointing](projects/distributed-training.md)** | Built GPT-style distributed training workflows with FSDP, ZeRO sharding analysis, activation checkpointing, microbatch tuning, and rank-averaged metrics. | [github](https://github.com/licheng2018/gpt_training) |
| GPU Communication | **[NCCL Parallelism and Communication Benchmarking](projects/nccl-parallelism.md)** | Benchmarked AllReduce, ReduceScatter, AllGather, tensor parallelism, and pipeline parallelism, explaining latency-bound vs bandwidth-bound communication behavior. | [github](https://github.com/licheng2018/NCCL) |
| Model Deployment | **[LLM Quantization Benchmark and Inference Optimization](projects/quantization.md)** | Compared FP16 and INT8 inference on memory, TTFT, TPOT, throughput, total latency, and deployment trade-offs; connected results with AWQ/GPTQ concepts. | [github](https://github.com/licheng2018/Quantization-Deployment-Basics) |
| Applied AI | **[Multi-Agent RL for 5G Resource Scheduling](projects/multi-agent-rl.md)** | Modeled multi-cell interference and resource allocation as a multi-agent optimization problem using MADDPG and Actor-Critic methods. |  |

## Highlights

- Ranked **30/616, Top 5%** in Kaggle's Google - Fast or Slow? Predict AI Model Runtime competition with a GCN-based performance predictor.
- Research Assistant at ÉTS Sychromedia, working on communication-aware optimization frameworks for UAV-assisted 5G and LEO satellite systems.
- Mitacs research experience with Ciena and VMware on optimization, resource allocation, and system-level performance evaluation.
