# Licheng Zheng

Ph.D. Candidate in Electrical Engineering at École de technologie supérieure (ÉTS), Montreal.

I focus on **ML systems**, **AI infrastructure**, **compiler/runtime analysis**, **LLM inference serving**, **distributed training**, and **GPU kernel optimization**. My recent work connects PyTorch compiler workflows, CUDA/Triton kernels, FlashAttention-style optimization, and rigorous GPU benchmarking. I am especially interested in using LLM agents to automate kernel generation, performance debugging, and ML compiler optimization.

## Research and Engineering Interests

- ML compiler/runtime systems: PyTorch 2.x compile, TorchDynamo, FX, TorchInductor, graph breaks, dynamic-shape guards, SDPA, and kernel fusion
- GPU programming and optimization: CUDA C++, Triton, FlashAttention-style kernels, Nsight Systems/Compute, PyTorch Profiler, roofline analysis, and memory bandwidth modeling
- LLM inference systems: TTFT, TPOT, throughput, p95 latency, prefill/decode behavior, KV cache, batching, queueing, and concurrency limits
- Distributed training: PyTorch FSDP, ZeRO-1/2/3, activation checkpointing, NCCL collectives, DP/TP/PP, and communication profiling
- System-level optimization: multi-agent reinforcement learning for wireless, satellite, and resource scheduling systems

## Featured Projects

### [PyTorch Compiler and AI Compiler Pipeline](projects/pytorch-compiler.md) [[github](https://github.com/licheng2018/AI-compiler)]

Built a compiler/runtime benchmarking project and toy AI compiler pipeline to study PyTorch 2.x compilation, TorchDynamo graph capture, FX graphs, AOTAutograd, TorchInductor-generated Triton/CUDA kernels, graph breaks, dynamic-shape guards, operator fusion, and GPU runtime behavior.

### [LLM Inference Serving Benchmark](projects/llm-serving.md) [[github](https://github.com/licheng2018/serving-benchmark)]

Built a lightweight benchmark harness on T4 GPU to analyze TTFT, TPOT, throughput, p95 latency, prefill/decode behavior, and concurrency trade-offs across controlled prompt, output, and concurrency settings.

### [CUDA Kernel Optimization for ML Operators](projects/cuda-kernels.md) [[github](https://github.com/licheng2018/cuda_projects)]

Implemented and optimized CUDA kernels for Matmul, LayerNorm, and Softmax using shared memory, block tiling, loop unrolling, register-level tuning, and PyTorch C++/CUDA extensions.

### [Triton FlashAttention / IO-Aware Attention](projects/triton-flashattention.md) [[github](https://github.com/licheng2018/triton)]

Implemented a mini FlashAttention-style kernel with tiled Q/K/V loading, online softmax, SRAM reuse, kernel fusion, and reduced DRAM traffic.

### [Distributed Training with FSDP, ZeRO, and NCCL](projects/distributed-training.md) [[github](https://github.com/licheng2018/gpt_training)]

Built distributed training workflows for GPT-Neo-1.3B using PyTorch FSDP, activation checkpointing, ZeRO sharding analysis, NCCL profiling, and microbatch tuning.

### [NCCL Parallelism and Communication Benchmarking](projects/nccl-parallelism.md) [[github](https://github.com/licheng2018/NCCL)]

Benchmarked AllReduce, ReduceScatter, AllGather, Data Parallelism, Tensor Parallelism, and Pipeline Parallelism on a dual-GPU setup, analyzing latency-bound vs bandwidth-bound communication behavior.

### [LLM Quantization Benchmark and Inference Optimization](projects/quantization.md) [[github](https://github.com/licheng2018/Quantization-Deployment-Basics)]

Compared FP16, FP8, INT8, AWQ, and GPTQ inference settings under consistent workload configurations, measuring latency, throughput, memory footprint, p95 latency, and generation consistency.

### [Multi-Agent RL for 5G Resource Scheduling](projects/multi-agent-rl.md)

Modeled multi-cell interference and resource allocation as a multi-agent optimization problem using MADDPG and Actor-Critic frameworks for distributed power scheduling.

## Highlights

- Ranked **30/616, Top 5%** in Kaggle's Google - Fast or Slow? Predict AI Model Runtime competition with a GCN-based performance predictor.
- Research Assistant at ÉTS Sychromedia, working on communication-aware optimization frameworks for UAV-assisted 5G and LEO satellite systems.
- Mitacs research experience with Ciena and VMware on optimization, resource allocation, and system-level performance evaluation.

## Links

- [Resume](resume.md)
- [Publications](publications.md)
- [GitHub](https://github.com/licheng2018)
- [LinkedIn](https://www.linkedin.com/in/licheng-zheng-589807178/)
- Email: musicsir at outlook dot com
