# Licheng Zheng

Ph.D. Candidate in Electrical Engineering at École de technologie supérieure (ÉTS), Montreal.

I focus on **AI Infrastructure**, **ML Systems**, **LLM Inference Serving**, **Distributed Training**, and **GPU Kernel Optimization**.

My recent work includes LLM serving benchmarking, CUDA/Triton kernel optimization, distributed training with FSDP/NCCL, quantization benchmarking, and IO-aware attention kernels.

## Featured Projects

### [LLM Inference Serving Benchmark](projects/llm-serving.md)
Built a lightweight benchmark harness to analyze TTFT, TPOT, throughput, p95 latency, prefill/decode behavior, and concurrency trade-offs.

### [CUDA Kernel Optimization for ML Operators](projects/cuda-kernels.md)
Implemented and optimized CUDA kernels for LayerNorm, Softmax, and Matmul using shared memory, warp-level primitives, and PyTorch C++/CUDA extensions.

### [Triton FlashAttention / IO-Aware Attention](projects/triton-flashattention.md)
Implemented a mini FlashAttention-style kernel with tiled Q/K/V loading, online softmax, SRAM reuse, and kernel fusion.

### [Distributed Training with FSDP, ZeRO, and NCCL](projects/distributed-training.md)
Built distributed training workflows for GPT-Neo-1.3B using PyTorch FSDP, activation checkpointing, and NCCL profiling.

### [LLM Quantization Benchmark](projects/quantization.md)
Compared FP16, FP8, INT8, AWQ, and GPTQ inference settings under consistent workload configurations.

## Links

- [Resume](resume.md)
- [Publications](publications.md)
- [GitHub](https://github.com/licheng2018)
- [LinkedIn](https://www.linkedin.com/in/licheng-zheng-589807178/)
- Email: musicsir at outlook com