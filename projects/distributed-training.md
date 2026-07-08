# Distributed Training Optimization with FSDP, ZeRO, and Activation Checkpointing

Designed and optimized large-scale distributed training workflows for billion-parameter language models using PyTorch and NCCL.

## Work

- Implemented FSDP full-shard training for GPT-Neo-1.3B under constrained multi-GPU memory.
- Analyzed ZeRO-1/2/3 sharding strategies and their impact on parameter, gradient, and optimizer-state memory footprint.
- Integrated activation checkpointing to reduce peak activation memory and support larger effective batch sizes.
- Tuned microbatch size, gradient accumulation, and sharding configurations to balance memory usage, GPU utilization, and training stability.
- Profiled NCCL collectives and synchronization points to identify communication bottlenecks.

## Outcome

Observed up to 2x training throughput improvement in controlled experiments after tuning microbatching, checkpointing, and sharding configurations.

[Back to Home](../index.md)
