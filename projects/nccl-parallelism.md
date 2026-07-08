# Distributed Training Systems and NCCL Parallelism

Designed and optimized distributed training experiments for large-model parallelism with communication-aware benchmarking.

## Work

- Built and benchmarked distributed communication workflows on a dual-GPU setup using NCCL collectives.
- Evaluated AllReduce, ReduceScatter, and AllGather across message sizes.
- Implemented and compared Data Parallelism, Tensor Parallelism, and Pipeline Parallelism.
- Used hands-on experiments inspired by Megatron-style tensor parallelism and GPipe-style pipeline parallelism.
- Measured communication overhead, scalability, and throughput under different execution patterns and workload granularities.

## Outcome

Optimized communication efficiency by improving message granularity, reducing unnecessary synchronization, and refining communication patterns, achieving 25-30% communication overhead reduction in the project benchmark.

[Back to Home](../index.md)
