# IO-Aware Attention System with Triton and FlashAttention-style Optimization

Designed and implemented IO-aware attention kernels to reduce global memory traffic.

## Motivation

Attention has high compute complexity and substantial memory traffic. This project studied the gap between theoretical computation and practical memory movement, focusing on IO-bound bottlenecks in naive attention implementations.

## Work

- Modeled attention compute complexity and memory traffic.
- Built a naive Triton attention baseline using QK^T, Softmax, and V with mask support.
- Implemented a mini FlashAttention-style kernel with tiled Q/K/V loading.
- Used online softmax with running max and running sum.
- Reused SRAM and fused computation to reduce global memory access.

## Outcome

Achieved roughly 2x speedup compared with the naive attention implementation, with improved global memory efficiency and reduced DRAM traffic.

[Back to Home](../index.md)
