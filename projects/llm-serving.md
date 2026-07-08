# LLM Serving Benchmark and Inference Systems Analysis

Designed and benchmarked LLM inference workloads to analyze serving performance, prefill/decode behavior, and concurrency trade-offs on GPU.

## Metrics

- Time to first token (TTFT)
- Time per output token (TPOT)
- End-to-end latency
- Throughput
- p95 latency
- Prefill/decode bottlenecks
- Concurrency scaling

## Work

- Built a lightweight LLM serving benchmark harness on T4 GPU across controlled prompt and output settings.
- Implemented single-request experiments to characterize how prompt length impacts TTFT and how output length affects TPOT and generation latency.
- Developed prompt-length and output-length sweeps, generating benchmark curves such as TTFT vs. prompt length and TPOT vs. output length.
- Extended the benchmark to multi-request concurrency levels from 1 to 16.
- Analyzed throughput scaling, average latency, and tail latency under request contention.

## Outcome

Produced deployment-oriented performance analysis explaining throughput saturation, latency degradation under contention, and how prefill/decode bottlenecks affect batching, queueing, concurrency limits, and latency-sensitive serving placement.

[Back to Home](../index.md)
