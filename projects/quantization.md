# LLM Quantization Benchmark and Inference Optimization

Built an LLM quantization benchmark pipeline to compare FP16 inference with FP8, INT8, AWQ, and GPTQ models under consistent prompt, output-length, batch/concurrency, and hardware settings.

## Metrics

- TTFT
- TPOT
- Throughput
- End-to-end latency
- p95 latency
- GPU memory footprint
- Generation consistency

## Work

- Measured quantized and non-quantized inference under fixed workload settings.
- Analyzed how reduced weight and activation precision affects prefill/decode latency.
- Studied KV-cache memory pressure and serving throughput under quantized inference.
- Identified cases where quantized inference improved serving efficiency.
- Documented cases where dequantization overhead, unsupported kernels, or quality degradation reduced practical benefit.

## Outcome

Generated deployment-oriented analysis to guide quantization choices for memory-constrained and latency-sensitive LLM serving workloads.

[Back to Home](../index.md)
