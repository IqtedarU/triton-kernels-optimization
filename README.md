# Entropy-Gated Mixed-Precision Inference Engine

**Status:** Work in Progress

This repository contains a custom, mixed-precision inference engine for Large Language Models (LLMs) optimized for the NVIDIA L4 (Ada Lovelace) GPU. The engine dynamically routes matrix multiplications to FP16, FP8 (e4m3), or INT8 precisions at runtime based on the Shannon Entropy of the layer's token distribution. 

The goal is to provide a Pareto-optimal tradeoff between model perplexity and inference latency by utilizing high-throughput tensor cores only when the statistical confidence of the network allows it.

## Architecture

The project is split into three main components:
1. **The Dispatcher:** A dual-threshold Python dispatcher that calculates row-wise entropy and routes the tensor to the appropriate precision kernel.
2. **Triton Kernels:** Custom INT8 and FP8 GEMM kernels written in Triton. To eliminate VRAM round-trips, the system utilizes a custom fused Softmax + Entropy kernel optimized with 128-bit vectorized memory loads (`tt.divisibility=16`), pushing the L4 to its physical memory bandwidth limit (~300 GB/s).
3. **C++ Bare-Metal Bridge:** To bypass PyTorch and Python interpreter overhead, the Triton kernels are Ahead-Of-Time (AOT) compiled into `.cubin` binaries and launched directly via the CUDA Driver API using a custom C++ ABI struct wrapper (`CubinEngine`).

## Current Results

All benchmarks were performed on an NVIDIA L4 GPU (Compute Capability 8.9, CUDA 12.8).

### 1. Bare-Metal Hardware Performance (C++ API)
Testing the raw AOT-compiled INT8 engine against NVIDIA's proprietary `cuBLAS` FP16 baseline. This isolated benchmark runs 200 iterations via the C++ Driver API to measure pure hardware execution without Python overhead.

* **Shape:** [1024, 4096] x [4096, 4096]
* **cuBLAS FP16:** 0.573 ms (59.97 TFLOPS)
* **CUBIN INT8:** 0.553 ms (62.12 TFLOPS)
* **Speedup:** 1.04x over cuBLAS

### 2. Kernel Throughput (Llama-3 Shapes)
Performance of the precision-routed blocks for standard Llama-3 shapes.

| Kernel | Shape | Latency | vs cuBLAS FP16 |
| :--- | :--- | :--- | :--- |
| **Attention** | 4096 x 4096 | | |
| cuBLAS FP16 | | 639.7 μs | 1.00x |
| Triton FP8 e4m3 | | 576.7 μs | 1.11x |
| Triton INT8 native | | 577.1 μs | 1.11x |
| **MLP** | 4096 x 14336 | | |
| cuBLAS FP16 | | 2217.8 μs | 1.00x |
| Triton INT8 native | | 2266.3 μs | 0.98x |

*Note: The vectorized Fused Softmax + Entropy calculation overhead is currently benchmarked at 94.0 μs (Attention) and 149.4 μs (MLP), representing an overhead of just 6.6% - 16.3% over the fastest GEMM.*

### 3. End-to-End Quality Evaluation
Evaluating TinyLlama (1.1B) on the WikiText-2 test set (338,535 tokens) to map the Perplexity (PPL) vs. Latency Pareto curve across various gating thresholds (`tau_low`, `tau_high`).

* **Baseline (100% FP16):** PPL 8.51
* **Optimal Gating (0.41, 1.68):** PPL 8.54 (+0.33% degradation) 
* **Aggressive Gating (2.34, 6.03):** PPL 8.74 (+2.71% degradation) with 18% INT8 / 81% FP8 utilization.

## Work In Progress

While the hardware-level C++ benchmark proves the INT8 kernel outperforms cuBLAS, the current end-to-end Python evaluation loop still suffers from Python-level dispatch latency. 

**Next Steps:**
* Port the full 22-layer TinyLlama sequence loop into the C++ `CubinEngine` to realize the end-to-end wall-clock speedup observed in the isolated `test_cubin.py` benchmarks.
* Explore dynamic KV-Cache compression tied to the entropy metric.
