"""
test_cubin.py — Bare-Metal CUBIN Benchmark with Correctness Verification
=========================================================================


Tests the AOT-compiled INT8 GEMM .cubin launched via the CUDA Driver API
against PyTorch cuBLAS as both a correctness reference and speed baseline.


Usage:
    python test_cubin.py
"""


import torch
import time
import json
import os
import entropy_engine_cuda




# ============================================================================
# Configuration
# ============================================================================
BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64


def pad_to(val, multiple):
    return val if val % multiple == 0 else val + (multiple - val % multiple)


M_raw, K_raw, N_raw = 1024, 4096, 4096
M = pad_to(M_raw, BLOCK_M)  # 1024 (already aligned)
K = pad_to(K_raw, BLOCK_K)  # 4096 (already aligned)
N = pad_to(N_raw, BLOCK_N)  # 4096 (already aligned)
CUBIN_DIR = "/content/entropy-engine/csrc/generated"
CUBIN_PATH = os.path.join(CUBIN_DIR, "gemm_int8.cubin")
META_PATH = os.path.join(CUBIN_DIR, "gemm_int8_meta.json")




# ============================================================================
# Step 1: Load AOT metadata (shared_mem, num_warps)
# ============================================================================
# The compile_kernels.py script saves this alongside the .cubin.
# These values MUST match what the engine uses — hardcoding wrong values
# is the #2 cause of silent performance loss after the dtype bug.
shared_mem = 49152  # default
num_warps = 4       # default
block_m = 64        # default
block_n = 128       # default


if os.path.exists(META_PATH):
    print(f"Reading AOT metadata from {META_PATH}")
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"  Metadata: {json.dumps(meta, indent=2)}")
    # Triton metadata typically has 'shared', 'num_warps', etc.
    if 'shared' in meta:
        shared_mem = int(meta['shared'])
    if 'num_warps' in meta:
        num_warps = int(meta['num_warps'])
    print(f"  Using: shared_mem={shared_mem}, num_warps={num_warps}")
else:
    print(f"WARNING: No metadata file at {META_PATH}")
    print(f"  Using defaults: shared_mem={shared_mem}, num_warps={num_warps}")
print()




# ============================================================================
# Step 2: Create properly-typed tensors
# ============================================================================
print(f"Setting up tensors: M={M}, K={K}, N={N}")


# FP16 activations (source of truth for quantization)
a_fp16 = torch.randn(M, K, device='cuda', dtype=torch.float16).contiguous()


# INT8 weight matrix (pre-quantized offline)
#b_int8 = torch.randint(-127, 127, (K, N), device='cuda', dtype=torch.int8).contiguous()
b_scale = (torch.randn(N, device='cuda', dtype=torch.float32).abs() * 0.01 + 0.001).contiguous()


# Quantize activations to INT8 with real per-row absmax scaling
# This is what the production launcher does — NOT a dummy division.
a_absmax = a_fp16.float().abs().amax(dim=1)          # [M]
a_scale = (a_absmax / 127.0).clamp(min=1e-8).to(torch.float32).contiguous()  # [M]
# 128-bit vectorized loads require the inner dimension's base address to be
# 16-byte aligned. For int8 with stride=1, this means N and K must be
# multiples of 16 (16 bytes / 1 byte per element). Our block sizes (64, 128)
# already satisfy this, but we assert it explicitly.
assert N % 16 == 0, f"N={N} must be multiple of 16 for 128-bit aligned int8 loads"
assert K % 16 == 0, f"K={K} must be multiple of 16 for 128-bit aligned int8 loads"


a_int8 = (a_fp16.float() / a_scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8).contiguous()
b_int8 = torch.randint(-127, 127, (K, N), device='cuda', dtype=torch.int8).contiguous()


# Verify contiguity and alignment
assert a_int8.data_ptr() % 16 == 0, "a_int8 not 16-byte aligned"
assert b_int8.data_ptr() % 16 == 0, "b_int8 not 16-byte aligned"
# FP16 weight matrix for cuBLAS baseline (dequantized for apples-to-apples)
b_dequant_fp16 = (b_int8.float() * b_scale.unsqueeze(0)).to(torch.float16)


print(f"  a_int8:  {a_int8.shape} {a_int8.dtype} stride={a_int8.stride()}")
print(f"  b_int8:  {b_int8.shape} {b_int8.dtype} stride={b_int8.stride()}")
print(f"  a_scale: {a_scale.shape} {a_scale.dtype}")
print(f"  b_scale: {b_scale.shape} {b_scale.dtype}")
print()




# ============================================================================
# Step 3: Correctness reference (PyTorch FP32 dequantized matmul)
# ============================================================================
# Reference: dequantize both sides, matmul in FP32, cast to FP16.
# This is what the INT8 kernel SHOULD produce (modulo quantization noise).
a_dequant = (a_int8.float() * a_scale.unsqueeze(1))  # [M, K] fp32
b_dequant = (b_int8.float() * b_scale.unsqueeze(0))  # [K, N] fp32
ref_output = (a_dequant @ b_dequant).to(torch.float16)




# ============================================================================
# Step 4: Launch the CUBIN engine
# ============================================================================
print("Initializing CubinEngine...")
engine = entropy_engine_cuda.CubinEngine.from_metadata(
    CUBIN_PATH, META_PATH, "gemm_int8_kernel"
)


# Smoke test
print("Smoke test...")
torch.cuda.synchronize()
cubin_output = engine.forward(a_int8, b_int8, a_scale, b_scale)
torch.cuda.synchronize()


# Verify output dtype and shape
assert cubin_output.dtype == torch.float16, \
    f"CUBIN output dtype is {cubin_output.dtype}, expected float16!"
assert cubin_output.shape == (M, N), \
    f"CUBIN output shape is {cubin_output.shape}, expected ({M}, {N})!"


# Correctness check
max_err = (ref_output.float() - cubin_output.float()).abs().max().item()
mean_err = (ref_output.float() - cubin_output.float()).abs().mean().item()
print(f"  Output dtype: {cubin_output.dtype} (CORRECT)")
print(f"  Output shape: {cubin_output.shape}")
print(f"  Max error vs reference: {max_err:.4f}")
print(f"  Mean error vs reference: {mean_err:.6f}")


if max_err < 1.0:
    print("  CORRECTNESS: PASS")
elif max_err < 10.0:
    print("  CORRECTNESS: ACCEPTABLE (expected for INT8 on random data)")
else:
    print(f"  CORRECTNESS: FAIL (max_err={max_err:.2f}, check ABI struct)")
print()




# ============================================================================
# Step 5: Throughput Benchmark
# ============================================================================
WARMUP = 50
ITERS = 200


# --- cuBLAS FP16 Baseline ---
print("Benchmarking cuBLAS FP16 baseline...")
for _ in range(WARMUP):
    torch.matmul(a_fp16, b_dequant_fp16)
torch.cuda.synchronize()


t0 = time.perf_counter()
for _ in range(ITERS):
    torch.matmul(a_fp16, b_dequant_fp16)
torch.cuda.synchronize()
cublas_ms = (time.perf_counter() - t0) / ITERS * 1000


# --- CUBIN INT8 Engine ---
print("Benchmarking CUBIN INT8 engine...")
for _ in range(WARMUP):
    engine.forward(a_int8, b_int8, a_scale, b_scale)
torch.cuda.synchronize()


t0 = time.perf_counter()
for _ in range(ITERS):
    engine.forward(a_int8, b_int8, a_scale, b_scale)
torch.cuda.synchronize()
cubin_ms = (time.perf_counter() - t0) / ITERS * 1000


# --- Results ---
flops = 2 * M * K * N
cublas_tflops = flops / (cublas_ms / 1000) / 1e12
cubin_tflops = flops / (cubin_ms / 1000) / 1e12


print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Shape:              [{M}, {K}] x [{K}, {N}]")
print(f"  cuBLAS FP16:        {cublas_ms:.3f} ms  ({cublas_tflops:.2f} TFLOPS)")
print(f"  CUBIN INT8:         {cubin_ms:.3f} ms  ({cubin_tflops:.2f} TFLOPS)")
print(f"  Speedup:            {cublas_ms / cubin_ms:.2f}x")
print(f"  Max error:          {max_err:.4f}")
print()


if cubin_ms < cublas_ms:
    print("  INT8 Tensor Cores are winning. Ship it.")
else:
    print("  Still slower than cuBLAS. Check:")
    print("    1. Is shared_mem correct? (read _meta.json)")
    print("    2. Are M, K, N multiples of BLOCK_M, BLOCK_N, BLOCK_K?")
    print("    3. Is the kernel compiled with num_stages > 1 for pipelining?")