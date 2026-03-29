"""
kernels.py — Phase 2: Three-Tier Triton Kernels for NVIDIA L4 (sm_89)
======================================================================
"""




import torch
import triton
import triton.language as tl




# ============================================================================
# Autotune Configurations for L4 (sm_89, Ada Lovelace)
# ============================================================================
# L4 SM limit: ~100KB dynamic shared memory.
# Memory per stage (INT8/FP8) = (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * 1 byte.
# We must ensure memory_per_stage * num_stages < ~95,000 bytes.
QUANTIZED_GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
]




# FP16 elements are 2 bytes, so shared memory usage is doubled.
# We reduce BLOCK_K to keep the pipeline stages safe.
FP16_GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
]




# ============================================================================
# Kernel 1: Native INT8×INT8 GEMM (Ada Tensor Cores)
# ============================================================================
@triton.autotune(configs=QUANTIZED_GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_int8_kernel(
    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_m = pid // num_n_tiles
    tile_n = pid % num_n_tiles




    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)




    # 1. Load scales (Row-wise A, Col-wise B)
    # We keep the mask here because scales are small and not in the inner loop
    a_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=1.0).to(tl.float32)
    b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)




    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
   
    # 2. THE CRITICAL INNER LOOP
    # REMOVED MASKS: Shapes are multiples of BLOCK_K/M/N.
    # This triggers Asynchronous Hardware Copies (cp.async)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k




        a_tile = tl.load(a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)
        b_tile = tl.load(b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)




        acc += tl.dot(a_tile, b_tile)




    # 3. Vectorized De-quantization
    result = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
   
    # Final store back to FP16
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        result.to(tl.float16)
    )




# ============================================================================
# Kernel 2: FP8 e4m3 GEMM (Ada FP8 Tensor Cores)
# ============================================================================
@triton.autotune(configs=QUANTIZED_GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_fp8_kernel(
    a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_m = pid // num_n_tiles
    tile_n = pid % num_n_tiles




    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)




    a_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=1.0).to(tl.float32)
    b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
   
    a_inv_scale = (1.0 / a_scale).to(tl.float16)




    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
   
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k




        a_tile = tl.load(a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)
       
        a_scaled = a_tile * a_inv_scale[:, None]
        a_fp8 = a_scaled.to(tl.float8e4nv)




        b_int8 = tl.load(b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_fp8 = b_int8.to(tl.float16).to(tl.float8e4nv)




        acc += tl.dot(a_fp8, b_fp8)




    result = acc * a_scale[:, None] * b_scale[None, :]
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        result.to(tl.float16),
    )








# ============================================================================
# Kernel 3: FP16 GEMM with INT8 Weight Upcast in SRAM
# ============================================================================
@triton.autotune(configs=FP16_GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_fp16_kernel(
    a_ptr, b_ptr, c_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_m = pid // num_n_tiles
    tile_n = pid % num_n_tiles




    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)




    b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float16)




    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k




        a_tile = tl.load(a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)




        b_int8 = tl.load(b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_fp16 = b_int8.to(tl.float16) * b_scale[None, :]




        acc += tl.dot(a_tile, b_fp16)




    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.float16),
    )




# ============================================================================
# Kernel 4: Fused Softmax + Entropy (L4-Optimized)
# ============================================================================
# Design principles:
#   - One global read (logits), two global writes (probs, entropy)
#   - All intermediate math in registers/SRAM
#   - Vectorized 128-bit loads: BLOCK_S padded to multiple of 8 (8 × fp16 = 128 bits)
#   - Warp count tuned for L4 occupancy at S=512–4096
#   - Causal masking via branchless tl.where (no divergence)
#   - Entropy computed from shifted logits (avoids tl.log on probs):
#       H = log(sum_exp) - (1/sum_exp) * sum(logit * exp(logit))
#     This is numerically equivalent but avoids the expensive log(prob) per element.

@triton.jit
def fused_softmax_entropy_kernel(
    logits_ptr, probs_ptr, entropy_ptr,
    S,
    stride_row: tl.constexpr,
    BLOCK_S: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    row_idx = tl.program_id(0)
    query_pos = row_idx % S

    # Base pointer for this row — stride_row is constexpr so Triton
    # folds it into the address computation at compile time
    row_base = row_idx * stride_row
    offs = tl.arange(0, BLOCK_S)

    # --- VECTORIZED GLOBAL READ (128-bit) ---
    # With stride_col=1 and BLOCK_S as power-of-2, Triton emits
    # ld.global.v4.b32 (128-bit) when base is 16-byte aligned.
    # Padding elements beyond S get -inf → zero after exp.
    mask = offs < S
    logits = tl.load(
        logits_ptr + row_base + offs,
        mask=mask, other=float('-inf'),
    ).to(tl.float32)

    # Branchless causal mask
    if IS_CAUSAL:
        logits = tl.where(offs <= query_pos, logits, float('-inf'))

    # --- ALL MATH IN REGISTERS ---
    max_logit = tl.max(logits, axis=0)
    shifted = logits - max_logit
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=0)
    probs = exp_shifted * (1.0 / sum_exp)

    # Entropy: H(p) = log(Z) - sum(p * shifted)
    # Guard 0 * (-inf) = nan on causal boundary
    safe_shifted = tl.where(shifted == float('-inf'), 0.0, shifted)
    entropy = tl.log(sum_exp) - tl.sum(probs * safe_shifted, axis=0)

    # --- VECTORIZED GLOBAL WRITES (128-bit) ---
    tl.store(
        probs_ptr + row_base + offs,
        probs.to(tl.float16),
        mask=mask,
    )
    tl.store(entropy_ptr + row_idx, entropy)





# ============================================================================
# Python Launcher Functions
# ============================================================================




# ============================================================================
# Python Launcher Functions (Updated for Bare-Metal INT8 Speed)
# ============================================================================


def triton_int8_gemm(a: torch.Tensor, b_int8: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b_int8.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
   
    # 1. Quantize A to INT8 (Entropy-Gated Online Quantization)
    a_scale = (a.abs().amax(dim=1) / 127.0).clamp(min=1e-8).to(torch.float32)
    # Ensure quantization happens before kernel launch
    a_int8 = (a / a_scale[:, None]).to(torch.int8)


    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)


    # 2. Launch with a_int8 (Type *i8)
    gemm_int8_kernel[grid](
        a_int8, b_int8, c, a_scale, b_scale, M, N, K,
        a_int8.stride(0), a_int8.stride(1), b_int8.stride(0), b_int8.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def triton_fp8_gemm(a: torch.Tensor, b_int8: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b_int8.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
   
    # FP8 kernel handles the FP16 -> FP8 cast internally,
    # but we still need the scale.
    a_scale = (a.abs().amax(dim=1) / 448.0).clamp(min=1e-12).to(torch.float32)


    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)


    gemm_fp8_kernel[grid](
        a, b_int8, c, a_scale, b_scale, M, N, K,
        a.stride(0), a.stride(1), b_int8.stride(0), b_int8.stride(1),
        c.stride(0), c.stride(1),
    )
    return c






def triton_fp16_gemm(a: torch.Tensor, b_int8: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b_int8.shape
   
    # 1. Performance Insurance: Ensure memory is contiguous.
    # If the input is a view or slice, Triton's vectorized loads will stall.
    if not a.is_contiguous():
        a = a.contiguous()
    if not b_int8.is_contiguous():
        b_int8 = b_int8.contiguous()


    # 2. Correctness Fix: Ensure output is FP16
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)


    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)


    # 3. Launch Kernel
    # This kernel upcasts b_int8 -> fp16 inside SRAM, so 'a' remains fp16.
    gemm_fp16_kernel[grid](
        a, b_int8, c, b_scale,
        M, N, K,
        a.stride(0), a.stride(1),
        b_int8.stride(0), b_int8.stride(1),
        c.stride(0), c.stride(1),
    )
    return c






def fused_softmax_entropy(logits: torch.Tensor, causal: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, S, S2 = logits.shape
    assert S == S2

    # Force contiguity so stride_col=1 and rows are packed
    if not logits.is_contiguous():
        logits = logits.contiguous()

    total_rows = B * H * S
    probs = torch.empty_like(logits, dtype=torch.float16)
    row_entropies = torch.empty(total_rows, device=logits.device, dtype=torch.float32)

    BLOCK_S = triton.next_power_of_2(S)

    # stride_row = S for contiguous [B,H,S,S] tensor (last two dims)
    stride_row = logits.stride(-2)

    # Warp scaling: more threads for larger S to maximize memory bandwidth
    if BLOCK_S <= 256:
        num_warps = 4
    elif BLOCK_S <= 1024:
        num_warps = 8
    elif BLOCK_S <= 4096:
        num_warps = 16
    else:
        num_warps = 32

    fused_softmax_entropy_kernel[(total_rows,)](
        logits, probs, row_entropies,
        S,
        stride_row,       # constexpr — Triton sees divisibility at compile time
        BLOCK_S=BLOCK_S,
        IS_CAUSAL=causal,
        num_warps=num_warps,
    )

    per_batch_entropy = row_entropies.reshape(B, H, S).mean(dim=(1, 2))
    return probs, per_batch_entropy





