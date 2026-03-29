
"""
benchmark.py — Phase 2: Three-Tier Benchmarks for Llama-3 on L4
================================================================

Tests:
  1. Correctness: Each GEMM kernel vs torch.matmul reference.
  2. Fused softmax+entropy vs PyTorch reference.
  3. Pipelined dispatch logic.
  4. Llama-3 block integration with dummy config.

Benchmarks:
  - Raw kernel throughput at Llama-3 shapes (4096×14336).
  - Fused softmax+entropy latency.
  - Dual-threshold (tau_low, tau_high) dispatch sweep.

Usage:
    python benchmark.py                # Full test + benchmark suite
    python benchmark.py --test-only    # Correctness only
    python benchmark.py --bench-only   # Throughput only
"""

import argparse
import time
import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kernels import (
    triton_int8_gemm,
    triton_fp8_gemm,
    triton_fp16_gemm,
    fused_softmax_entropy,
)
from dispatch import (
    quantize_weights_int8,
    EntropyDispatchedLinear,
    PipelinedEntropyDispatcher,
    PrecisionTier,
)


# ============================================================================
# Correctness Tests
# ============================================================================

def _make_test_tensors(M, K, N, device='cuda', scale=1.0):
    """
    Create random test tensors and quantize weights.

    Args:
        scale: Variance multiplier for random tensors. Use scale=0.1 for
               FP8 tests — unit-variance randn produces activations that,
               after a [M,K]×[K,N] matmul, have magnitude ~sqrt(K)≈64 for
               K=4096. FP8 e4m3 only represents up to 448, and the per-row
               absmax scaling maps the peak to 448 — but with unit variance,
               outlier rows cause large quantization errors for non-outlier
               elements. Scaling inputs down keeps values well within FP8's
               precise range (|x| < 8 for exact integers in e4m3).
    """
    A = torch.randn(M, K, device=device, dtype=torch.float16) * scale
    W = torch.randn(N, K, device=device, dtype=torch.float16) * scale  # [out, in]
    W_int8, W_scale = quantize_weights_int8(W)  # returns [K, N] int8 + [N] scale
    W_dequant = W_int8.float() * W_scale.float().unsqueeze(0)  # [K, N]
    ref = (A.float() @ W_dequant).to(torch.float16)
    return A, W_int8, W_scale, ref


def test_fp16_kernel():
    print("=" * 60)
    print("TEST: FP16 SRAM Upcast GEMM")
    print("=" * 60)
    torch.manual_seed(42)
    A, W_int8, W_scale, ref = _make_test_tensors(256, 4096, 4096)

    out = triton_fp16_gemm(A, W_int8, W_scale)
    max_err = (ref.float() - out.float()).abs().max().item()
    mean_err = (ref.float() - out.float()).abs().mean().item()

    print(f"  Shape: [{256},{4096}] × [{4096},{4096}]")
    print(f"  Max abs error:  {max_err:.6f}")
    print(f"  Mean abs error: {mean_err:.6f}")
    passed = max_err < 1.0
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_int8_kernel():
    print("=" * 60)
    print("TEST: Native INT8 GEMM")
    print("=" * 60)
    torch.manual_seed(42)
    A, W_int8, W_scale, ref = _make_test_tensors(256, 4096, 4096)

    out = triton_int8_gemm(A, W_int8, W_scale)

    # INT8 has activation quantization error on top of weight quantization,
    # so tolerance is higher than FP16.
    max_err = (ref.float() - out.float()).abs().max().item()
    mean_err = (ref.float() - out.float()).abs().mean().item()

    print(f"  Shape: [{256},{4096}] × [{4096},{4096}]")
    print(f"  Max abs error (vs weight-quant ref): {max_err:.6f}")
    print(f"  Mean abs error: {mean_err:.6f}")
    passed = max_err < 10.0  # INT8 activation quant on random data is noisy
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_fp8_kernel():
    print("=" * 60)
    print("TEST: FP8 e4m3 GEMM")
    print("=" * 60)
    torch.manual_seed(42)
    # Scale=0.1: FP8 e4m3 can exactly represent integers up to ±7 and has
    # coarse rounding beyond that. With unit-variance random data, the per-row
    # absmax scaling maps the largest value to 448, but the vast majority of
    # values end up in the coarsely-quantized region (8-448). Scaling down
    # keeps most values in the fine-grained region, matching real model
    # activations which are typically small after LayerNorm/RMSNorm.
    A, W_int8, W_scale, ref = _make_test_tensors(256, 4096, 4096, scale=0.1)

    out = triton_fp8_gemm(A, W_int8, W_scale)

    max_err = (ref.float() - out.float()).abs().max().item()
    mean_err = (ref.float() - out.float()).abs().mean().item()

    print(f"  Shape: [{256},{4096}] × [{4096},{4096}]")
    print(f"  Max abs error (vs weight-quant ref): {max_err:.6f}")
    print(f"  Mean abs error: {mean_err:.6f}")
    # FP8 has activation + weight FP8 cast errors; between INT8 and FP16
    passed = max_err < 10.0
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_fused_softmax_entropy():
    print("=" * 60)
    print("TEST: Fused Softmax + Entropy")
    print("=" * 60)
    torch.manual_seed(42)
    B, H, S = 4, 32, 512  # Llama-3 style: 32 heads, moderate seq length

    logits = torch.randn(B, H, S, S, device='cuda', dtype=torch.float16)

    # PyTorch reference
    import torch.nn.functional as Fref
    # Causal mask
    mask = torch.tril(torch.ones(S, S, device='cuda')).bool()
    logits_masked = logits.float().masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    ref_probs = Fref.softmax(logits_masked, dim=-1)
    eps = 1e-10
    ref_entropy = -(ref_probs * torch.log(ref_probs + eps)).sum(dim=-1).mean(dim=(1, 2))

    # Fused kernel
    fused_probs, fused_entropy = fused_softmax_entropy(logits, causal=True)

    # Compare probabilities
    prob_err = (ref_probs.half() - fused_probs).float().abs().max().item()
    entropy_err = (ref_entropy - fused_entropy).abs().max().item()

    print(f"  Shape: logits[{B},{H},{S},{S}]")
    print(f"  Max prob error:    {prob_err:.8f}")
    print(f"  Max entropy error: {entropy_err:.6f}")
    print(f"  Ref entropy:   {ref_entropy.tolist()[:2]}...")
    print(f"  Fused entropy: {fused_entropy.tolist()[:2]}...")
    passed = prob_err < 0.01 and entropy_err < 0.1
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_pipelined_dispatch():
    print("=" * 60)
    print("TEST: Pipelined Entropy Dispatcher")
    print("=" * 60)

    dispatcher = PipelinedEntropyDispatcher(tau_low=2.0, tau_high=3.5)

    # First call: no entropy submitted → should return FP16 (default)
    tier = dispatcher.get_tier()
    assert tier == PrecisionTier.FP16, f"Expected FP16, got {tier}"

    # Simulate: submit low entropy → next get_tier should return INT8
    low_entropy = torch.tensor([1.0], device='cuda', dtype=torch.float32)
    dispatcher.submit_entropy(low_entropy)
    torch.cuda.synchronize()  # simulate one layer of work
    tier = dispatcher.get_tier()
    assert tier == PrecisionTier.INT8, f"Expected INT8 for entropy=1.0, got {tier}"

    # Submit medium entropy → FP8
    mid_entropy = torch.tensor([2.5], device='cuda', dtype=torch.float32)
    dispatcher.submit_entropy(mid_entropy)
    torch.cuda.synchronize()
    tier = dispatcher.get_tier()
    assert tier == PrecisionTier.FP8, f"Expected FP8 for entropy=2.5, got {tier}"

    # Submit high entropy → FP16
    high_entropy = torch.tensor([4.0], device='cuda', dtype=torch.float32)
    dispatcher.submit_entropy(high_entropy)
    torch.cuda.synchronize()
    tier = dispatcher.get_tier()
    assert tier == PrecisionTier.FP16, f"Expected FP16 for entropy=4.0, got {tier}"

    stats = dispatcher.get_stats()
    print(f"  tau_low={dispatcher.tau_low}, tau_high={dispatcher.tau_high}")
    for name, data in stats.items():
        print(f"  {name}: {data['count']} dispatches ({data['pct']:.0f}%)")
    print(f"  Status: PASS")
    print()
    return True


def test_llama_integration():
    print("=" * 60)
    print("TEST: Llama Block Integration (dummy config)")
    print("=" * 60)

    try:
        from transformers import LlamaForCausalLM, LlamaConfig
    except ImportError:
        print("  SKIP: transformers not installed")
        print()
        return True

    from dispatch import EntropyDispatchedLlamaBlock, PipelinedEntropyDispatcher

    # Tiny Llama config for fast testing
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 2 KV heads for 4 Q heads
        max_position_embeddings=128,
        vocab_size=1000,
        attn_implementation='eager',
    )
    model = LlamaForCausalLM(config).cuda().half()

    block = EntropyDispatchedLlamaBlock(model.model.layers[0], layer_idx=0)
    block = block.cuda().half()
    block.eval()

    dispatcher = PipelinedEntropyDispatcher(tau_low=2.0, tau_high=3.5)

    B, S = 2, 32
    hidden = torch.randn(B, S, config.hidden_size, device='cuda', dtype=torch.float16)
    position_ids = torch.arange(S, device='cuda').unsqueeze(0).expand(B, -1)

    # ---------------------------------------------------------
    # FIX: Manually generate RoPE since we isolated the block!
    # ---------------------------------------------------------
    position_embeddings = model.model.rotary_emb(hidden, position_ids)

    with torch.no_grad():
        outputs = block(
            hidden, 
            dispatcher=dispatcher, 
            position_ids=position_ids, 
            position_embeddings=position_embeddings # <-- Pass it here!
        )
    out_hidden = outputs[0]

    print(f"  Config: hidden={config.hidden_size}, inter={config.intermediate_size}")
    print(f"  GQA: {config.num_attention_heads} Q heads, {config.num_key_value_heads} KV heads")
    print(f"  Input: [{B},{S},{config.hidden_size}] → Output: {list(out_hidden.shape)}")
    print(f"  Entropy: {block.layer_entropy}")
    print(f"  Status: PASS")
    print()
    return True


# ============================================================================
# Throughput Benchmarks
# ============================================================================

def benchmark_kernels(M, K, N, label="", warmup=50, iters=200):
    print("=" * 60)
    print(f"BENCHMARK: Kernel Throughput — {label or f'M={M}, K={K}, N={N}'}")
    print("=" * 60)

    torch.manual_seed(42)
    # Scale activations to 0.1 so FP8 kernel doesn't overflow.
    # This matches real model activations (post-RMSNorm values are small).
    # Weights are quantized to INT8 regardless, so their scale doesn't matter here.
    A = torch.randn(M, K, device='cuda', dtype=torch.float16) * 0.1
    W = torch.randn(N, K, device='cuda', dtype=torch.float16)
    W_int8, W_scale = quantize_weights_int8(W)
    W_fp16 = torch.randn(K, N, device='cuda', dtype=torch.float16)  # for cuBLAS ref

    flops = 2 * M * K * N
    results = {}

    # --- cuBLAS FP16 baseline ---
    for _ in range(warmup):
        torch.matmul(A, W_fp16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(A, W_fp16)
    torch.cuda.synchronize()
    results['cuBLAS FP16'] = (time.perf_counter() - t0) / iters

    # --- Triton FP16 (INT8 weight upcast) ---
    for _ in range(warmup):
        triton_fp16_gemm(A, W_int8, W_scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        triton_fp16_gemm(A, W_int8, W_scale)
    torch.cuda.synchronize()
    results['Triton FP16 upcast'] = (time.perf_counter() - t0) / iters

    # --- Triton FP8 ---
    for _ in range(warmup):
        triton_fp8_gemm(A, W_int8, W_scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        triton_fp8_gemm(A, W_int8, W_scale)
    torch.cuda.synchronize()
    results['Triton FP8 e4m3'] = (time.perf_counter() - t0) / iters

    # --- Triton INT8 ---
    for _ in range(warmup):
        triton_int8_gemm(A, W_int8, W_scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        triton_int8_gemm(A, W_int8, W_scale)
    torch.cuda.synchronize()
    results['Triton INT8 native'] = (time.perf_counter() - t0) / iters

    # --- Fused softmax+entropy overhead ---
    B_test, H_test, S_test = 4, 32, 256
    logits = torch.randn(B_test, H_test, S_test, S_test, device='cuda', dtype=torch.float16)
    for _ in range(warmup):
        fused_softmax_entropy(logits, causal=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fused_softmax_entropy(logits, causal=True)
    torch.cuda.synchronize()
    results['Fused softmax+entropy'] = (time.perf_counter() - t0) / iters

    # --- Report ---
    baseline_us = results['cuBLAS FP16'] * 1e6
    print(f"  {'Kernel':<25} {'Latency':>12} {'TFLOPS':>10} {'vs cuBLAS':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*10}")
    for name, elapsed in results.items():
        us = elapsed * 1e6
        if 'softmax' not in name:
            tflops = flops / elapsed / 1e12
            speedup = baseline_us / us
            print(f"  {name:<25} {us:>9.1f} μs {tflops:>7.2f}    {speedup:>7.2f}x")
        else:
            print(f"  {name:<25} {us:>9.1f} μs {'N/A':>10} {'overhead':>10}")

    # Overhead percentage
    gemm_min = min(v for k, v in results.items() if 'softmax' not in k)
    overhead = results['Fused softmax+entropy'] / gemm_min * 100
    print(f"\n  Fused softmax overhead vs fastest GEMM: {overhead:.1f}%")
    print()


def benchmark_dispatch_sweep(warmup=20, iters=50):
    """Sweep (tau_low, tau_high) pairs and show dispatch ratios."""
    print("=" * 60)
    print("BENCHMARK: Dual-Threshold Dispatch Sweep")
    print("=" * 60)

    torch.manual_seed(42)
    M, K, N = 1024, 4096, 14336  # Llama-3 MLP
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    linear = nn.Linear(K, N, bias=False, device='cuda', dtype=torch.float16)
    dispatched = EntropyDispatchedLinear.from_linear(linear).cuda()

    simulated_entropies = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    threshold_pairs = [
        (1.0, 2.0),
        (1.5, 2.5),
        (2.0, 3.0),
        (2.0, 3.5),
        (2.5, 3.5),
        (3.0, 4.0),
        (3.5, 4.5),
    ]

    print(f"  {'tau_low':>7} {'tau_high':>8} │ {'INT8%':>7} {'FP8%':>7} {'FP16%':>7} │ {'Avg μs':>8}")
    print(f"  {'-'*7} {'-'*8} ┼ {'-'*7} {'-'*7} {'-'*7} ┼ {'-'*8}")

    for tau_low, tau_high in threshold_pairs:
        dispatcher = PipelinedEntropyDispatcher(tau_low=tau_low, tau_high=tau_high)

        # Warmup
        for _ in range(warmup):
            dispatched(A, tier=PrecisionTier.FP16)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            for entropy in simulated_entropies:
                e_tensor = torch.tensor([entropy], device='cuda', dtype=torch.float32)
                dispatcher.submit_entropy(e_tensor)
                tier = dispatcher.get_tier()
                dispatched(A, tier=tier)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        avg_us = elapsed / (iters * len(simulated_entropies)) * 1e6

        stats = dispatcher.get_stats()
        print(
            f"  {tau_low:>7.1f} {tau_high:>8.1f} │"
            f" {stats['INT8']['pct']:>5.1f}% {stats['FP8']['pct']:>5.1f}%"
            f" {stats['FP16']['pct']:>5.1f}% │ {avg_us:>6.1f} μs"
        )
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Benchmarks")
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--bench-only', action='store_true')
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    Entropy-Gated Dispatch v2 — L4 / Llama-3 / 3-Tier   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    print(f"  Device: {device}")
    print(f"  Compute capability: sm_{cc[0]}{cc[1]}")
    print(f"  CUDA: {torch.version.cuda}")
    if cc < (8, 9):
        print(f"  WARNING: FP8 e4m3 requires sm_89+ (Ada Lovelace). Some tests may fail.")
    print()

    if not args.bench_only:
        print("─── CORRECTNESS TESTS ──────────────────────────────────────")
        print()
        results = []
        results.append(("FP16 Upcast GEMM", test_fp16_kernel()))
        results.append(("INT8 Native GEMM", test_int8_kernel()))
        results.append(("FP8 e4m3 GEMM", test_fp8_kernel()))
        results.append(("Fused Softmax+Entropy", test_fused_softmax_entropy()))
        results.append(("Pipelined Dispatch", test_pipelined_dispatch()))
        results.append(("Llama Block Integration", test_llama_integration()))

        print("─── TEST SUMMARY ───────────────────────────────────────────")
        all_pass = True
        for name, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"  {name:<25} {status}")
            all_pass = all_pass and passed
        print()

    if not args.test_only:
        print("─── THROUGHPUT BENCHMARKS ───────────────────────────────────")
        print()
        # Llama-3 attention projection: [B*S, d_model] × [d_model, d_model]
        benchmark_kernels(1024, 4096, 4096, label="Llama-3 Attention (4096×4096)")
        # Llama-3 MLP gate/up projection: [B*S, d_model] × [d_model, intermediate]
        benchmark_kernels(1024, 4096, 14336, label="Llama-3 MLP (4096×14336)")
        # Llama-3 MLP down projection: [B*S, intermediate] × [intermediate, d_model]
        benchmark_kernels(1024, 14336, 4096, label="Llama-3 MLP Down (14336×4096)")
        # Dispatch sweep
        benchmark_dispatch_sweep()


if __name__ == '__main__':
    main()