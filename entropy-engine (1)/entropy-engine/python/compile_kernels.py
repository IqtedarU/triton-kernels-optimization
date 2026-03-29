"""
compile_kernels.py — JIT-to-AOT Bypass for Triton 3.6
======================================================
"""
import torch
import triton
import os
import json
from kernels import gemm_int8_kernel, gemm_fp8_kernel, gemm_fp16_kernel

def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "csrc", "generated"))
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Dummy Tensors
    # PyTorch automatically aligns these to 16-byte boundaries.
    # When Triton's JIT sees them, it automatically applies `tt.divisibility=16`,
    # which permanently fixes the "AOT Blindness" and unlocks 4-stage pipelining.
    M, K, N = 1024, 1024, 1024
    a_i8 = torch.empty((M, K), dtype=torch.int8, device='cuda')
    b_i8 = torch.empty((K, N), dtype=torch.int8, device='cuda')
    c_f16 = torch.empty((M, N), dtype=torch.float16, device='cuda')
    a_f16 = torch.empty((M, K), dtype=torch.float16, device='cuda')
    scale = torch.empty(M, dtype=torch.float32, device='cuda')

    # Common compile kwargs
    kwargs = {
        "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64,
        "grid": (1,), "num_warps": 8, "num_stages": 4, "warmup": True
    }

    print("🚀 Triggering JIT-to-AOT Bypass...\n")

    # --- INT8 ---
    print("Compiling gemm_int8...")
    comp_i8 = gemm_int8_kernel.fn.run(
        a_i8, b_i8, c_f16, scale, scale, M, N, K,
        a_i8.stride(0), a_i8.stride(1), b_i8.stride(0), b_i8.stride(1), c_f16.stride(0), c_f16.stride(1),
        **kwargs
    )
    with open(os.path.join(out_dir, "gemm_int8.cubin"), "wb") as f: f.write(comp_i8.asm["cubin"])
    with open(os.path.join(out_dir, "gemm_int8_meta.json"), "w") as f:
        json.dump({"shared": comp_i8.metadata.shared, "num_warps": 8, "name": "gemm_int8_kernel", "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, f, indent=2)
    print(f"  [CRITICAL] shared_mem = {comp_i8.metadata.shared} bytes\n")

    # --- FP8 ---
    print("Compiling gemm_fp8...")
    comp_fp8 = gemm_fp8_kernel.fn.run(
        a_f16, b_i8, c_f16, scale, scale, M, N, K,
        a_f16.stride(0), a_f16.stride(1), b_i8.stride(0), b_i8.stride(1), c_f16.stride(0), c_f16.stride(1),
        **kwargs
    )
    with open(os.path.join(out_dir, "gemm_fp8.cubin"), "wb") as f: f.write(comp_fp8.asm["cubin"])
    with open(os.path.join(out_dir, "gemm_fp8_meta.json"), "w") as f:
        json.dump({"shared": comp_fp8.metadata.shared, "num_warps": 8, "name": "gemm_fp8_kernel", "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, f, indent=2)
    print(f"  [CRITICAL] shared_mem = {comp_fp8.metadata.shared} bytes\n")

    # --- FP16 ---
    print("Compiling gemm_fp16...")
    comp_fp16 = gemm_fp16_kernel.fn.run(
        a_f16, b_i8, c_f16, scale, M, N, K,
        a_f16.stride(0), a_f16.stride(1), b_i8.stride(0), b_i8.stride(1), c_f16.stride(0), c_f16.stride(1),
        **kwargs
    )
    with open(os.path.join(out_dir, "gemm_fp16.cubin"), "wb") as f: f.write(comp_fp16.asm["cubin"])
    with open(os.path.join(out_dir, "gemm_fp16_meta.json"), "w") as f:
        json.dump({"shared": comp_fp16.metadata.shared, "num_warps": 8, "name": "gemm_fp16_kernel", "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, f, indent=2)
    print(f"  [CRITICAL] shared_mem = {comp_fp16.metadata.shared} bytes\n")
    
    print("✅ AOT Dump Complete! Ready for C++.")

if __name__ == "__main__":
    main()