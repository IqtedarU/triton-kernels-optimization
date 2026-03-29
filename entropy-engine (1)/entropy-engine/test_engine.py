# test_engine.py
import torch
import torch.nn.functional as F
import time
import entropy_engine_cuda

print("✅ C++ Engine Loaded Successfully!")

# Setup Dummy Tensors
B, S, H, Inter = 1, 1024, 2048, 5632
hidden = torch.randn(B, S, H, device='cuda', dtype=torch.float16)
gate_w = torch.randn(Inter, H, device='cuda', dtype=torch.float16)
up_w = torch.randn(Inter, H, device='cuda', dtype=torch.float16)
down_w = torch.randn(H, Inter, device='cuda', dtype=torch.float16)

# Init C++ Engine
cpp_engine = entropy_engine_cuda.EntropyMLPEngine(gate_w, up_w, down_w, 1.5, 2.5)

# Warmup
for _ in range(10): cpp_engine.forward(hidden, 1.0)
torch.cuda.synchronize()

# Time C++
t0 = time.perf_counter()
for _ in range(100):
    _ = cpp_engine.forward(hidden, 1.0)
torch.cuda.synchronize()
cpp_time = (time.perf_counter() - t0) / 100 * 1000

# Time Python Base
t0 = time.perf_counter()
for _ in range(100):
    entropy = 1.0
    if entropy < 1.5:
        _ = F.linear(F.silu(F.linear(hidden, gate_w)) * F.linear(hidden, up_w), down_w)
    elif entropy < 2.5:
        _ = F.linear(F.silu(F.linear(hidden, gate_w)) * F.linear(hidden, up_w), down_w)
    else:
        _ = F.linear(F.silu(F.linear(hidden, gate_w)) * F.linear(hidden, up_w), down_w)
torch.cuda.synchronize()
python_time = (time.perf_counter() - t0) / 100 * 1000

print(f"Python Latency: {python_time:.3f} ms")
print(f"C++ Latency:    {cpp_time:.3f} ms")
print(f"Speedup:        {python_time / cpp_time:.2f}x faster dispatch!")