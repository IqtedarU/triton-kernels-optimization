"""
dispatch.py — Phase 2: Three-Tier Async Dispatch for Llama-3
=============================================================

Core components:
  1. PrecisionTier — INT8 / FP8 / FP16 enum.
  2. quantize_weights_int8() — Offline weight quantization (shared by all 3 tiers).
  3. EntropyDispatchedLinear — Drop-in nn.Linear replacement with 3-tier dispatch.
  4. PipelinedEntropyDispatcher — Zero-sync dispatch via pinned memory pipeline.
  5. EntropyDispatchedLlamaBlock — Wraps LlamaDecoderLayer with SwiGLU + fused entropy.

Dispatch logic (two thresholds):
  entropy < tau_low              →  INT8  (highest throughput, lowest precision)
  tau_low <= entropy < tau_high  →  FP8   (middle ground)
  entropy >= tau_high            →  FP16  (safe fallback, full precision)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from enum import IntEnum

from kernels import (
    triton_int8_gemm,
    triton_fp8_gemm,
    triton_fp16_gemm,
    fused_softmax_entropy,
)


class PrecisionTier(IntEnum):
    """Precision tiers ordered by compute throughput (highest first)."""
    INT8 = 0
    FP8 = 1
    FP16 = 2


# ============================================================================
# Weight Quantization (offline, run once at model load time)
# ============================================================================

def quantize_weights_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = weight.detach().float().t().contiguous()  # [K, N]
    col_absmax = w.abs().amax(dim=0)
    scale = (col_absmax / 127.0).clamp(min=1e-8)
    w_int8 = (w / scale.unsqueeze(0)).round().clamp(-127, 127).to(torch.int8)
    return w_int8.contiguous(), scale.contiguous()


# ============================================================================
# Pipelined Entropy Dispatcher (zero CPU-GPU sync)
# ============================================================================

class PipelinedEntropyDispatcher:
    def __init__(self, tau_low: float = 2.0, tau_high: float = 3.5):
        self.tau_low = tau_low
        self.tau_high = tau_high
        self._pinned = torch.empty(1, dtype=torch.float32, pin_memory=True)
        self._has_value = False
        self.tier_counts = {PrecisionTier.INT8: 0, PrecisionTier.FP8: 0, PrecisionTier.FP16: 0}

    def submit_entropy(self, entropy_batch: torch.Tensor):
        max_entropy = entropy_batch.max()
        self._pinned.copy_(max_entropy.unsqueeze(0), non_blocking=True)
        self._has_value = True

    def get_tier(self) -> PrecisionTier:
        if not self._has_value:
            return PrecisionTier.FP16

        val = self._pinned[0].item()

        if val < self.tau_low:
            tier = PrecisionTier.INT8
        elif val < self.tau_high:
            tier = PrecisionTier.FP8
        else:
            tier = PrecisionTier.FP16

        self.tier_counts[tier] += 1
        return tier

    def reset_stats(self):
        self._has_value = False
        self.tier_counts = {t: 0 for t in PrecisionTier}

    def get_stats(self) -> dict:
        total = sum(self.tier_counts.values())
        return {
            tier.name: {
                'count': self.tier_counts[tier],
                'pct': self.tier_counts[tier] / max(total, 1) * 100,
            }
            for tier in PrecisionTier
        }


# ============================================================================
# Entropy-Dispatched Linear Layer (3-tier)
# ============================================================================

class EntropyDispatchedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_int8', None)
        self.register_buffer('weight_scale', None)
        if bias:
            self.register_buffer('bias', None)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> 'EntropyDispatchedLinear':
        has_bias = linear.bias is not None
        layer = cls(linear.in_features, linear.out_features, bias=has_bias)
        w_int8, w_scale = quantize_weights_int8(linear.weight)
        layer.weight_int8 = w_int8
        layer.weight_scale = w_scale
        if has_bias:
            layer.bias = linear.bias.detach().to(torch.float16)
        return layer

    def forward(self, x: torch.Tensor, tier: PrecisionTier = PrecisionTier.FP16) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous().to(torch.float16)

        if tier == PrecisionTier.INT8:
            out = triton_int8_gemm(x_2d, self.weight_int8, self.weight_scale)
        elif tier == PrecisionTier.FP8:
            out = triton_fp8_gemm(x_2d, self.weight_int8, self.weight_scale)
        else:
            out = triton_fp16_gemm(x_2d, self.weight_int8, self.weight_scale)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)

        return out.reshape(orig_shape[:-1] + (self.out_features,))

    def extra_repr(self):
        return f'in={self.in_features}, out={self.out_features}, bias={self.bias is not None}'


# ============================================================================
# Llama-3 Entropy-Dispatched Transformer Block
# ============================================================================

class EntropyDispatchedLlamaBlock(nn.Module):
    def __init__(self, llama_block, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        self.self_attn = llama_block.self_attn
        self.input_layernorm = llama_block.input_layernorm
        self.post_attention_layernorm = llama_block.post_attention_layernorm

        self.gate_proj = EntropyDispatchedLinear.from_linear(llama_block.mlp.gate_proj)
        self.up_proj = EntropyDispatchedLinear.from_linear(llama_block.mlp.up_proj)
        self.down_proj = EntropyDispatchedLinear.from_linear(llama_block.mlp.down_proj)

        self.act_fn = nn.SiLU()
        self.layer_entropy: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        dispatcher: Optional[PipelinedEntropyDispatcher] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # FIX: Pass **kwargs down to catch position_embeddings from the parent LlamaModel
        attn_output, entropy_batch, present_kv = self._attention_with_fused_entropy(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )

        self.layer_entropy = entropy_batch
        if dispatcher is not None:
            dispatcher.submit_entropy(entropy_batch)

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        tier = dispatcher.get_tier() if dispatcher is not None else PrecisionTier.FP16

        gate = self.gate_proj(hidden_states, tier=tier)
        up = self.up_proj(hidden_states, tier=tier)
        hidden_states = self.down_proj(self.act_fn(gate) * up, tier=tier)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_kv,)
        return outputs

    def _attention_with_fused_entropy(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        
        attn_kwargs = dict(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,
        )
        # FIX: Inject all passed kwargs (including position_embeddings) into the Attention module args
        attn_kwargs.update(kwargs)

        # Fallback for older HF models where the attention module still computes RoPE internally
        if 'position_embeddings' not in attn_kwargs and hasattr(self.self_attn, 'rotary_emb'):
            cos, sin = self.self_attn.rotary_emb(hidden_states, position_ids)
            attn_kwargs['position_embeddings'] = (cos, sin)

        attn_outputs = self.self_attn(hidden_states, **attn_kwargs)
        
        attn_output = attn_outputs[0]
        attn_weights = attn_outputs[1]  # [B, H, S, S] post-softmax
        present_kv = attn_outputs[2] if use_cache and len(attn_outputs) > 2 else None

        eps = 1e-10
        entropy_full = -(attn_weights.float() * torch.log(attn_weights.float() + eps)).sum(dim=-1)
        entropy_batch = entropy_full.mean(dim=(1, 2))  # [B]

        return attn_output, entropy_batch, present_kv


def build_dispatched_llama(model, tau_low: float, tau_high: float):
    device = next(model.parameters()).device
    blocks = []
    for i, layer in enumerate(model.model.layers):
        block = EntropyDispatchedLlamaBlock(layer, layer_idx=i)
        block = block.to(device).half()
        block.eval()
        blocks.append(block)

    dispatcher = PipelinedEntropyDispatcher(tau_low=tau_low, tau_high=tau_high)
    return blocks, dispatcher