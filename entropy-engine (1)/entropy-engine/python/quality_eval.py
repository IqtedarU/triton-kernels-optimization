
"""
quality_eval.py — Phase 2: Perplexity vs. (tau_low, tau_high) on WikiText-2
============================================================================

Evaluates quality cost of 3-tier entropy-gated dispatch on a Llama model.

Pipeline:
  1. Load a Llama-architecture model (default: TinyLlama for quick testing).
  2. Tokenize WikiText-2 test set into fixed-length chunks.
  3. Baseline: FP16 perplexity (original model, no quantization).
  4. Entropy profile: run a few chunks to measure per-layer entropy distribution.
  5. Sweep: for each (tau_low, tau_high) pair, evaluate dispatched perplexity.
  6. Output: table showing PPL, Δ%, and per-tier dispatch ratios.
"""

import argparse
import math
import sys
import os
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dispatch import (
    EntropyDispatchedLlamaBlock,
    PipelinedEntropyDispatcher,
    build_dispatched_llama,
    PrecisionTier,
)

def load_wikitext2_chunks(tokenizer, max_samples=100, max_length=1024):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print("  Loading WikiText-2 test set...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = "\n\n".join([t for t in dataset['text'] if t.strip()])

    print("  Tokenizing...")
    token_ids = tokenizer.encode(text)
    print(f"  Total tokens: {len(token_ids):,}")

    chunks = []
    for start in range(0, len(token_ids) - 1, max_length):
        end = min(start + max_length + 1, len(token_ids))
        chunk = token_ids[start:end]
        if len(chunk) > 1:
            chunks.append(chunk)
        if len(chunks) >= max_samples:
            break

    print(f"  Chunks: {len(chunks)} × ~{max_length} tokens")
    return chunks

def load_llama_model(model_name: str, device: torch.device):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: pip install transformers")
        sys.exit(1)

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation='eager',
    ).to(device)
    model.eval()

    config = model.config
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")
    print(f"  hidden_size={config.hidden_size}, intermediate={config.intermediate_size}")
    print(f"  layers={config.num_hidden_layers}, heads={config.num_attention_heads}")
    if hasattr(config, 'num_key_value_heads'):
        print(f"  KV heads={config.num_key_value_heads} (GQA)")

    return model, tokenizer

def eval_baseline_fp16(model, chunks, device):
    total_loss = 0.0
    total_tokens = 0

    for chunk in chunks:
        input_ids = torch.tensor([chunk[:-1]], device=device, dtype=torch.long)
        target_ids = torch.tensor([chunk[1:]], device=device, dtype=torch.long)

        with torch.no_grad():
            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                target_ids.view(-1),
                reduction='sum',
            )
            total_loss += loss.item()
            total_tokens += target_ids.numel()

    return math.exp(total_loss / total_tokens)

def eval_dispatched(model, blocks, dispatcher, chunks, device):
    total_loss = 0.0
    total_tokens = 0

    for chunk in chunks:
        input_ids = torch.tensor([chunk[:-1]], device=device, dtype=torch.long)
        target_ids = torch.tensor([chunk[1:]], device=device, dtype=torch.long)
        B, S = input_ids.shape

        dispatcher.reset_stats()

        with torch.no_grad():
            hidden = model.model.embed_tokens(input_ids)
            hidden = hidden.to(torch.float16)

            position_ids = torch.arange(S, device=device).unsqueeze(0)
            position_embeddings = model.model.rotary_emb(hidden, position_ids)

            # --- FIX: CAUSAL MASK ---
            causal_mask = torch.full((1, 1, S, S), torch.finfo(torch.float16).min, device=device)
            causal_mask = torch.triu(causal_mask, diagonal=1)

            for block in blocks:
                outputs = block(
                    hidden,
                    dispatcher=dispatcher,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask # <-- Pass it here!
                )
                hidden = outputs[0]

            hidden = model.model.norm(hidden)
            logits = model.lm_head(hidden)

            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                target_ids.view(-1),
                reduction='sum',
            )
            total_loss += loss.item()
            total_tokens += target_ids.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    stats = dispatcher.get_stats()
    return ppl, stats

def collect_entropy_profile(model, blocks, chunks, device, n_chunks=5):
    profiler = PipelinedEntropyDispatcher(tau_low=999.0, tau_high=999.0)
    layer_entropies = {i: [] for i in range(len(blocks))}

    for chunk in chunks[:n_chunks]:
        input_ids = torch.tensor([chunk[:-1]], device=device, dtype=torch.long)
        B, S = input_ids.shape
        hidden = model.model.embed_tokens(input_ids).to(torch.float16)
        
        position_ids = torch.arange(S, device=device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)

        # --- FIX: CAUSAL MASK ---
        causal_mask = torch.full((1, 1, S, S), torch.finfo(torch.float16).min, device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        with torch.no_grad():
            for i, block in enumerate(blocks):
                outputs = block(
                    hidden, 
                    dispatcher=profiler, 
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask # <-- Pass it here!
                )
                hidden = outputs[0]
                if block.layer_entropy is not None:
                    layer_entropies[i].append(block.layer_entropy.max().item())

    return layer_entropies

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Quality Evaluation")
    parser.add_argument('--model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Phase 2: 3-Tier Quality Evaluation (Llama + WikiText) ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    model, tokenizer = load_llama_model(args.model, device)
    chunks = load_wikitext2_chunks(tokenizer, args.max_samples, args.max_length)
    print()

    print("─── BASELINE: FP16 (original model) ────────────────────────")
    t0 = time.time()
    ppl_fp16 = eval_baseline_fp16(model, chunks, device)
    t_fp16 = time.time() - t0
    print(f"  Perplexity: {ppl_fp16:.2f}")
    print(f"  Time: {t_fp16:.1f}s")
    print()

    print("─── ENTROPY PROFILE ────────────────────────────────────────")
    profile_blocks, _ = build_dispatched_llama(model, tau_low=999.0, tau_high=999.0)
    layer_entropies = collect_entropy_profile(model, profile_blocks, chunks, device)
    del profile_blocks
    torch.cuda.empty_cache()

    n_layers = len(layer_entropies)
    all_entropies = []
    print(f"  {'Layer':<8} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for i in range(n_layers):
        vals = layer_entropies[i]
        if vals:
            all_entropies.extend(vals)
            print(f"  {i:<8} {sum(vals)/len(vals):>8.3f} {min(vals):>8.3f} {max(vals):>8.3f}")

    print()
    if all_entropies:
        sorted_e = sorted(all_entropies)
        p25 = sorted_e[len(sorted_e) // 4]
        p50 = sorted_e[len(sorted_e) // 2]
        p75 = sorted_e[3 * len(sorted_e) // 4]
        print(f"  Percentiles: p25={p25:.3f}  p50={p50:.3f}  p75={p75:.3f}")
    print()

    if all_entropies:
        e_min = min(all_entropies)
        e_max = max(all_entropies)
        sweep_pairs = [
            (round(e_min * 0.5, 2), round(p25, 2)),       
            (round(p25, 2), round(p50, 2)),                
            (round(p50, 2), round(p75, 2)),                
            (round(p75, 2), round(e_max * 1.1, 2)),        
            (round(e_max * 1.2, 2), round(e_max * 1.5, 2)),
        ]
        sweep_pairs = [(lo, hi) for lo, hi in sweep_pairs if lo < hi]
    else:
        sweep_pairs = [(1.0, 2.0), (2.0, 3.0), (2.5, 3.5), (3.0, 4.0), (4.0, 5.0)]

    print("─── PERPLEXITY vs (tau_low, tau_high) ──────────────────────")
    print()
    header = f"  {'tau_low':>7} {'tau_high':>8} │ {'PPL':>9} {'Δ%':>7} │ {'INT8%':>6} {'FP8%':>6} {'FP16%':>6} │ {'Time':>6}"
    print(header)
    print(f"  {'─'*7} {'─'*8} ┼ {'─'*9} {'─'*7} ┼ {'─'*6} {'─'*6} {'─'*6} ┼ {'─'*6}")


    # Tell PyTorch to fuse the Python loops and dispatch logic into C++
    print("  Compiling model graph to C++ (this will take a minute on the first run)...")
    model = torch.compile(model, mode="reduce-overhead")
    
    results = []
    for tau_low, tau_high in sweep_pairs:
        blocks, dispatcher = build_dispatched_llama(model, tau_low, tau_high)
        t0 = time.time()
        ppl, stats = eval_dispatched(model, blocks, dispatcher, chunks, device)
        elapsed = time.time() - t0

        delta_pct = (ppl - ppl_fp16) / ppl_fp16 * 100

        results.append({
            'tau_low': tau_low, 'tau_high': tau_high, 'ppl': ppl,
            'delta_pct': delta_pct, 'stats': stats, 'time': elapsed,
        })

        int8_p = stats['INT8']['pct']
        fp8_p = stats['FP8']['pct']
        fp16_p = stats['FP16']['pct']

        print(
            f"  {tau_low:>7.2f} {tau_high:>8.2f} │ {ppl:>9.2f} {delta_pct:>+6.2f}% │"
            f" {int8_p:>4.1f}% {fp8_p:>4.1f}% {fp16_p:>4.1f}% │ {elapsed:>4.1f}s"
        )

        del blocks, dispatcher
        torch.cuda.empty_cache()

    print(f"  {'─'*7} {'─'*8} ┼ {'─'*9} {'─'*7} ┼ {'─'*6} {'─'*6} {'─'*6} ┼ {'─'*6}")
    print(f"  {'FP16':>7} {'base':>8} │ {ppl_fp16:>9.2f} {'±0.00%':>7} │ {'0.0%':>6} {'0.0%':>6} {'100%':>6} │ {t_fp16:>4.1f}s")
    print()

    if results:
        acceptable = [r for r in results if r['delta_pct'] < 1.0]
        if acceptable:
            best = max(acceptable, key=lambda r: r['stats']['INT8']['pct'] + r['stats']['FP8']['pct'])
            print(f"  Recommended: tau_low={best['tau_low']}, tau_high={best['tau_high']}")
            print(f"    PPL: {best['ppl']:.2f} (Δ{best['delta_pct']:+.2f}% vs FP16)")
            print(f"    INT8: {best['stats']['INT8']['pct']:.1f}%, FP8: {best['stats']['FP8']['pct']:.1f}%, FP16: {best['stats']['FP16']['pct']:.1f}%")
        else:
            print("  No threshold pair achieves < 1% PPL degradation.")
            print("  Try higher thresholds or check model calibration.")
    print()

if __name__ == '__main__':
    main()