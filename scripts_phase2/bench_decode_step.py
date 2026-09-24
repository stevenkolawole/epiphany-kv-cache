#!/usr/bin/env python3
"""Per-step decode latency against cache length, eager attention (with the attention
weights returned, as an attention-based policy needs them) against FlashAttention, at
several batch sizes. The cache is built by chunked prefill; each single-token step is
timed after one warm-up (median of five). Companion of bench_prefill_memory.py."""
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

CONFIGS = [(1, 1024), (1, 2048), (1, 4096), (1, 8192), (1, 16384),
           (8, 1024), (8, 2048), (8, 4096), (8, 8192),
           (32, 1024), (32, 2048), (32, 4096)]

def load(model_id, mode):
    impl = "eager" if mode == "eager_attn" else "flash_attention_2"
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                             attn_implementation=impl, device_map="cuda")
    m.eval()
    return m

def measure(model, batch, n, want_attn, chunk=1024):
    V = model.config.vocab_size
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            pkv = None
            ids = torch.randint(0, V, (batch, n - 1), device="cuda")
            for s in range(0, n - 1, chunk):
                o = model(input_ids=ids[:, s:s + chunk], past_key_values=pkv, use_cache=True, logits_to_keep=1)
                pkv = o.past_key_values
            del ids
            times = []
            for rep in range(6):
                nxt = torch.randint(0, V, (batch, 1), device="cuda")
                torch.cuda.synchronize(); t0 = time.perf_counter()
                o = model(input_ids=nxt, past_key_values=pkv, use_cache=True, output_attentions=want_attn)
                torch.cuda.synchronize()
                if rep:
                    times.append(time.perf_counter() - t0)
                pkv = o.past_key_values
        times.sort()
        r = {"step_ms": round(1000 * times[2], 3), "step_ms_all": [round(1000 * t, 3) for t in times],
             "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), "oom": False}
    except torch.cuda.OutOfMemoryError:
        r = {"step_ms": None, "oom": True}
    torch.cuda.empty_cache()
    return r

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-ai/deepseek-r1-distill-llama-8b")
    p.add_argument("--modes", nargs="+", default=["eager_attn", "flash"])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = {"model": a.model, "gpu": torch.cuda.get_device_name(0),
           "gpu_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1), "results": {}}
    for mode in a.modes:
        model = load(a.model, mode)
        out["results"][mode] = {}
        for batch, n in CONFIGS:
            r = measure(model, batch, n, want_attn=(mode == "eager_attn"))
            out["results"][mode][f"b{batch}_n{n}"] = {"batch": batch, "seq_len": n, **r}
            print(f"  {mode:10s} batch={batch:2d} n={n:6d} -> {r}", flush=True)
            a.output.parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(a.output, "w"), indent=1)
        del model; torch.cuda.empty_cache()
    print("Written to", a.output)
