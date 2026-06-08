#!/usr/bin/env python3
"""
bench_prefill_memory.py

The memory figure for the paper: peak GPU memory of a single forward pass as a
function of context length, comparing the two scoring regimes.

  eager_attn : attn_implementation="eager", output_attentions=True
               — what any attention-based eviction method (H2O/ThinKV/RaaS) must
                 request. Materialises the (batch, heads, n, n) attention tensor
                 in HBM → quadratic memory, OOM at long context.

  flash      : attn_implementation="flash_attention_2", output_hidden_states=True
               — our FA2-compatible path. Reads hidden states / cached KV only;
                 never materialises the attention matrix → linear memory.

This is forward-only (no generation), so it is cheap — a few minutes total.

Usage
-----
    python scripts/bench_prefill_memory.py \\
        --seq_lens 4096 8192 16384 32768 65536 \\
        --batch_sizes 1 \\
        --output reports/prefill_memory.json

    # also show batch scaling (forward-only, still cheap)
    python scripts/bench_prefill_memory.py --seq_lens 4096 8192 16384 --batch_sizes 1 2 4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

try:
    from transformers import AutoModelForCausalLM, AutoConfig
except ImportError:
    sys.exit("Missing dependency: pip install transformers")

_DEFAULT_HF_CACHE = "/data/hf_cache/skolawol"


def _hf_cache() -> str:
    return os.environ.get("HF_HOME", _DEFAULT_HF_CACHE)


def load_model(model_id: str, mode: str):
    impl = "eager" if mode == "eager_attn" else "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": 0},          # single GPU — matches the Phase 1 flash setup
        attn_implementation=impl,
        cache_dir=_hf_cache(),
    )
    model.eval()
    return model


def measure(model, vocab_size: int, batch: int, seq_len: int, mode: str, device) -> dict:
    """One forward pass; return peak GPU MB, or OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    want_attn = mode == "eager_attn"
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                use_cache=False,
                output_attentions=want_attn,
                output_hidden_states=not want_attn,
            )
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        result = {"peak_gpu_mb": round(peak_mb, 1), "oom": False}
    except torch.cuda.OutOfMemoryError:
        result = {"peak_gpu_mb": None, "oom": True}
    finally:
        del input_ids
        torch.cuda.empty_cache()
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="deepseek-ai/deepseek-r1-distill-llama-8b")
    p.add_argument("--seq_lens", type=int, nargs="+",
                   default=[4096, 8192, 16384, 32768, 65536])
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1])
    p.add_argument("--modes", nargs="+", default=["eager_attn", "flash"],
                   choices=["eager_attn", "flash"])
    p.add_argument("--output", type=Path, default=Path("reports/prefill_memory.json"))
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — run on a GPU node.")
    device = torch.device("cuda:0")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model, cache_dir=_hf_cache())
    vocab_size = cfg.vocab_size
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    print(f"GPU: {torch.cuda.get_device_name(device)} ({total_gb:.0f} GB)")

    results = {}
    for mode in args.modes:
        print(f"\nLoading model (mode={mode}) ...")
        model = load_model(args.model, mode)
        results[mode] = {}
        for batch in args.batch_sizes:
            for seq_len in args.seq_lens:
                r = measure(model, vocab_size, batch, seq_len, mode, device)
                tag = f"b{batch}_n{seq_len}"
                results[mode][tag] = {"batch": batch, "seq_len": seq_len, **r}
                status = "OOM" if r["oom"] else f"{r['peak_gpu_mb']:.0f} MB"
                print(f"  {mode:10s}  batch={batch}  n={seq_len:>6}  →  {status}")
        del model
        torch.cuda.empty_cache()

    with open(args.output, "w") as f:
        json.dump({"model": args.model, "gpu_gb": round(total_gb, 1),
                   "results": results}, f, indent=2)
    print(f"\nWritten to {args.output}")

    # Compact table
    print(f"\n{'seq_len':>8}", end="")
    for mode in args.modes:
        print(f"  {mode:>12}", end="")
    print()
    for batch in args.batch_sizes:
        for seq_len in args.seq_lens:
            print(f"{seq_len:>8}", end="")
            for mode in args.modes:
                r = results[mode].get(f"b{batch}_n{seq_len}", {})
                cell = "OOM" if r.get("oom") else f"{r.get('peak_gpu_mb', 0):.0f}MB"
                print(f"  {cell:>12}", end="")
            print(f"   (batch={batch})")


if __name__ == "__main__":
    main()
