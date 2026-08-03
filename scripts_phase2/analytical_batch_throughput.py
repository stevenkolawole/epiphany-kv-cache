#!/usr/bin/env python3
"""
analytical_batch_throughput.py

The deployment figure: how many concurrent reasoning requests fit on one GPU,
with and without KV-cache eviction, as the trace grows.

Pure KV-cache arithmetic — no GPU, no model weights loaded. The model
architecture (layers, KV heads, head dim) is read from the HuggingFace config so
nothing is hand-entered. Per-token KV memory is

    bytes/token = 2 (key+value) * L * H_kv * d_head * dtype_bytes

The maximum batch (concurrent sequences) at a KV budget of K tokens on a GPU
with M bytes is

    batch(K) = floor( (M * util - W) / (K * bytes/token) )

where W is the model weight footprint and `util` the usable-memory fraction.
Without eviction the budget equals the full context length n; with eviction it is
fixed at the cache budget. Plotting batch against n shows the no-eviction curve
collapsing as traces lengthen while the eviction curve stays flat — the
throughput argument for cache eviction. This advantage is shared by all eviction
methods; the paper's separate contribution is obtaining it *without leaving the
FlashAttention kernel* (see the prefill-memory microbenchmark).

Usage
-----
    python scripts/analytical_batch_throughput.py \\
        --gpu_gb 80 --weight_gb 16 \\
        --budgets 2048 4096 \\
        --context_lens 4096 8192 16384 32768 65536 131072 \\
        --output reports/analytical_batch.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from transformers import AutoConfig
except ImportError:
    sys.exit("Missing dependency: pip install transformers")

_DEFAULT_HF_CACHE = "/data/hf_cache/skolawol"


def _hf_cache() -> str:
    return os.environ.get("HF_HOME", _DEFAULT_HF_CACHE)


def kv_bytes_per_token(cfg, dtype_bytes: int) -> int:
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return 2 * n_layers * n_kv * head_dim * dtype_bytes


def max_batch(budget_tokens: int, per_token: int, gpu_bytes: float,
              weight_bytes: float, util: float) -> int:
    usable = gpu_bytes * util - weight_bytes
    if usable <= 0:
        return 0
    return max(0, int(usable // (budget_tokens * per_token)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="deepseek-ai/deepseek-r1-distill-llama-8b")
    p.add_argument("--gpu_gb", type=float, default=80.0, help="GPU memory (GB)")
    p.add_argument("--weight_gb", type=float, default=16.0,
                   help="Model weight footprint (GB); ~16 for an 8B model in bf16")
    p.add_argument("--util", type=float, default=0.90,
                   help="Usable fraction of GPU memory (leaves headroom for activations)")
    p.add_argument("--dtype_bytes", type=int, default=2, help="KV dtype bytes (bf16=2)")
    p.add_argument("--budgets", type=int, nargs="+", default=[2048, 4096],
                   help="Fixed cache budgets under eviction")
    p.add_argument("--context_lens", type=int, nargs="+",
                   default=[4096, 8192, 16384, 32768, 65536, 131072])
    p.add_argument("--output", type=Path, default=Path("reports/analytical_batch.json"))
    p.add_argument("--plot", type=Path, default=Path("reports/analytical_batch.pdf"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model, cache_dir=_hf_cache())
    per_token = kv_bytes_per_token(cfg, args.dtype_bytes)
    gpu_bytes = args.gpu_gb * 1024 ** 3
    weight_bytes = args.weight_gb * 1024 ** 3

    print(f"Model: {args.model}")
    print(f"  layers={cfg.num_hidden_layers}  "
          f"kv_heads={getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)}  "
          f"head_dim={getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)}")
    print(f"  KV per token = {per_token/1024:.1f} KB  "
          f"({per_token/1024/1024:.3f} MB)")
    print(f"GPU={args.gpu_gb:.0f}GB  weights={args.weight_gb:.0f}GB  util={args.util}")

    # No-eviction: budget == full context length n.
    no_evict = {n: max_batch(n, per_token, gpu_bytes, weight_bytes, args.util)
                for n in args.context_lens}
    # Eviction: budget fixed regardless of n (n must exceed budget to apply).
    evict = {
        K: {n: max_batch(K, per_token, gpu_bytes, weight_bytes, args.util)
            for n in args.context_lens}
        for K in args.budgets
    }

    results = {
        "model": args.model,
        "kv_bytes_per_token": per_token,
        "gpu_gb": args.gpu_gb, "weight_gb": args.weight_gb, "util": args.util,
        "no_eviction_batch": no_evict,
        "eviction_batch": evict,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Table
    print(f"\n{'context':>9}  {'no-evict':>9}", end="")
    for K in args.budgets:
        print(f"  {'K='+str(K):>9}", end="")
    print()
    for n in args.context_lens:
        print(f"{n:>9}  {no_evict[n]:>9}", end="")
        for K in args.budgets:
            print(f"  {evict[K][n]:>9}", end="")
        print()

    # Plot (optional — skips cleanly if matplotlib is absent)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        xs = args.context_lens
        ax.plot(xs, [no_evict[n] for n in xs], "o-", label="no eviction")
        for K in args.budgets:
            ax.plot(xs, [evict[K][n] for n in xs], "s--", label=f"eviction K={K}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel(f"max concurrent requests\n({args.gpu_gb:.0f}GB GPU)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"\nPlot → {args.plot}")
    except ImportError:
        print("\n(matplotlib not available — JSON written, plot skipped)")

    print(f"JSON  → {args.output}")


if __name__ == "__main__":
    main()
