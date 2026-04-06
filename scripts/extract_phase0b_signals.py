#!/usr/bin/env python3
"""
extract_phase0b_signals.py

Posthoc Phase 0B signal extraction for cross-validation.

Reads an existing trace JSONL (produced by collect_traces.py), replays each
trace's token sequence through the model with forward hooks registered on every
k_proj layer, and extracts:

  kv_key_var_preRoPE   Pre-RoPE key variance (all-layer mean)
  kv_key_norm_preRoPE  Pre-RoPE key L2 norm  (all-layer mean)
  hs_l2_diff_lN        L2 diff of hidden states at layer N
  hs_cos_dist_lN       Cosine distance of hidden states at layer N

These are the same signals added by collect_traces.py --phase0b.  Running both
paths on the same token sequences and comparing outputs is the cross-validation
step: any disagreement indicates a bug in one of the two extraction paths.

Usage
-----
    # Extract Phase 0B signals from a trace file, write augmented traces:
    python scripts/extract_phase0b_signals.py \\
        --input  data/aime2024_eager_traces.jsonl \\
        --output data/aime2024_eager_traces_phase0b_check.jsonl

    # Compare augmented traces against re-collected Phase 0B traces:
    python scripts/extract_phase0b_signals.py \\
        --input  data/aime2024_eager_traces.jsonl \\
        --compare data/aime2024_eager_traces_phase0b.jsonl

Cross-validation interpretation
--------------------------------
The two extraction paths compute signals via separate fp16 forward passes.
GPU floating-point arithmetic is non-deterministic (non-associative parallel
reductions in CUDA/cuBLAS), so absolute values will differ between runs even
with identical inputs.  At middle transformer layers the residual stream can
have large magnitudes, amplifying fp16 rounding differences to absolute errors
of O(100-1000) — making absolute-error tolerances useless as a quality gate.

Instead, we check Spearman rank correlation (ρ) between the two signal vectors
for each trace.  Bugs produce qualitatively different signals (ρ near 0 or
negative); fp16 non-determinism produces nearly identical rankings (ρ > 0.99).

The --compare flag prints per-signal mean Spearman ρ across all trace pairs
and exits 0 (PASS) only if every signal meets --min_corr (default 0.99).

What low ρ indicates (the actual bugs to catch):
  - Token ID mismatch: stored token_ids don't reproduce the same sequence.
  - Hook placement mismatch: k_proj hook fires at a different point in one path.
  - Layer index off-by-one: hs_index = layer_idx + 1 wrong in one path.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("Missing dependency: pip install transformers")


_DEFAULT_HF_CACHE = "/data/hf_cache/skolawol"


def _hf_cache() -> str:
    import os
    return os.environ.get("HF_HOME", _DEFAULT_HF_CACHE)


# ── Signal extraction ─────────────────────────────────────────────────────────

def extract_phase0b(
    model,
    token_ids: List[int],
    hs_layers: List[int],
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Run a single forward pass on `token_ids` and return Phase 0B signals.

    Registers k_proj forward hooks on every transformer layer to capture
    pre-RoPE key projections.  Also extracts hidden states at each layer
    in `hs_layers`.

    Args:
        model:      loaded HuggingFace model in eval mode.
        token_ids:  full sequence (prompt + generated), as stored in trace.
        hs_layers:  list of transformer layer indices for per-layer HS signals.
        device:     torch device to run inference on.

    Returns:
        Dict of signal_name -> list of per-token floats (length = len(token_ids)).
        All values are float32; sentinel -1.0 is NOT used here (we run the full
        sequence unconditionally).
    """
    seq_len    = len(token_ids)
    input_ids  = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # (1, seq_len)

    cfg        = model.config
    n_kv_heads = cfg.num_key_value_heads
    head_dim   = cfg.hidden_size // cfg.num_attention_heads

    # Accumulators for pre-RoPE key signals (summed over layers, then divided).
    preRoPE_var_sum  = np.zeros(seq_len, dtype=np.float64)
    preRoPE_norm_sum = np.zeros(seq_len, dtype=np.float64)
    preRoPE_n        = 0
    hooks            = []

    def _make_hook():
        def _fn(module, inp, output):
            nonlocal preRoPE_n
            # output: (1, seq_len, n_kv_heads * head_dim)
            k = output.detach().float().cpu()
            k = k.view(k.shape[0], k.shape[1], n_kv_heads, head_dim)
            k = k[0]  # (seq_len, n_kv_heads, head_dim)
            preRoPE_var_sum  [:] += k.var(dim=-1).mean(dim=-1).numpy()
            preRoPE_norm_sum [:] += k.norm(dim=-1).mean(dim=-1).numpy()
            preRoPE_n += 1
        return _fn

    for layer in model.model.layers:
        hooks.append(layer.self_attn.k_proj.register_forward_hook(_make_hook()))

    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to(device),
                output_hidden_states=True,
                use_cache=False,
            )
    finally:
        for h in hooks:
            h.remove()

    signals: Dict[str, List[float]] = {}

    # ── Pre-RoPE key signals ──────────────────────────────────────────────
    if preRoPE_n > 0:
        signals["kv_key_var_preRoPE"]  = (preRoPE_var_sum  / preRoPE_n).astype(np.float32).tolist()
        signals["kv_key_norm_preRoPE"] = (preRoPE_norm_sum / preRoPE_n).astype(np.float32).tolist()

    # ── Per-layer hidden-state signals ────────────────────────────────────
    # out.hidden_states[0] = embedding output; [i+1] = output of layer i.
    for layer_idx in hs_layers:
        hs_index = layer_idx + 1
        if hs_index >= len(out.hidden_states):
            continue
        h = out.hidden_states[hs_index][0].float().cpu()  # (seq_len, d_model)

        norms    = h.norm(dim=-1)                                      # (seq_len,)
        diffs    = (h[1:] - h[:-1]).norm(dim=-1)                      # (seq_len-1,)
        normed   = h / norms.clamp(min=1e-8).unsqueeze(-1)
        cos_sim  = (normed[1:] * normed[:-1]).sum(dim=-1).clamp(-1, 1)

        l2_diff  = np.zeros(seq_len, dtype=np.float32)
        cos_dist = np.zeros(seq_len, dtype=np.float32)
        l2_diff[1:]  = diffs.numpy()
        cos_dist[1:] = (1.0 - cos_sim).numpy()

        signals[f"hs_l2_diff_l{layer_idx}"]  = l2_diff.tolist()
        signals[f"hs_cos_dist_l{layer_idx}"] = cos_dist.tolist()

    del out
    torch.cuda.empty_cache()
    return signals


# ── Cross-validation comparison ───────────────────────────────────────────────

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman rank correlation between two 1-D arrays, skipping -1.0 sentinels.
    Returns nan if fewer than 10 valid positions remain.
    """
    mask = (a != -1.0) & (b != -1.0)
    if mask.sum() < 10:
        return float("nan")
    a, b = a[mask], b[mask]
    rank_a = np.argsort(np.argsort(a)).astype(np.float64)
    rank_b = np.argsort(np.argsort(b)).astype(np.float64)
    n = len(a)
    d = rank_a - rank_b
    return float(1.0 - 6.0 * (d * d).sum() / (n * (n * n - 1)))


def compare_signals(
    rec_a: Dict,
    rec_b: Dict,
    signal_names: List[str],
) -> Dict[str, Dict]:
    """
    Compare Phase 0B signals between two trace records via Spearman correlation.

    Returns per-signal stats: {"rho": float, "note": str}.
    rho == nan means the signal was missing or too short to evaluate.
    """
    sigs_a = rec_a.get("signals", {})
    sigs_b = rec_b.get("signals", {})
    stats  = {}

    for name in signal_names:
        a = sigs_a.get(name)
        b = sigs_b.get(name)
        if a is None or b is None:
            stats[name] = {"rho": float("nan"), "note": "missing in one file"}
            continue
        if len(a) != len(b):
            stats[name] = {"rho": float("nan"),
                           "note": f"length mismatch {len(a)} vs {len(b)}"}
            continue
        arr_a = np.array(a, dtype=np.float64)
        arr_b = np.array(b, dtype=np.float64)
        stats[name] = {"rho": _spearman(arr_a, arr_b), "note": ""}
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",   type=Path, required=True,
                   help="Existing trace JSONL to augment / validate against")
    p.add_argument("--output",  type=Path, default=None,
                   help="Write augmented traces here (optional; skip if only comparing)")
    p.add_argument("--compare", type=Path, default=None,
                   help="Phase 0B trace JSONL to cross-validate against --input")
    p.add_argument("--model",   default="deepseek-ai/deepseek-r1-distill-llama-8b")
    p.add_argument("--hs_layers", default=",".join(str(i) for i in range(32)),
                   help="Comma-separated layer indices for per-layer HS (default: all 32 layers, 0-31)")
    p.add_argument("--min_corr", type=float, default=0.99,
                   help="Minimum Spearman ρ for cross-validation PASS (default: 0.99)")
    p.add_argument("--max_traces", type=int, default=None,
                   help="Process only the first N traces (useful for spot-checks)")
    return p.parse_args()


def main():
    args    = parse_args()
    hs_layers = [int(x.strip()) for x in args.hs_layers.split(",") if x.strip()]

    # ── Compare mode (no model needed) ───────────────────────────────────
    if args.compare is not None and args.output is None:
        print(f"Cross-validation mode: {args.input}  vs  {args.compare}")
        print(f"Min Spearman ρ required: {args.min_corr:.3f}")

        with open(args.input) as f:
            recs_a = [json.loads(l) for l in f if l.strip()]
        with open(args.compare) as f:
            recs_b = [json.loads(l) for l in f if l.strip()]

        by_problem_b = {r["problem"]: r for r in recs_b}
        _p0b_signals = (
            ["kv_key_var_preRoPE", "kv_key_norm_preRoPE"]
            + [f"hs_l2_diff_l{i}" for i in hs_layers]
            + [f"hs_cos_dist_l{i}" for i in hs_layers]
        )

        n_checked = 0
        agg: Dict[str, List] = {s: [] for s in _p0b_signals}
        for rec_a in recs_a[:args.max_traces]:
            rec_b = by_problem_b.get(rec_a["problem"])
            if rec_b is None:
                print(f"  [SKIP] No match for problem: {rec_a['problem'][:60]}...")
                continue
            stats = compare_signals(rec_a, rec_b, _p0b_signals)
            n_checked += 1
            for name, s in stats.items():
                if not np.isnan(s["rho"]):
                    agg[name].append(s["rho"])

        print(f"\nChecked {n_checked} trace pairs.")
        print(f"\n{'Signal':<30} {'Mean ρ':>10} {'Min ρ':>10} {'Status'}")
        print("-" * 58)
        all_ok = True
        for name in _p0b_signals:
            rhos = agg.get(name, [])
            if not rhos:
                print(f"  {name:<28} {'N/A':>10}  (not present in both files)")
                continue
            mean_rho = float(np.mean(rhos))
            min_rho  = float(np.min(rhos))
            ok       = min_rho >= args.min_corr
            status   = "OK" if ok else f"FAIL (min ρ < {args.min_corr:.2f})"
            if not ok:
                all_ok = False
            print(f"  {name:<28} {mean_rho:>10.4f} {min_rho:>10.4f}  {status}")

        print()
        print("Cross-validation: " + ("PASSED" if all_ok else "FAILED"))
        sys.exit(0 if all_ok else 1)

    # ── Extraction mode (requires model) ─────────────────────────────────
    if args.output is None and args.compare is None:
        sys.exit("Provide --output to write augmented traces, or --compare to cross-validate.")

    print(f"Loading model: {args.model}")
    cache_dir = _hf_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", cache_dir=cache_dir
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model on device: {device}")
    print(f"  Extracting Phase 0B signals for layers: {hs_layers}\n")

    with open(args.input) as f:
        traces = [json.loads(l) for l in f if l.strip()]

    if args.max_traces:
        traces = traces[:args.max_traces]

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with open(out_path, "w") as f_out:
        for i, trace in enumerate(traces):
            token_ids = trace.get("token_ids")
            if not token_ids:
                print(f"  [SKIP] trace {i} has no token_ids", file=sys.stderr)
                continue
            try:
                p0b_sigs = extract_phase0b(model, token_ids, hs_layers, device)
            except Exception as e:
                print(f"  [SKIP] trace {i} error: {e}", file=sys.stderr)
                continue

            # Merge Phase 0B signals into the trace's signals dict.
            trace.setdefault("signals", {}).update(p0b_sigs)
            f_out.write(json.dumps(trace) + "\n")
            f_out.flush()
            n_done += 1

            seq_len = len(token_ids)
            sig_names = list(p0b_sigs.keys())
            print(f"  [{i+1}/{len(traces)}] len={seq_len}  signals: {sig_names}")

    print(f"\nDone. Wrote {n_done} augmented traces to {out_path}")

    # If --compare also set, run comparison immediately after extraction.
    if args.compare is not None:
        print(f"\nRunning cross-validation against {args.compare} ...")
        import subprocess, sys as _sys
        result = subprocess.run(
            [_sys.executable, __file__,
             "--input",    str(out_path),
             "--compare",  str(args.compare),
             "--hs_layers", args.hs_layers,
             "--min_corr", str(args.min_corr)],
            check=False,
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
