#!/usr/bin/env python3
"""
signal_ablation.py

Evaluates every collected signal variant against counterfactual importance
labels (produced by label_importance.py) and reports Spearman rank
correlations.  This is the core Phase 0 experiment: does any residual-stream
or KV-vector signal outperform cumulative attention (H2O) as a proxy for
token importance?

Input
-----
  --traces   JSONL from collect_traces.py    (has 'signals' field)
  --labels   JSONL from label_importance.py  (has 'importance' field)

  Traces and labels are joined by matching 'problem' text.  Alternatively,
  if the files are in the same order and have the same problems, use
  --assume_aligned to skip the join and match by position.

Output
------
  Console table of Spearman ρ and p-value per signal, sorted by ρ.
  Optional --output CSV for downstream plotting.

Signals evaluated
-----------------
  Dimension 1 (signal type) — all collected in collect_traces.py:
    kv_key_var      KV key variance (post-RoPE), mean over heads+layers
    kv_key_norm     KV key L2 norm (post-RoPE), mean over heads+layers
    kv_val_var      KV value variance, mean over heads+layers
    cross_head_var  Cross-head key variance, mean over layers
    h2o_attn        H2O cumulative attention (attention baseline, SOTA)
    attn_entropy    Attention entropy (ThinKV baseline; requires eager attn)
    hs_l2_diff      L2 norm of consecutive hidden-state diffs (post-hoc)
    hs_cos_dist     Cosine distance between consecutive hidden states (post-hoc)
    hs_norm         L2 norm of hidden state (post-hoc)

  Derived variants (Dimension 4 — temporal aggregation) computed on-the-fly:
    kv_key_var_rolling64  Rolling mean of kv_key_var over past 64 tokens
    kv_key_var_ema09      EMA (α=0.9) of kv_key_var
    kv_val_var_rolling64  Rolling mean of kv_val_var over past 64 tokens
    kv_val_var_ema09      EMA (α=0.9) of kv_val_var
    hs_l2_diff_rolling64  Rolling mean of hs_l2_diff over past 64 tokens
    hs_l2_diff_ema09      EMA (α=0.9) of hs_l2_diff

  Note: cumsum was removed — cumsum of non-negative signals is monotonically
  increasing, making ranks equivalent to sequence position (a position proxy,
  not a signal discriminator). rolling64 tests "sustained elevated signal"
  without that artifact.

  NOTE: Pre-RoPE (Dim 2) requires a forward hook and is NOT computed here.
  Layer-wise ablation (Dim 3) requires per-layer signals — not stored in
  current traces. These are deferred to a separate ablation pass.

Correlation metric
------------------
  Spearman rank correlation between signal values and binary importance labels
  (1 = important, 0 = not important) over all token positions across all
  traces.  Only positions with label ∈ {0, 1} are included (label=-1
  excluded).  Prompt positions (always label=1) are excluded from the
  correlation since they are structurally important, not signal-driven.

Usage
-----
    python scripts/signal_ablation.py \\
        --traces data/math500_traces.jsonl \\
        --labels data/math500_traces_labelled.jsonl

    python scripts/signal_ablation.py \\
        --traces data/math500_traces.jsonl \\
        --labels data/math500_traces_labelled.jsonl \\
        --output results/signal_ablation.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:
    sys.exit("Missing dependency: pip install scipy")


# ── Derived signal helpers ────────────────────────────────────────────────────

def ema(values: List[float], alpha: float = 0.9) -> List[float]:
    """Exponential moving average.  score[t] = alpha*score[t-1] + (1-alpha)*x[t]."""
    out = []
    s = 0.0
    for x in values:
        s = alpha * s + (1.0 - alpha) * x
        out.append(s)
    return out


def rolling_mean(values: List[float], window: int = 64) -> List[float]:
    """
    Rolling mean over the past `window` positions.
    Partial windows at the start use all available values.
    NaN positions are excluded from the window average.

    Replaces cumsum: cumsum of non-negative signals is monotonically increasing
    (ranks ≡ sequence position), making it a position proxy rather than a signal
    discriminator. Rolling mean tests "sustained elevated signal" without that
    artifact. Window=64 ≈ 2× the masking window used in label_importance.
    """
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        valid = [v for v in values[start : i + 1] if not np.isnan(v)]
        out.append(float(np.mean(valid)) if valid else float("nan"))
    return out


def derive_signals(signals: Dict[str, Optional[List[float]]]) -> Dict[str, List[float]]:
    """
    Expand raw collected signals with derived temporal variants.
    Returns a dict of signal_name -> list of per-token floats.
    Signals that are None (not collected, e.g. attn signals without eager attn)
    are skipped.
    """
    derived = {}
    for name, vals in signals.items():
        if vals is None:
            continue
        # Replace any -1.0 sentinels (hs_* on long sequences) with NaN
        # so they don't contaminate the correlation.
        vals_clean = [float("nan") if v == -1.0 else v for v in vals]
        derived[name] = vals_clean

    # Temporal variants: rolling mean (window=64) and EMA (α=0.9)
    # Both test whether temporal aggregation improves signal discriminability.
    # rolling64: "sustained elevated signal over past 64 tokens"
    # ema09:     exponentially weighted recent history (α=0.9 = heavy recency bias)
    for base in ("kv_key_var", "kv_val_var", "hs_l2_diff"):
        if base in derived:
            v = derived[base]
            # Replace NaN with 0 for temporal smoothing (treat unknown as no-signal)
            v_filled = [0.0 if np.isnan(x) else x for x in v]
            derived[f"{base}_rolling64"] = rolling_mean(v_filled, window=64)
            derived[f"{base}_ema09"]     = ema(v_filled, alpha=0.9)

    return derived


# ── Data loading and joining ──────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def join_traces_labels(
    traces: List[Dict],
    labels: List[Dict],
    assume_aligned: bool,
) -> List[Tuple[Dict, Dict]]:
    """
    Return list of (trace, label) pairs.
    If assume_aligned: pair by position (faster, requires same order).
    Otherwise: join by matching 'problem' field.
    """
    if assume_aligned:
        n = min(len(traces), len(labels))
        if len(traces) != len(labels):
            print(
                f"  Warning: {len(traces)} traces vs {len(labels)} labels — "
                f"using first {n} aligned pairs.",
                file=sys.stderr,
            )
        return list(zip(traces[:n], labels[:n]))

    label_by_problem = {l["problem"]: l for l in labels}
    pairs = []
    skipped = 0
    for t in traces:
        l = label_by_problem.get(t["problem"])
        if l is None:
            skipped += 1
            continue
        pairs.append((t, l))
    if skipped:
        print(f"  Warning: {skipped} traces had no matching label (skipped).", file=sys.stderr)
    return pairs


# ── Correlation computation ───────────────────────────────────────────────────

def collect_signal_label_pairs(
    pairs: List[Tuple[Dict, Dict]],
) -> Tuple[Dict[str, List[float]], List[int]]:
    """
    Flatten all (signal_value, importance_label) pairs across all traces and
    all token positions.

    Returns:
        signal_arrays: dict of signal_name -> flat list of floats
        label_array:   flat list of int (0 or 1), same length as each signal list
    """
    # First pass: collect all signal names from first trace with signals
    all_signal_names = set()
    for t, l in pairs:
        if "signals" in t:
            all_signal_names.update(derive_signals(t["signals"]).keys())
            break

    signal_arrays: Dict[str, List[float]] = {name: [] for name in all_signal_names}
    label_array: List[int] = []

    for trace, label_rec in pairs:
        if "signals" not in trace or "importance" not in label_rec:
            continue

        importance = label_rec["importance"]
        prompt_len = trace["prompt_len"]
        total_len  = len(trace["token_ids"])

        derived = derive_signals(trace["signals"])

        # Validate signal lengths
        for name, vals in derived.items():
            if len(vals) != total_len:
                # Length mismatch — skip this trace for this signal
                derived[name] = None  # type: ignore

        for pos in range(prompt_len, total_len):  # skip prompt (always label=1, trivial)
            label = importance[pos] if pos < len(importance) else -1
            if label not in (0, 1):
                continue   # unknown position

            label_array.append(label)
            for name in list(signal_arrays.keys()):
                vals = derived.get(name)
                if vals is None or pos >= len(vals) or np.isnan(vals[pos]):
                    signal_arrays[name].append(float("nan"))
                else:
                    signal_arrays[name].append(vals[pos])

    return signal_arrays, label_array


def compute_correlations(
    signal_arrays: Dict[str, List[float]],
    label_array: List[int],
) -> List[Dict]:
    """
    Compute Spearman ρ between each signal and the binary importance labels.
    Pairs where the signal is NaN are excluded per-signal.
    Returns list of dicts sorted by |ρ| descending.
    """
    labels = np.array(label_array, dtype=np.float64)
    results = []

    for name, vals in signal_arrays.items():
        v = np.array(vals, dtype=np.float64)
        valid = ~np.isnan(v)
        n_valid = valid.sum()

        if n_valid < 10:
            results.append({
                "signal": name, "spearman_rho": float("nan"),
                "p_value": float("nan"), "n_pairs": int(n_valid),
                "note": "too few valid pairs",
            })
            continue

        rho, pval = spearmanr(v[valid], labels[valid])
        note = ""
        if name.endswith("_rolling64"):
            note = "rolling mean (window=64)"
        results.append({
            "signal":       name,
            "spearman_rho": float(rho),
            "p_value":      float(pval),
            "n_pairs":      int(n_valid),
            "note":         note,
        })

    results.sort(key=lambda x: abs(x["spearman_rho"]) if not np.isnan(x["spearman_rho"]) else -1, reverse=True)
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_table(results: List[Dict], n_traces: int, n_tokens: int):
    print(f"\n{'='*70}")
    print(f"Signal Ablation Results — {n_traces} traces, {n_tokens} labelled token positions")
    print(f"Metric: Spearman ρ with counterfactual importance labels")
    print(f"(prompt tokens excluded; only generated tokens with label ∈ {{0,1}})")
    print(f"{'='*70}")
    print(f"{'Signal':<28} {'Spearman ρ':>12} {'p-value':>12} {'n_pairs':>10}  Note")
    print(f"{'-'*70}")

    for r in results:
        rho  = f"{r['spearman_rho']:+.4f}" if not (isinstance(r['spearman_rho'], float) and r['spearman_rho'] != r['spearman_rho']) else "  nan"
        pval = f"{r['p_value']:.2e}"        if not (isinstance(r['p_value'], float)       and r['p_value'] != r['p_value'])             else "  nan"
        print(f"  {r['signal']:<26} {rho:>12} {pval:>12} {r['n_pairs']:>10}  {r['note']}")

    print(f"{'='*70}")

    # Highlight key comparison
    rho_map = {r["signal"]: r["spearman_rho"] for r in results if not (isinstance(r["spearman_rho"], float) and r["spearman_rho"] != r["spearman_rho"])}
    h2o     = rho_map.get("h2o_attn")
    entropy = rho_map.get("attn_entropy")
    _residual_signals = (
        "hs_l2_diff", "hs_l2_diff_ema09", "hs_l2_diff_rolling64",
        "hs_cos_dist", "hs_norm",
        "kv_key_var", "kv_key_var_ema09", "kv_key_var_rolling64",
        "kv_val_var", "kv_val_var_ema09", "kv_val_var_rolling64",
        "kv_key_norm", "cross_head_var",
    )
    best_residual = max(
        (rho_map.get(s, float("-inf")) for s in _residual_signals),
        default=None,
    )

    print("\nKey comparisons:")
    if h2o is not None:
        print(f"  H2O (attention SOTA):       ρ = {h2o:+.4f}")
    if entropy is not None:
        print(f"  Attn entropy (ThinKV-style):ρ = {entropy:+.4f}")
    if best_residual is not None:
        print(f"  Best residual-stream signal:ρ = {best_residual:+.4f}")
    if h2o is not None and best_residual is not None:
        delta = best_residual - h2o
        verdict = "BEATS H2O" if delta > 0 else "DOES NOT beat H2O"
        print(f"\n  Hypothesis: residual-stream signals beat attention → {verdict} (Δρ = {delta:+.4f})")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Loading traces:  {args.traces}")
    traces = load_jsonl(args.traces)
    print(f"  {len(traces)} traces loaded.")

    print(f"Loading labels:  {args.labels}")
    labels = load_jsonl(args.labels)
    print(f"  {len(labels)} labelled traces loaded.")

    pairs = join_traces_labels(traces, labels, args.assume_aligned)
    print(f"  {len(pairs)} matched pairs.")

    if not pairs:
        sys.exit("No matched pairs found. Check that --traces and --labels use the same problems.")

    print("\nComputing signal–importance correlations...")
    signal_arrays, label_array = collect_signal_label_pairs(pairs)

    n_labelled_tokens = len(label_array)
    print(f"  {n_labelled_tokens} labelled token positions across {len(pairs)} traces.")

    if n_labelled_tokens == 0:
        sys.exit(
            "No labelled positions found.\n"
            "Make sure label_importance.py ran successfully and produced importance fields."
        )

    results = compute_correlations(signal_arrays, label_array)
    print_table(results, n_traces=len(pairs), n_tokens=n_labelled_tokens)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["signal", "spearman_rho", "p_value", "n_pairs", "note"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results written to {output_path}")


_DATASETS = ["math500", "aime2024", "livecodebench", "gsm8k"]

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=_DATASETS, default=None,
                   help="Dataset name — sets default paths to match collect_traces.py / label_importance.py conventions")
    p.add_argument("--traces",  type=Path, default=None,
                   help="JSONL from collect_traces.py (default: data/<dataset>_traces.jsonl)")
    p.add_argument("--labels",  type=Path, default=None,
                   help="JSONL from label_importance.py (default: data/<dataset>_traces_labelled.jsonl)")
    p.add_argument("--output",  type=Path, default=None,
                   help="CSV output path (default: results/<dataset>_signal_ablation.csv)")
    p.add_argument("--assume_aligned", action="store_true",
                   help="Match traces and labels by position instead of by problem text")
    args = p.parse_args()

    if args.dataset:
        if args.traces is None:
            args.traces = Path(f"data/{args.dataset}_traces.jsonl")
        if args.labels is None:
            args.labels = Path(f"data/{args.dataset}_traces_labelled.jsonl")
        if args.output is None:
            args.output = Path(f"results/{args.dataset}_signal_ablation.csv")
    if args.traces is None or args.labels is None:
        p.error("Provide either --dataset or both --traces and --labels")

    return args


if __name__ == "__main__":
    main()
