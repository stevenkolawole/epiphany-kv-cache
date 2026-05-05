#!/usr/bin/env python3
"""
quality_check.py

Manual trace quality check: cross-references labelled traces with posthoc signals
to verify that signal directions match Phase 0B findings at the trace level.

For each matched trace, computes mean signal at KEEP (label=1) vs DROP (label=0)
positions among generated tokens.  Reports per-trace and aggregate statistics.

Expected directions (from Phase 0B — competition math):
  l10_rolling64 :  KEEP > DROP  (Band A positive rho)
  l21_rolling64 :  KEEP < DROP  (Band B negative rho)
  combined score:  KEEP > DROP  (Phase 1 eviction signal)
  kv_val_var_r64:  KEEP > DROP  (consistently non-negative across datasets)
  kv_key_var_r64:  KEEP > DROP  (positive for math500; sign-flips for GSM8K)
  h2o_attn_r64  :  KEEP > DROP  (weakly positive; weakest signal)
  attn_entropy  :  direction uncertain (sign flips across datasets)

Usage:
    python scripts/quality_check.py --dataset math500 --n 50
    python scripts/quality_check.py --dataset aime2024 --n 20
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


WINDOW = 64
SENTINEL = -1.0


def rolling_mean(values: List[float], window: int = WINDOW) -> List[float]:
    """Causal rolling mean, NaN-aware, consistent with signal_ablation.py."""
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        valid = [v for v in values[start:i + 1] if not math.isnan(v)]
        out.append(sum(valid) / len(valid) if valid else float('nan'))
    return out


def clean(values: List[float]) -> List[float]:
    """Replace SENTINEL (-1.0) with NaN."""
    return [float('nan') if v == SENTINEL else v for v in values]


def load_jsonl(path: Path, n: Optional[int]) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if n and len(records) >= n * 4:  # load extra for matching
                break
    return records


def keep_drop_means(
    importance: List[int],
    signal_r64: List[float],
    prompt_len: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Mean signal at KEEP (1) and DROP (0) positions among generated tokens.
    Returns (keep_mean, drop_mean); None if fewer than 5 usable values.
    """
    keep, drop = [], []
    for pos in range(prompt_len, len(importance)):
        lbl = importance[pos]
        if lbl not in (0, 1):
            continue
        if pos >= len(signal_r64):
            continue
        v = signal_r64[pos]
        if math.isnan(v):
            continue
        if lbl == 1:
            keep.append(v)
        else:
            drop.append(v)
    k = sum(keep) / len(keep) if len(keep) >= 5 else None
    d = sum(drop) / len(drop) if len(drop) >= 5 else None
    return k, d


def fmt(v: Optional[float]) -> str:
    if v is None:
        return "     N/A"
    return f"{v:8.4f}"


def direction_ok(k: Optional[float], d: Optional[float], expected: int) -> Optional[bool]:
    """expected: +1 means KEEP > DROP expected, -1 means KEEP < DROP expected, 0 = uncertain."""
    if k is None or d is None:
        return None
    if expected == 1:
        return k > d
    if expected == -1:
        return k < d
    return None  # uncertain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="math500",
                    choices=["math500", "math500_eager", "aime2024", "aime2024_eager"])
    ap.add_argument("--n", type=int, default=50, help="Max traces to analyse")
    args = ap.parse_args()

    label_path = Path(f"data/{args.dataset}_traces_labelled.jsonl")
    posthoc_path = Path(f"data/{args.dataset}_traces_posthoc.jsonl")

    for p in [label_path, posthoc_path]:
        if not p.exists():
            print(f"Missing: {p}")
            return

    print(f"Loading labelled traces from {label_path} ...", flush=True)
    lab_records = load_jsonl(label_path, args.n)
    print(f"  {len(lab_records)} records loaded", flush=True)

    print(f"Loading posthoc signals from {posthoc_path} ...", flush=True)
    ph_records = load_jsonl(posthoc_path, args.n)
    ph_by_problem = {r["problem"]: r for r in ph_records}
    print(f"  {len(ph_by_problem)} unique problems in posthoc", flush=True)

    # Signals and expected directions
    # (key, source, expected_sign, description)
    CHECKS = [
        ("l10",          "posthoc", +1, "l10_rolling64: Band A, KEEP > DROP"),
        ("l21",          "posthoc", -1, "l21_rolling64: Band B, KEEP < DROP"),
        ("combined",     "derived", +1, "l10-l21 rolling64: Phase 1 signal, KEEP > DROP"),
        ("kv_val_var",   "label",   +1, "kv_val_var_r64: stable KV, KEEP > DROP"),
        ("kv_key_var",   "label",   +1, "kv_key_var_r64: math500 positive, may flip"),
        ("h2o_attn",     "label",   +1, "h2o_attn_r64: weakest signal, KEEP > DROP (weakly)"),
        ("attn_entropy", "label",    0, "attn_entropy_r64: uncertain direction"),
    ]

    direction_correct = {k: 0 for k, _, _, _ in CHECKS}
    direction_total = {k: 0 for k, _, _, _ in CHECKS}

    all_rows = []
    matched = 0

    for lab in lab_records:
        if matched >= args.n:
            break
        ph = ph_by_problem.get(lab["problem"])
        if ph is None:
            continue

        prompt_len = lab["prompt_len"]
        importance = lab["importance"]
        gen_labels = importance[prompt_len:]
        n_keep = sum(1 for x in gen_labels if x == 1)
        n_drop = sum(1 for x in gen_labels if x == 0)
        n_total_gen = len(gen_labels)

        if n_keep < 10 or n_drop < 10:
            continue

        posthoc_sigs = ph.get("signals", {})
        label_sigs = lab.get("signals", {})

        # Compute rolling64 for each signal
        def get_r64_from_posthoc(name):
            raw = posthoc_sigs.get(name, [])
            if not raw:
                return []
            return rolling_mean(clean(raw))

        def get_r64_from_label(name):
            raw = label_sigs.get(name, [])
            if not raw:
                return []
            return rolling_mean(clean(raw))

        l10_r64 = get_r64_from_posthoc("hs_l2_diff_l10")
        l21_r64 = get_r64_from_posthoc("hs_l2_diff_l21")
        kv_val_r64 = get_r64_from_label("kv_val_var")
        kv_key_r64 = get_r64_from_label("kv_key_var")
        h2o_r64 = get_r64_from_label("h2o_attn")
        ent_r64 = get_r64_from_label("attn_entropy")

        k_l10, d_l10 = keep_drop_means(importance, l10_r64, prompt_len)
        k_l21, d_l21 = keep_drop_means(importance, l21_r64, prompt_len)
        k_kvv, d_kvv = keep_drop_means(importance, kv_val_r64, prompt_len)
        k_kvk, d_kvk = keep_drop_means(importance, kv_key_r64, prompt_len)
        k_h2o, d_h2o = keep_drop_means(importance, h2o_r64, prompt_len)
        k_ent, d_ent = keep_drop_means(importance, ent_r64, prompt_len)

        # Combined score = l10 rolling64 - l21 rolling64 (per-position difference)
        combined_r64 = []
        for i, (a, b) in enumerate(zip(l10_r64, l21_r64)):
            if math.isnan(a) or math.isnan(b):
                combined_r64.append(float('nan'))
            else:
                combined_r64.append(a - b)
        k_comb, d_comb = keep_drop_means(importance, combined_r64, prompt_len)

        row = {
            "problem": lab["problem"][:55],
            "correct": lab.get("correct"),
            "n_keep": n_keep,
            "n_drop": n_drop,
            "important_frac": n_keep / (n_keep + n_drop) if (n_keep + n_drop) > 0 else None,
            "l10": (k_l10, d_l10),
            "l21": (k_l21, d_l21),
            "combined": (k_comb, d_comb),
            "kv_val_var": (k_kvv, d_kvv),
            "kv_key_var": (k_kvk, d_kvk),
            "h2o_attn": (k_h2o, d_h2o),
            "attn_entropy": (k_ent, d_ent),
        }
        all_rows.append(row)

        sig_map = {
            "l10": (k_l10, d_l10),
            "l21": (k_l21, d_l21),
            "combined": (k_comb, d_comb),
            "kv_val_var": (k_kvv, d_kvv),
            "kv_key_var": (k_kvk, d_kvk),
            "h2o_attn": (k_h2o, d_h2o),
            "attn_entropy": (k_ent, d_ent),
        }
        for key, _, expected, _ in CHECKS:
            k, d = sig_map[key]
            ok = direction_ok(k, d, expected)
            if ok is not None:
                direction_total[key] += 1
                if ok:
                    direction_correct[key] += 1

        matched += 1

    if not all_rows:
        print("No traces matched. Check file paths and content.")
        return

    # ── Per-trace table: HS signals ──────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"DATASET: {args.dataset}  |  {matched} traces  |  HS SIGNALS")
    print(f"{'='*100}")
    hdr = f"{'Problem':<40} {'Corr':>5} {'K/D':>7} {'frac':>5} |"
    hdr += f" {'l10_K':>8} {'l10_D':>8} | {'l21_K':>8} {'l21_D':>8} | {'comb_K':>8} {'comb_D':>8}"
    print(hdr)
    print("-" * 100)
    for r in all_rows:
        k_l10, d_l10 = r["l10"]
        k_l21, d_l21 = r["l21"]
        k_c, d_c = r["combined"]
        frac_str = f"{r['important_frac']:.2f}" if r["important_frac"] is not None else " N/A"
        correct_str = ("T" if r["correct"] else "F") if r["correct"] is not None else "?"
        print(
            f"{r['problem']:<40} {correct_str:>5} {r['n_keep']}/{r['n_drop']:<4} {frac_str:>5} |"
            f" {fmt(k_l10)} {fmt(d_l10)} | {fmt(k_l21)} {fmt(d_l21)} | {fmt(k_c)} {fmt(d_c)}"
        )

    # ── Per-trace table: KV + attention signals ──────────────────────────────
    print(f"\n{'='*100}")
    print(f"DATASET: {args.dataset}  |  KV + ATTENTION SIGNALS")
    print(f"{'='*100}")
    hdr2 = f"{'Problem':<40} {'Corr':>5} |"
    hdr2 += f" {'kvval_K':>8} {'kvval_D':>8} | {'kvkey_K':>8} {'kvkey_D':>8}"
    hdr2 += f" | {'h2o_K':>8} {'h2o_D':>8} | {'ent_K':>8} {'ent_D':>8}"
    print(hdr2)
    print("-" * 100)
    for r in all_rows:
        k_v, d_v = r["kv_val_var"]
        k_k, d_k = r["kv_key_var"]
        k_h, d_h = r["h2o_attn"]
        k_e, d_e = r["attn_entropy"]
        correct_str = ("T" if r["correct"] else "F") if r["correct"] is not None else "?"
        print(
            f"{r['problem']:<40} {correct_str:>5} |"
            f" {fmt(k_v)} {fmt(d_v)} | {fmt(k_k)} {fmt(d_k)}"
            f" | {fmt(k_h)} {fmt(d_h)} | {fmt(k_e)} {fmt(d_e)}"
        )

    # ── Aggregate direction check ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"DIRECTION CONSISTENCY — fraction of traces where expected direction holds")
    print(f"{'='*80}")
    print(f"{'Signal':<20} {'Expected':>20} {'Correct':>10} {'Total':>8} {'Frac':>8}  Interpretation")
    print("-" * 85)
    for key, _, expected, desc in CHECKS:
        corr = direction_correct[key]
        tot = direction_total[key]
        frac = corr / tot if tot > 0 else float('nan')
        mark = ("✓" if frac >= 0.6 else ("~" if frac >= 0.4 else "✗")) if not math.isnan(frac) else "?"
        print(f"{key:<20} {desc[:20]:>20} {corr:>10} {tot:>8} {frac:>7.1%}  {mark}")

    # ── Quick sanity checks on implementation ───────────────────────────────
    print(f"\n{'='*80}")
    print("IMPLEMENTATION SANITY CHECKS")
    print(f"{'='*80}")

    # Check 1: prompt tokens should have label=1 (all kept by construction)
    # We just verify prompt_len is consistent with importance list
    sample = all_rows[0]
    print(f"[1] Sample important_frac: {sample['important_frac']:.2f} "
          f"(expected ~0.20 for math500, ~0.50+ for AIME)")

    # Check 2: signal coverage — are l10/l21 signals available for most traces?
    l10_avail = sum(1 for r in all_rows if r["l10"][0] is not None)
    kv_avail = sum(1 for r in all_rows if r["kv_val_var"][0] is not None)
    print(f"[2] l10_rolling64 available for {l10_avail}/{len(all_rows)} traces")
    print(f"[2] kv_val_var_rolling64 available for {kv_avail}/{len(all_rows)} traces")

    # Check 3: mean important_frac across traces
    fracs = [r["important_frac"] for r in all_rows if r["important_frac"] is not None]
    mean_frac = sum(fracs) / len(fracs) if fracs else float('nan')
    print(f"[3] Mean important_frac across {len(fracs)} traces: {mean_frac:.3f}")

    # Check 4: l10 > l21 at KEEP positions on average (expected from Phase 0B)
    l10_keep_vals = [r["l10"][0] for r in all_rows if r["l10"][0] is not None]
    l21_keep_vals = [r["l21"][0] for r in all_rows if r["l21"][0] is not None]
    if l10_keep_vals and l21_keep_vals:
        mean_l10_k = sum(l10_keep_vals) / len(l10_keep_vals)
        mean_l21_k = sum(l21_keep_vals) / len(l21_keep_vals)
        print(f"[4] Mean l10_r64 at KEEP: {mean_l10_k:.4f}  |  mean l21_r64 at KEEP: {mean_l21_k:.4f}")
        print(f"    l10_keep > l21_keep: {mean_l10_k > mean_l21_k} (expected True for competition math)")

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
