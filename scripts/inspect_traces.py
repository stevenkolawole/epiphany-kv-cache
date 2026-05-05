#!/usr/bin/env python3
"""
inspect_traces.py

Manual trace inspection: reads labelled traces and posthoc signal traces,
matches them by problem text, then for each trace produces a token-level
breakdown of: token text, importance label, and key signal values.

Outputs a plain-text report to stdout (redirect to a file for review).

Usage:
    python scripts/inspect_traces.py --dataset math500 --n 5 > reports/math500_trace_inspection.txt
    python scripts/inspect_traces.py --dataset aime2024 --n 3 > reports/aime2024_trace_inspection.txt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def rolling_mean(values: List[float], window: int = 64) -> List[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        valid = [v for v in values[start : i + 1] if not np.isnan(v)]
        out.append(float(np.mean(valid)) if valid else float("nan"))
    return out


def load_jsonl(path: Path, n: Optional[int] = None) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if n and len(records) >= n * 3:  # load extra to find n matches
                break
    return records


def decode_tokens(token_ids: List[int], tokenizer) -> List[str]:
    return [tokenizer.decode([t]) for t in token_ids]


def get_signal_at(signals: Dict, name: str, pos: int) -> Optional[float]:
    v = signals.get(name)
    if v is None or pos >= len(v):
        return None
    val = v[pos]
    return None if val == -1.0 else val


def analyze_trace(
    label_rec: Dict,
    signal_rec: Dict,
    tokenizer,
    max_show_tokens: int = 200,
) -> str:
    out_lines = []

    problem = label_rec["problem"]
    prompt_len = label_rec["prompt_len"]
    token_ids = label_rec["token_ids"]
    importance = label_rec["importance"]
    generated_text = label_rec["generated_text"]
    correct = label_rec.get("correct")
    pred_answer = label_rec.get("pred_answer")

    out_lines.append("=" * 80)
    out_lines.append(f"PROBLEM: {problem[:200]}")
    out_lines.append(f"CORRECT: {correct}  |  PREDICTED: {pred_answer}")
    out_lines.append(f"Prompt tokens: {prompt_len}  |  Total tokens: {len(token_ids)}")
    out_lines.append(f"Generated chars: {len(generated_text)}")

    # Label distribution (generated tokens only)
    gen_labels = importance[prompt_len:]
    n1 = sum(1 for x in gen_labels if x == 1)
    n0 = sum(1 for x in gen_labels if x == 0)
    nm = sum(1 for x in gen_labels if x == -1)
    total_gen = len(gen_labels)
    out_lines.append(f"Generated token labels: important=1:{n1} ({n1/total_gen:.1%})  dispensable=0:{n0} ({n0/total_gen:.1%})  unlabelled=-1:{nm} ({nm/total_gen:.1%})")
    out_lines.append("")

    # Decode tokens
    try:
        token_texts = decode_tokens(token_ids, tokenizer)
    except Exception as e:
        token_texts = [f"[{i}]" for i in token_ids]
        out_lines.append(f"(Token decode failed: {e})")

    # Compute rolling64 for key signals
    signals = signal_rec.get("signals", {})
    key_signals = ["hs_l2_diff_l10", "hs_l2_diff_l21", "kv_key_var", "kv_val_var", "h2o_attn", "attn_entropy"]
    rolling = {}
    for sig in key_signals:
        raw = signals.get(sig)
        if raw:
            clean = [float("nan") if v == -1.0 else v for v in raw]
            rolling[sig] = rolling_mean(clean)

    # Show generated tokens with labels and signals
    out_lines.append(f"{'Pos':>6}  {'Label':>6}  {'Token text':<20}  {'l10_r64':>9}  {'l21_r64':>9}  {'kv_key':>9}  {'kv_val':>9}  {'h2o':>7}  {'entropy':>8}")
    out_lines.append("-" * 100)

    # Always show first 30 generated tokens (warmup)
    # Show all important transitions (label changes)
    # Show last 20 generated tokens (answer formation)
    shown = 0
    prev_label = None
    important_transitions = []
    for pos in range(prompt_len, min(len(token_ids), len(importance))):
        lbl = importance[pos]
        if prev_label is not None and lbl != prev_label and lbl in (0, 1):
            important_transitions.append(pos)
        prev_label = lbl

    gen_end = min(len(token_ids), len(importance))
    positions_to_show = set()
    # First 30 generated tokens
    for pos in range(prompt_len, min(prompt_len + 30, gen_end)):
        positions_to_show.add(pos)
    # Around label transitions (±3 tokens)
    for t in important_transitions[:30]:
        for delta in range(-3, 4):
            if prompt_len <= t + delta < gen_end:
                positions_to_show.add(t + delta)
    # Last 20 generated tokens
    for pos in range(max(prompt_len, gen_end - 20), gen_end):
        positions_to_show.add(pos)

    sorted_positions = sorted(positions_to_show)
    last_shown = None
    for pos in sorted_positions:
        if last_shown is not None and pos > last_shown + 1:
            out_lines.append(f"{'...':>6}  {'':>6}  {'':>20}  {'':>9}  {'':>9}  {'':>9}  {'':>9}  {'':>7}  {'':>8}")
        lbl = importance[pos] if pos < len(importance) else -1
        tok_text = token_texts[pos] if pos < len(token_texts) else "?"
        # Escape whitespace for readability
        tok_display = repr(tok_text)[1:-1][:20]

        def fmt(sig_name):
            r = rolling.get(sig_name)
            if r is None or pos >= len(r):
                return "       N/A"
            v = r[pos]
            return f"{v:>9.4f}" if not np.isnan(v) else "       nan"

        label_str = {1: "KEEP", 0: "DROP", -1: "----"}.get(lbl, str(lbl))
        row = f"{pos:>6}  {label_str:>6}  {tok_display:<20}  {fmt('hs_l2_diff_l10')}  {fmt('hs_l2_diff_l21')}  {fmt('kv_key_var')}  {fmt('kv_val_var')}  {fmt('h2o_attn')}  {fmt('attn_entropy')}"
        out_lines.append(row)
        last_shown = pos
        shown += 1

    out_lines.append("")

    # Aggregate: mean signal at KEEP vs DROP positions
    keep_sigs = {s: [] for s in key_signals}
    drop_sigs = {s: [] for s in key_signals}
    for pos in range(prompt_len, gen_end):
        lbl = importance[pos]
        if lbl not in (0, 1):
            continue
        for sig in key_signals:
            r = rolling.get(sig)
            if r and pos < len(r) and not np.isnan(r[pos]):
                if lbl == 1:
                    keep_sigs[sig].append(r[pos])
                else:
                    drop_sigs[sig].append(r[pos])

    out_lines.append("AGGREGATE: mean signal at KEEP (label=1) vs DROP (label=0) positions:")
    out_lines.append(f"  {'Signal':<28}  {'KEEP mean':>10}  {'DROP mean':>10}  {'diff (K-D)':>12}  {'direction':>12}")
    out_lines.append("  " + "-" * 76)
    for sig in key_signals:
        k_vals = keep_sigs[sig]
        d_vals = drop_sigs[sig]
        if not k_vals or not d_vals:
            out_lines.append(f"  {sig:<28}  {'N/A':>10}  {'N/A':>10}  {'N/A':>12}  {'N/A':>12}")
            continue
        k_mean = np.mean(k_vals)
        d_mean = np.mean(d_vals)
        diff = k_mean - d_mean
        direction = "K > D ✓" if diff > 0 else "D > K ✗"
        out_lines.append(f"  {sig:<28}  {k_mean:>10.4f}  {d_mean:>10.4f}  {diff:>+12.4f}  {direction:>12}")
    out_lines.append("")

    # Show 10 highest-signal Band A tokens that are DROP
    # (the failure cases: high Band A but labeled dispensable)
    out_lines.append("TOP 10 BAND-A SIGNAL TOKENS LABELED DROP (false positives for Band A rule):")
    l10_r = rolling.get("hs_l2_diff_l10", [])
    drop_l10 = [(pos, l10_r[pos]) for pos in range(prompt_len, gen_end)
                if importance[pos] == 0 and pos < len(l10_r) and not np.isnan(l10_r[pos])]
    drop_l10.sort(key=lambda x: -x[1])
    for pos, val in drop_l10[:10]:
        tok = repr(token_texts[pos])[1:-1][:25] if pos < len(token_texts) else "?"
        out_lines.append(f"  pos={pos:5d}  l10_r64={val:.4f}  tok='{tok}'")

    out_lines.append("")

    # Show 10 highest Band-B signal tokens that are KEEP
    out_lines.append("TOP 10 BAND-B SIGNAL TOKENS LABELED KEEP (false positives for Band B rule):")
    l21_r = rolling.get("hs_l2_diff_l21", [])
    keep_l21 = [(pos, l21_r[pos]) for pos in range(prompt_len, gen_end)
                if importance[pos] == 1 and pos < len(l21_r) and not np.isnan(l21_r[pos])]
    keep_l21.sort(key=lambda x: -x[1])
    for pos, val in keep_l21[:10]:
        tok = repr(token_texts[pos])[1:-1][:25] if pos < len(token_texts) else "?"
        out_lines.append(f"  pos={pos:5d}  l21_r64={val:.4f}  tok='{tok}'")

    out_lines.append("")
    return "\n".join(out_lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="math500", choices=["math500", "math500_eager", "aime2024", "aime2024_eager", "aime2025", "aime2025_eager", "aime2026", "aime2026_eager", "gsm8k_eager"])
    p.add_argument("--n", type=int, default=5, help="Number of traces to inspect")
    p.add_argument("--model", default="deepseek-ai/deepseek-r1-distill-llama-8b")
    p.add_argument("--no_tokenizer", action="store_true", help="Skip tokenizer loading (tokens shown as IDs)")
    args = p.parse_args()

    # Paths
    label_path = Path(f"data/{args.dataset}_traces_labelled.jsonl")
    posthoc_path = Path(f"data/{args.dataset}_traces_posthoc.jsonl")

    if not label_path.exists():
        print(f"Label file not found: {label_path}")
        return
    if not posthoc_path.exists():
        print(f"Posthoc file not found: {posthoc_path}")
        return

    # Tokenizer
    tokenizer = None
    if not args.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            import os
            cache_dir = os.environ.get("HF_HOME", "/data/hf_cache/skolawol")
            print(f"Loading tokenizer from {cache_dir}...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
            print("Tokenizer loaded.", flush=True)
        except Exception as e:
            print(f"Tokenizer load failed ({e}); tokens will be shown as IDs.", flush=True)

    class FallbackTokenizer:
        def decode(self, ids):
            return f"[{ids[0]}]"

    if tokenizer is None:
        tokenizer = FallbackTokenizer()

    print(f"Loading label records from {label_path}...", flush=True)
    label_records = load_jsonl(label_path, n=args.n)
    print(f"  Loaded {len(label_records)} label records.", flush=True)

    print(f"Loading posthoc signal records from {posthoc_path}...", flush=True)
    posthoc_records = load_jsonl(posthoc_path, n=args.n)
    print(f"  Loaded {len(posthoc_records)} posthoc records.", flush=True)

    # Match by problem text
    posthoc_by_problem = {r["problem"]: r for r in posthoc_records}
    matched = 0

    for lab in label_records:
        sig = posthoc_by_problem.get(lab["problem"])
        if sig is None:
            print(f"  [SKIP] No posthoc match for: {lab['problem'][:60]}...")
            continue

        report = analyze_trace(lab, sig, tokenizer)
        print(report)
        matched += 1
        if matched >= args.n:
            break

    print(f"\n=== DONE: {matched} traces inspected ===")


if __name__ == "__main__":
    main()
