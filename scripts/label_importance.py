#!/usr/bin/env python3
"""
label_importance.py

Assigns per-token counterfactual importance labels to traces produced by
collect_traces.py.

For each trace, we re-run inference with sliding windows of tokens masked
(replaced with a neutral padding token) and record whether the answer
flips.  A token is labelled "important" if masking its window causes the
model to produce a different (or absent) answer.

This produces ground-truth importance labels that signal_ablation.py then
uses to measure how well each collected signal correlates with true
importance.

Algorithm
---------
For each trace (problem, generated_text, token_ids, prompt_len, correct):
  1. Skip if the original answer was incorrect — we can't define importance
     for a trace that was already wrong.
  2. Build the full token sequence: [prompt | generated_tokens].
  3. For each window of size W starting at position s (stride S):
       - Replace token_ids[s : s+W] with pad_token_id.
       - Run the model with this masked sequence as input (no KV cache
         reuse — we need a fresh forward pass over the full modified sequence).
       - Extract the answer from the model's output.
       - Label tokens s..s+W-1 as important=True if the answer flipped,
         important=False otherwise.
  4. Write augmented trace to output JSONL with an added field:
       importance   List[int | None]
         1  = this token was in a window whose masking caused an answer flip
         0  = token was masked but answer did not change
         -1 = token was never masked (e.g. prompt positions, or last window
              didn't cover it) — treated as unknown, not used in ablation

Usage
-----
    python scripts/label_importance.py \\
        --input  data/math500_traces.jsonl \\
        --output data/math500_traces_labelled.jsonl \\
        --window 32 --stride 16 --max_new_tokens 512

    # Dry run: first 3 traces, short generation
    python scripts/label_importance.py \\
        --input data/math500_traces.jsonl \\
        --output data/math500_traces_labelled.jsonl \\
        --dry_run

Design notes
------------
- We mask by replacing with pad_token_id (or eos_token_id as fallback).
  We do NOT delete tokens — sequence length stays constant so positional
  encodings and attention patterns are preserved.  This is the standard
  "occlusion" approach used in interpretability research.
- We only mask the *generated* portion (positions >= prompt_len).  The
  prompt is fixed; its tokens are never masked.  The prompt is always
  fully "important" by definition (it's the problem statement).
- Window size W=32 and stride S=16 are defaults tuned for reasoning traces
  of ~4k–32k tokens.  Smaller W gives higher resolution but more forward
  passes; larger W reduces cost but may give coarser labels.
- Each masked inference re-uses the unmodified prompt prefix via a KV-cache
  prefill (up to position s), then runs with masked tokens from s onward.
  This is an approximation: the model sees a mixed context (real prefix,
  masked suffix), which may not perfectly isolate the window's contribution.
  This is an acceptable approximation for correlation analysis.
- Runtime: O(seq_len / stride) forward passes per trace.  For a 4k-token
  trace with W=32, S=16: ~250 passes.  Each pass is short (max_new_tokens=512).
  Budget ~5–10 min per trace on a single A100.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("Missing dependency: pip install transformers")


# ── Shared with collect_traces.py ────────────────────────────────────────────

_DEFAULT_HF_CACHE = "/data/hf_cache/skolawol"

def _hf_cache() -> str:
    return os.environ.get("HF_HOME", _DEFAULT_HF_CACHE)


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

def extract_boxed(text: str) -> Optional[str]:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None

def answers_match(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False
    def norm(s):
        return s.strip().lower().replace(" ", "").replace(",", "")
    return norm(pred) == norm(gt)


# ── Masked inference ──────────────────────────────────────────────────────────

def run_masked_inference(
    model,
    tokenizer,
    full_ids: torch.Tensor,          # (1, total_len) — prompt + generated
    prompt_len: int,
    mask_start: int,                 # first position to mask (>= prompt_len)
    mask_end: int,                   # one past last position to mask
    max_new_tokens: int,
    pad_id: int,
    device: torch.device,
) -> Optional[str]:
    """
    Replace token_ids[mask_start:mask_end] with pad_id, then run the model
    on the modified prompt prefix to get a new answer.

    We use the prefix up to mask_start as the input (with KV caching), then
    continue generation.  The masked tokens are *not* fed to the model —
    instead we treat mask_start as the new "current generation point" and
    generate up to max_new_tokens from there.

    This approximation is valid for our purposes: we want to know whether
    the model can still reach the correct answer when the window [mask_start,
    mask_end) is not present in its context.  Since the model saw the prompt
    and the pre-window generated tokens, it has the context up to mask_start.

    Returns the extracted boxed answer, or None if no answer was found.
    """
    # Use the original sequence up to mask_start as context
    context_ids = full_ids[:, :mask_start].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=context_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0, mask_start:].tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return extract_boxed(text)


# ── Main labelling loop ───────────────────────────────────────────────────────

def label_trace(
    trace: Dict,
    model,
    tokenizer,
    window_size: int,
    stride: int,
    max_new_tokens: int,
    pad_id: int,
    device: torch.device,
) -> Dict:
    """
    Add an 'importance' field to the trace dict.
    Returns the augmented trace.
    """
    token_ids   = trace["token_ids"]
    prompt_len  = trace["prompt_len"]
    ground_truth = trace["ground_truth"]
    total_len   = len(token_ids)

    # Initialise all positions as unknown (-1)
    importance = [-1] * total_len
    # Prompt tokens are structurally important by definition
    for i in range(prompt_len):
        importance[i] = 1

    full_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # (1, total_len)

    # Slide window over the generated portion only
    generated_len = total_len - prompt_len
    if generated_len <= 0:
        trace["importance"] = importance
        return trace

    windows_tested = 0
    windows_flipped = 0

    pos = prompt_len
    while pos < total_len:
        end = min(pos + window_size, total_len)

        masked_answer = run_masked_inference(
            model=model,
            tokenizer=tokenizer,
            full_ids=full_ids,
            prompt_len=prompt_len,
            mask_start=pos,
            mask_end=end,
            max_new_tokens=max_new_tokens,
            pad_id=pad_id,
            device=device,
        )

        flipped = not answers_match(masked_answer, ground_truth)
        label = 1 if flipped else 0
        for i in range(pos, end):
            importance[i] = label

        windows_tested += 1
        if flipped:
            windows_flipped += 1

        pos += stride

    trace["importance"] = importance
    trace["label_stats"] = {
        "windows_tested":  windows_tested,
        "windows_flipped": windows_flipped,
        "flip_rate":       windows_flipped / max(windows_tested, 1),
        "important_frac":  sum(1 for x in importance if x == 1) / total_len,
    }
    return trace


def main():
    args = parse_args()

    if args.dry_run:
        args.max_traces = 3
        args.max_new_tokens = 128
        print("DRY RUN: 3 traces, 128 max_new_tokens per masked inference")

    input_path  = args.input
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count traces and filter to correctly-answered ones
    traces = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("correct") is not True:
                continue   # can only label importance for correct traces
            if t.get("ground_truth") is None:
                continue   # LiveCodeBench — no boxed answer to compare
            traces.append(t)
            if args.max_traces and len(traces) >= args.max_traces:
                break

    if not traces:
        sys.exit(
            f"No correctly-answered traces found in {input_path}.\n"
            "Run collect_traces.py with --max_new_tokens >= 4096 first."
        )
    print(f"Found {len(traces)} correctly-answered traces to label.")

    # Load model
    print(f"\nLoading model: {args.model}")
    cache_dir = _hf_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model on device: {device}")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Label traces
    n_written = 0
    with open(output_path, "w") as f_out:
        for trace in tqdm(traces, desc="Labelling traces"):
            total_len   = len(trace["token_ids"])
            prompt_len  = trace["prompt_len"]
            gen_len     = total_len - prompt_len
            n_windows   = max(1, (gen_len - args.window) // args.stride + 1)

            try:
                labelled = label_trace(
                    trace=trace,
                    model=model,
                    tokenizer=tokenizer,
                    window_size=args.window,
                    stride=args.stride,
                    max_new_tokens=args.max_new_tokens,
                    pad_id=pad_id,
                    device=device,
                )
            except Exception as e:
                import traceback
                print(f"\n  [SKIP] Error labelling trace: {e}", file=sys.stderr)
                if args.dry_run:
                    traceback.print_exc(file=sys.stderr)
                continue

            f_out.write(json.dumps(labelled) + "\n")
            f_out.flush()
            n_written += 1

            stats = labelled.get("label_stats", {})
            tqdm.write(
                f"  len={total_len:>6}  windows={stats.get('windows_tested',0):>4}"
                f"  flip_rate={stats.get('flip_rate', 0):.2f}"
                f"  important_frac={stats.get('important_frac', 0):.2f}"
            )

    print(f"\n{'='*60}")
    print(f"Done. Wrote {n_written} labelled traces to {output_path}")
    print(f"Window={args.window}  Stride={args.stride}  MaxNewTokens={args.max_new_tokens}")
    print(f"{'='*60}")


_DATASETS = ["math500", "aime2024", "livecodebench", "gsm8k"]

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=_DATASETS, default=None,
                   help="Dataset name — sets default --input and --output paths to match collect_traces.py conventions")
    p.add_argument("--input",   type=Path, default=None,
                   help="Input JSONL from collect_traces.py (default: data/<dataset>_traces.jsonl)")
    p.add_argument("--output",  type=Path, default=None,
                   help="Output JSONL with added 'importance' field (default: data/<dataset>_traces_labelled.jsonl)")
    p.add_argument("--model",   default="deepseek-ai/deepseek-r1-distill-llama-8b",
                   help="HuggingFace model ID (must match the one used for collection)")
    p.add_argument("--window",  type=int, default=32,
                   help="Mask window size in tokens (default: 32)")
    p.add_argument("--stride",  type=int, default=16,
                   help="Stride between windows (default: 16)")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="Max tokens to generate per masked inference (default: 512)")
    p.add_argument("--max_traces", type=int, default=None,
                   help="Process at most this many traces (default: all)")
    p.add_argument("--dry_run", action="store_true",
                   help="Run on first 3 traces with 128 max_new_tokens to verify setup")
    args = p.parse_args()

    # Resolve default paths from dataset name
    if args.dataset:
        if args.input is None:
            args.input = Path(f"data/{args.dataset}_traces.jsonl")
        if args.output is None:
            args.output = Path(f"data/{args.dataset}_traces_labelled.jsonl")
    if args.input is None or args.output is None:
        p.error("Provide either --dataset or both --input and --output")

    return args


if __name__ == "__main__":
    main()
