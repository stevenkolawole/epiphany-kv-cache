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
       - Label tokens s..s+W-1 as important=True if the answer flipped.
         OR semantics for overlapping windows: a token is important if ANY
         window covering it caused a flip (label stays 1 once set).
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
- We mask by replacing with pad_token_id (occlusion, not truncation).
  Sequence length is preserved up to answer_start so positional encodings
  and the model's contextual understanding of all other positions are intact.
  This is the standard "occlusion" approach used in interpretability research.

- answer_start is the token position where the final answer section begins
  (after </think> for DeepSeek-R1, or estimated from the last \\boxed{ for
  other models).  We feed token_ids[:answer_start] with the window replaced
  by pads as context, then ask the model to generate the answer from scratch.
  This means: every masked-inference call feeds the same-length context
  (answer_start tokens), with the only difference being the content of the
  masked window.  There is no position proxy — a window near the start and a
  window near the end of the trace both feed the same total context length.

- Previously the code truncated at mask_start and regenerated from there.
  That tested "how much prefix is needed?" (a position test), not "is this
  window's content important?" (a content test).  The old approach inflated
  importance scores for early tokens regardless of their actual content.

- We only mask the *reasoning* portion (prompt_len <= pos < answer_start).
  The prompt is never masked (always important by definition).  The answer
  section (answer_start..total_len) is never masked — we regenerate it.

- Window size W=32 and stride S=16 are defaults tuned for reasoning traces
  of ~4k–32k tokens.  Smaller W gives higher resolution but more forward
  passes; larger W reduces cost but may give coarser labels.

- Runtime: O(seq_len / stride) forward passes per trace.  Each pass prefills
  answer_start tokens then generates up to max_new_tokens.  For a 16k-token
  trace: ~500 passes × ~30s each ≈ 4–6 hours per trace on a single A100.
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

    def norm(s: str) -> str:
        s = s.strip()
        s = s.replace("\\dfrac", "\\frac")                          # \dfrac == \frac
        s = re.sub(r"\\left\s*([(\[{|])", r"\1", s)                 # \left( → (
        s = re.sub(r"\\right\s*([)\]}|])", r"\1", s)                # \right) → )
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)                  # \text{X} → X
        m = re.match(r"^[a-zA-Z]\s*=\s*(.+)$", s.strip())          # x=5 → 5
        if m:
            s = m.group(1)
        return s.lower().replace(" ", "").replace(",", "")

    if norm(pred) == norm(gt):
        return True

    # Order-insensitive set comparison: "-2,1" vs "1,-2"
    pred_parts = sorted(norm(p) for p in pred.split(","))
    gt_parts   = sorted(norm(g) for g in gt.split(","))
    return pred_parts == gt_parts


# ── Answer boundary detection ─────────────────────────────────────────────────

def find_answer_start(
    token_ids: List[int],
    tokenizer,
    generated_text: str,
    prompt_len: int,
) -> int:
    """
    Find the token index where the final answer section begins — i.e. the
    point after which we ask the model to regenerate the answer during masked
    inference.  Everything before this index is the "reasoning context" we
    feed (with the target window occluded).

    Strategy 1 — DeepSeek-R1 </think> boundary:
        Search for the token sequence that encodes "</think>" in token_ids.
        The answer section starts immediately after it (skipping whitespace).

    Strategy 2 — last \\boxed{ in generated_text:
        Tokenize the text up to the last \\boxed{ to estimate the token index.

    Strategy 3 — fallback:
        Use total_len - 64 (assume the answer is at most 64 tokens).
    """
    total_len = len(token_ids)

    # Strategy 1: </think> boundary (DeepSeek-R1 style)
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    n = len(think_end_ids)
    for i in range(total_len - n, prompt_len - 1, -1):  # search from end
        if token_ids[i : i + n] == think_end_ids:
            pos = i + n
            # Skip whitespace/newline tokens immediately after </think>
            while pos < total_len and not tokenizer.decode([token_ids[pos]]).strip():
                pos += 1
            return pos

    # Strategy 2: last \boxed{ in generated text
    last_boxed = generated_text.rfind("\\boxed{")
    if last_boxed > 0:
        text_before = generated_text[:last_boxed]
        ids_before  = tokenizer.encode(text_before, add_special_tokens=False)
        candidate   = prompt_len + len(ids_before)
        if prompt_len < candidate < total_len:
            return candidate

    # Fallback: 64 tokens from end
    return max(prompt_len + 1, total_len - 64)


# ── Masked inference ──────────────────────────────────────────────────────────

class MultiTokenFiller:
    """Occlusion filler that writes a *different* random in-vocabulary token at
    every masked position.

    The single-token `random` arm did not test what it was built to test. Filling
    a window with 32 copies of one id is another degenerate repeated pattern,
    structurally the same manipulation as the `pad` arm, which is why the two came
    out statistically indistinguishable (p=0.55). That leaves the interesting
    question open: does EOS flip more windows because it perturbs harder, or
    because it means *stop*?

    A window of 32 distinct tokens is a genuine high-entropy perturbation carrying
    no stop semantics, so it separates the two. If it flips at EOS-like rates, EOS
    is simply a high-gain probe and the published bands stand; if it flips at
    pad-like rates, EOS's stop semantics are doing the work.

    Sampling is keyed on the window's start position rather than drawn from a
    running stream, so a resumed run reproduces the same fill as an uninterrupted
    one.
    """

    def __init__(self, tokenizer, seed: int = 0):
        import random as _r
        self._random = _r.Random
        self.seed = seed
        special = set(tokenizer.all_special_ids)
        self.vocab = [i for i in range(tokenizer.vocab_size) if i not in special]

    def sample(self, n: int, key: int) -> List[int]:
        # Seeded from a string, not a tuple: Random() rejects tuple seeds on 3.12.
        rng = self._random(f"{self.seed}:{key}")
        return [rng.choice(self.vocab) for _ in range(n)]


def run_masked_inference(
    model,
    tokenizer,
    full_ids: torch.Tensor,    # (1, total_len) — prompt + generated
    prompt_len: int,
    mask_start: int,           # first position to occlude (>= prompt_len)
    mask_end: int,             # one past last position to occlude (<= answer_start)
    answer_start: int,         # feed [:answer_start] as context, generate from here
    max_new_tokens: int,
    pad_id: int,
    device: torch.device,
    mask_id: Optional[int] = None,
) -> Optional[str]:
    """
    Occlude token_ids[mask_start:mask_end] with pad_id, feed the full modified
    reasoning context up to answer_start, then generate the answer from scratch.

    This is a content test, not a position test:
    - Every call feeds the same context length (answer_start tokens).
    - The only variable is the content of positions [mask_start, mask_end).
    - A window at position 200 and a window at position 8000 have identical
      positional context; the only difference is what's in that window.

    The old truncation approach fed [:mask_start], which meant early windows
    always had less context — a position proxy, not a content test.

    Returns the extracted \\boxed{} answer, or None if not found.
    """
    modified_ids = full_ids.clone()
    # The occlusion filler is deliberately separate from generate()'s pad_token_id
    # below: the filler is the manipulation under test, pad_token_id is only HF's
    # batching argument. Defaults to pad_id so existing behaviour is unchanged.
    if hasattr(mask_id, "sample"):
        # High-entropy arm: a distinct random token per position (see MultiTokenFiller).
        fill = mask_id.sample(mask_end - mask_start, mask_start)
        modified_ids[0, mask_start:mask_end] = torch.tensor(
            fill, dtype=modified_ids.dtype)
    else:
        modified_ids[0, mask_start:mask_end] = pad_id if mask_id is None else mask_id

    context_ids = modified_ids[:, :answer_start].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=context_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0, answer_start:].tolist()
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
    mask_id: Optional[int] = None,
) -> Dict:
    """
    Add an 'importance' field to the trace dict.
    Returns the augmented trace.
    """
    token_ids    = trace["token_ids"]
    prompt_len   = trace["prompt_len"]
    ground_truth = trace["ground_truth"]
    total_len    = len(token_ids)

    # Find where the final answer section starts (after </think> or last \boxed{).
    # Windows only slide over the reasoning portion [prompt_len, answer_start).
    answer_start = find_answer_start(
        token_ids, tokenizer, trace.get("generated_text", ""), prompt_len
    )
    # answer_start must be strictly after the prompt and before total_len
    answer_start = max(prompt_len + 1, min(answer_start, total_len - 1))

    # Initialise all positions as unknown (-1)
    importance = [-1] * total_len
    # Prompt tokens are structurally important by definition
    for i in range(prompt_len):
        importance[i] = 1
    # Answer tokens (answer_start..total_len) are left as -1 (not tested; we regenerate them)

    full_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # (1, total_len)

    # Slide window over the *reasoning* portion only [prompt_len, answer_start)
    reasoning_len = answer_start - prompt_len
    if reasoning_len <= 0:
        trace["importance"] = importance
        trace["label_stats"] = {"answer_start": answer_start, "windows_tested": 0,
                                 "windows_flipped": 0, "flip_rate": 0.0, "important_frac": 0.0}
        return trace

    windows_tested  = 0
    windows_flipped = 0

    total_windows = max(1, (reasoning_len - window_size) // stride + 1)
    pos = prompt_len
    with tqdm(total=total_windows, desc="  windows", leave=False, unit="win") as pbar:
        while pos < answer_start:
            end = min(pos + window_size, answer_start)  # never mask into answer section

            masked_answer = run_masked_inference(
                model=model,
                tokenizer=tokenizer,
                full_ids=full_ids,
                prompt_len=prompt_len,
                mask_start=pos,
                mask_end=end,
                answer_start=answer_start,
                max_new_tokens=max_new_tokens,
                pad_id=pad_id,
                device=device,
                mask_id=mask_id,
            )

            flipped = not answers_match(masked_answer, ground_truth)
            label = 1 if flipped else 0
            for i in range(pos, end):
                # OR semantics: once a position is covered by a flipping window,
                # it stays important even if a later overlapping window does not flip.
                if importance[i] == -1:
                    importance[i] = label
                else:
                    importance[i] = max(importance[i], label)

            windows_tested += 1
            if flipped:
                windows_flipped += 1

            pbar.update(1)
            pbar.set_postfix(flips=windows_flipped, flip_rate=f"{windows_flipped/windows_tested:.2f}")
            pos += stride

    trace["importance"] = importance
    trace["label_stats"] = {
        "answer_start":    answer_start,
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

    # Resolve the occlusion filler. It must encode to exactly one token: the
    # method's invariant is that every occluded call feeds an identical context
    # length, so a multi-token filler would silently break the comparison.
    choice = args.mask_token
    if choice == "eos":
        mask_id = pad_id
    elif choice == "pad":
        mask_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
        if mask_id is None or mask_id == tokenizer.unk_token_id:
            sys.exit("--mask_token pad: <|finetune_right_pad_id|> not in this vocab")
    elif choice == "random":
        import random as _r
        rng = _r.Random(args.mask_seed)
        special = set(tokenizer.all_special_ids)
        while True:
            cand = rng.randrange(tokenizer.vocab_size)
            if cand not in special:
                mask_id = cand
                break
    elif choice == "randmulti":
        mask_id = MultiTokenFiller(tokenizer, args.mask_seed)
    else:
        enc = tokenizer.encode(choice, add_special_tokens=False)
        if len(enc) != 1:
            sys.exit(f"--mask_token {choice!r} encodes to {len(enc)} tokens; need exactly 1")
        mask_id = enc[0]
    if hasattr(mask_id, "sample"):
        print(f"Occlusion filler: {choice} -> distinct random token per position "
              f"(seed {args.mask_seed}, {len(mask_id.vocab)} eligible ids); "
              f"generate pad_token_id stays {pad_id}")
    else:
        print(f"Occlusion filler: {choice} -> id {mask_id} "
              f"({tokenizer.convert_ids_to_tokens([mask_id])[0]!r}); "
              f"generate pad_token_id stays {pad_id}")

    # Resume support: skip traces already labelled
    done_problems: set = set()
    if output_path.exists():
        with open(output_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    try:
                        done_problems.add(json.loads(_line)["problem"])
                    except Exception:
                        pass
    if done_problems:
        n_skip = sum(1 for t in traces if t["problem"] in done_problems)
        traces = [t for t in traces if t["problem"] not in done_problems]
        print(f"  Resuming: {n_skip} traces already labelled, {len(traces)} remaining.")
    if not traces:
        print("All traces already labelled. Nothing to do.")
        sys.exit(0)
    write_mode = "a" if done_problems else "w"

    # Label traces
    n_written = 0
    with open(output_path, write_mode) as f_out:
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
                    mask_id=mask_id,
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
                f"  len={total_len:>6}  ans_start={stats.get('answer_start',0):>6}"
                f"  windows={stats.get('windows_tested',0):>4}"
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
    p.add_argument("--mask_token", default="eos",
                   help="Occlusion filler: 'eos' (current behaviour), 'pad' "
                        "(<|finetune_right_pad_id|>), 'random' (one fixed random "
                        "in-vocab token repeated), 'randmulti' (a distinct random "
                        "in-vocab token per position -- the high-entropy control), "
                        "or a literal string that encodes to exactly one token.")
    p.add_argument("--mask_seed", type=int, default=0,
                   help="Seed for --mask_token random/randmulti, so the arm is "
                        "reproducible.")
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
