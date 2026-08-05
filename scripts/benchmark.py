#!/usr/bin/env python3
"""
benchmark.py

Phase 1: accuracy vs. cache-size curves for all eviction policies.

For each (method, cache_size) combination, runs all problems, applies the
eviction policy at every decode step, scores answers against ground truth,
and records accuracy.  Results are written incrementally so partial runs
are not lost.

Methods
-------
  none                — full KV cache (no eviction); accuracy ceiling
  h2o                 — H2OEviction (cumulative attention)
  thinKV              — ThinKVEviction (segment R/E/T classification)
  raas                — RaaSEviction (LRU decode eviction)
  hs_variance         — HSVarianceEviction (l10_rolling64 − l21_rolling64)
  hs_variance_detrend — DetrendendHSVarianceEviction (rolling z-score detrending; FA2-compatible)
  band_adaptive_hs    — BandAdaptiveHSEviction (all Band A/B layers; empirical weights; FA2-compatible)
  attn_hs_product     — AttentionHSProductEviction (cumulative attn + detrended HS; eager only)
  hybrid_seg_hs       — HybridSegmentHSEviction (ThinKV segments + HS within-segment; eager only)
  kv_val              — KVValVarianceEviction (value-vector variance)
  kv_key              — KVKeyVarianceEviction (key-vector variance)
  lag_kv_key          — LagKVKeyVarianceEviction (lag-normalised key variance)
  lag_kv              — LagKVEviction (lag-normalised key + value)

Usage
-----
    python scripts/benchmark.py --dataset math500 --n_samples 50 \\
        --cache_sizes 512 1024 2048 4096 \\
        --methods none h2o thinKV raas hs_variance kv_val kv_key lag_kv_key lag_kv \\
        --output results/benchmark_math500.json

    # AIME2024 (30 problems)
    python scripts/benchmark.py --dataset aime2024 --n_samples 30 \\
        --cache_sizes 512 1024 2048 4096 --max_new_tokens 16384 \\
        --output results/benchmark_aime2024.json

Output
------
JSON file structured as:
    {
      "meta": { "model": "...", "dataset": "...", ... },
      "results": {
        "<method>": {
          "<cache_size>": {
            "accuracy": 0.72,
            "n_correct": 36,
            "n_total": 50,
            "per_problem": [ {"problem": "...", "correct": true, "pred": "..."}, ... ]
          }
        }
      }
    }
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

try:
    from eviction import (
        EvictionConfig,
        H2OEviction,
        ThinKVEviction,
        ThinKVFaithfulEviction,
        RaaSEviction,
        HSVarianceEviction,
        KVValVarianceEviction,
        KVKeyVarianceEviction,
        LagKVKeyVarianceEviction,
        LagKVEviction,
        DetrendendHSVarianceEviction,
        BandAdaptiveHSEviction,
        AttentionHSProductEviction,
        HybridSegmentHSEviction,
        KVSegHSEviction,
        KVSegmentHSEviction,
        RKVEviction,
        LongFlowEviction,
    )
except ImportError as e:
    sys.exit(f"Cannot import eviction module: {e}")

try:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("Missing dependencies: pip install datasets transformers")


# ── GPU memory helpers ────────────────────────────────────────────────────────

def _reset_gpu_peak():
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(i)
        except RuntimeError:
            break  # CUDA context not initialized for this device


def _peak_gpu_mb() -> float:
    total = 0.0
    for i in range(torch.cuda.device_count()):
        try:
            total += torch.cuda.max_memory_allocated(i)
        except RuntimeError:
            break
    return total / 1024 ** 2


# ── KV cache format helpers ───────────────────────────────────────────────────

def _as_legacy_kv(past_key_values) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Normalise past_key_values to a list of (key, value) pairs."""
    if hasattr(past_key_values, "layers"):
        return [(layer.keys, layer.values) for layer in past_key_values.layers]
    if hasattr(past_key_values, "key_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    return [(layer[0], layer[1]) for layer in past_key_values]


def _to_model_kv(kv_pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Convert a list of (k, v) pairs to a DynamicCache for model input.

    Uses DynamicCache.from_legacy_cache() which calls update() per layer,
    correctly setting _seen_tokens from actual tensor shapes.  This avoids
    a stale _seen_tokens after eviction: with the old ddp_cache_data
    constructor _seen_tokens could be one ahead of the actual cache size,
    making the model build a causal mask one token longer than the KV
    (causing "size of tensor a (N+1) must match tensor b (N)" errors).
    """
    from transformers import DynamicCache
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_pairs):
        cache.update(k, v, layer_idx)
    return cache


# ── Answer helpers (copied from collect_traces.py) ────────────────────────────

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_DEFAULT_HF_CACHE = "/data/hf_cache/skolawol"


def _hf_cache() -> str:
    return os.environ.get("HF_HOME", _DEFAULT_HF_CACHE)


def extract_boxed(text: str) -> Optional[str]:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def answers_match(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False

    def norm(s: str) -> str:
        s = s.strip()
        s = s.replace("\\dfrac", "\\frac")
        s = re.sub(r"\\left\s*([(\[{|])", r"\1", s)
        s = re.sub(r"\\right\s*([)\]}|])", r"\1", s)
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
        m = re.match(r"^[a-zA-Z]\s*=\s*(.+)$", s.strip())
        if m:
            s = m.group(1)
        return s.lower().replace(" ", "").replace(",", "")

    if norm(pred) == norm(gt):
        return True
    pred_parts = sorted(norm(p) for p in pred.split(","))
    gt_parts   = sorted(norm(g) for g in gt.split(","))
    return pred_parts == gt_parts


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_problems(dataset: str, n_samples: int, start_idx: int = 0) -> List[Dict]:
    if dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=_hf_cache())
        problems = []
        for item in ds:
            ans = extract_boxed(item["solution"])
            if ans:
                problems.append({"problem": item["problem"].strip(), "ground_truth": ans})
            if len(problems) >= start_idx + n_samples:
                break
    elif dataset == "aime2024":
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train", cache_dir=_hf_cache())
        problems = [
            {"problem": item["Problem"].strip(), "ground_truth": str(item["Answer"]).strip()}
            for item in ds
        ][:start_idx + n_samples]
    elif dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=_hf_cache())
        _re = re.compile(r"####\s*(-?[\d,]+)")
        problems = []
        for item in ds:
            m = _re.search(item["answer"])
            if m:
                problems.append({"problem": item["question"].strip(),
                                  "ground_truth": m.group(1).replace(",", "")})
            if len(problems) >= start_idx + n_samples:
                break
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    problems = problems[start_idx:start_idx + n_samples]
    print(f"  Loaded {len(problems)} problems from {dataset} "
          f"(shard [{start_idx}, {start_idx + n_samples})).")
    return problems


# ── Prompt construction ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt_ids(tokenizer, problem: str, device) -> torch.Tensor:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt")["input_ids"].to(device)


# ── Eviction factory ──────────────────────────────────────────────────────────

# Methods that need the full softmax attention matrix (eager attn only).
ATTN_ONLY_METHODS = {"h2o", "thinKV", "raas", "r_kv", "longflow", "thinkv_faithful"}
# Methods that need output_hidden_states=True only (FA2-compatible).
HS_ONLY_METHODS   = {"hs_variance", "hs_variance_detrend", "band_adaptive_hs", "kv_seg_hs", "kv_seg_hs_entropy"}
# Methods that need BOTH attention matrix AND hidden states (eager only).
BOTH_METHODS      = {"attn_hs_product", "hybrid_seg_hs"}
# Methods that only read past_key_values (compatible with flash attn).
KV_METHODS        = {"kv_val", "kv_key", "lag_kv_key", "lag_kv"}

ATTN_METHODS  = ATTN_ONLY_METHODS | BOTH_METHODS
HS_METHODS    = HS_ONLY_METHODS | BOTH_METHODS
ALL_METHODS   = {"none"} | ATTN_ONLY_METHODS | HS_ONLY_METHODS | BOTH_METHODS | KV_METHODS


def make_eviction(method: str, cache_size: int, keep_recent_k: int = 128):
    """Return a fresh eviction object for the given method and cache_size."""
    cfg = EvictionConfig(cache_size=cache_size, keep_recent_k=keep_recent_k)
    if method == "none":
        return None
    if method == "h2o":
        return H2OEviction(cfg)
    if method == "thinKV":
        return ThinKVEviction(cfg)
    if method == "raas":
        return RaaSEviction(cfg)
    if method == "hs_variance":
        return HSVarianceEviction(cfg)
    if method == "hs_variance_detrend":
        return DetrendendHSVarianceEviction(cfg)
    if method == "band_adaptive_hs":
        return BandAdaptiveHSEviction(cfg)
    if method == "attn_hs_product":
        return AttentionHSProductEviction(cfg)
    if method == "hybrid_seg_hs":
        return HybridSegmentHSEviction(cfg)
    if method == "kv_val":
        return KVValVarianceEviction(cfg)
    if method == "kv_key":
        return KVKeyVarianceEviction(cfg)
    if method == "lag_kv_key":
        return LagKVKeyVarianceEviction(cfg)
    if method == "lag_kv":
        return LagKVEviction(cfg)
    if method == "thinkv_faithful":
        return ThinKVFaithfulEviction(cfg)
    if method == "kv_seg_hs":
        return KVSegHSEviction(cfg)
    if method == "kv_seg_hs_entropy":
        return KVSegmentHSEviction(cfg)
    if method == "r_kv":
        return RKVEviction(cfg)
    if method == "longflow":
        return LongFlowEviction(cfg)
    raise ValueError(f"Unknown method: {method}")


# ── Single-problem runner ─────────────────────────────────────────────────────

def run_one(
    model,
    tokenizer,
    problem: Dict,
    method: str,
    eviction,
    max_new_tokens: int,
    device: torch.device,
) -> Dict:
    """
    Run one problem with the given eviction policy.  Returns a dict with
    'correct', 'pred_answer', and 'n_tokens_generated'.
    """
    need_attn = method in ATTN_METHODS
    need_hs   = method in HS_METHODS
    need_both = method in BOTH_METHODS

    prompt_ids = build_prompt_ids(tokenizer, problem["problem"], device)
    prompt_len = prompt_ids.shape[1]

    _reset_gpu_peak()
    t0 = time.perf_counter()

    # ── Prefill ───────────────────────────────────────────────────────────
    with torch.no_grad():
        prefill_out = model(
            input_ids=prompt_ids,
            use_cache=True,
            output_attentions=need_attn,
            output_hidden_states=need_hs,
        )

    past_kv = _as_legacy_kv(prefill_out.past_key_values)

    # Initialise eviction state
    _hs_eviction_classes = (
        HSVarianceEviction, DetrendendHSVarianceEviction, BandAdaptiveHSEviction,
        AttentionHSProductEviction, HybridSegmentHSEviction, KVSegHSEviction,
        KVSegmentHSEviction,
    )
    if eviction is not None:
        if hasattr(eviction, "reset"):
            if isinstance(eviction, (H2OEviction,)):
                eviction.reset()  # no prefill_len arg
            else:
                eviction.reset(prefill_len=prompt_len)
        if isinstance(eviction, _hs_eviction_classes) and need_hs:
            eviction.set_prefill_end(prefill_out.hidden_states)

    next_token = prefill_out.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)  # (1, 1)
    del prefill_out
    torch.cuda.empty_cache()

    # ── Decode loop ───────────────────────────────────────────────────────
    generated_ids: List[int] = []
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        token_id = int(next_token[0, 0])
        generated_ids.append(token_id)
        if token_id == eos_id:
            break

        with torch.no_grad():
            step_out = model(
                input_ids=next_token,
                past_key_values=_to_model_kv(past_kv),
                use_cache=True,
                output_attentions=need_attn,
                output_hidden_states=need_hs,
            )

        past_kv = _as_legacy_kv(step_out.past_key_values)

        # Apply eviction — eviction methods expect the legacy tuple format.
        if eviction is not None:
            kv_tuple = tuple(past_kv)
            if need_both:
                evicted = eviction.evict_past_key_values(
                    kv_tuple, step_out.attentions, step_out.hidden_states
                )
            elif need_attn:
                evicted = eviction.evict_past_key_values(kv_tuple, step_out.attentions)
            elif need_hs:
                evicted = eviction.evict_past_key_values(kv_tuple, step_out.hidden_states)
            else:
                evicted = eviction.evict_past_key_values(kv_tuple)
            past_kv = _as_legacy_kv(evicted)

        next_token = step_out.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        del step_out
        if len(generated_ids) % 512 == 0:
            torch.cuda.empty_cache()

    wall_time_s = time.perf_counter() - t0
    peak_gpu_mb = _peak_gpu_mb()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    pred_answer = extract_boxed(generated_text)
    correct = answers_match(pred_answer, problem["ground_truth"])

    return {
        "correct":            correct,
        "pred_answer":        pred_answer,
        "n_tokens_generated": len(generated_ids),
        "wall_time_s":        round(wall_time_s, 2),
        "peak_gpu_mb":        round(peak_gpu_mb, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",          default="deepseek-ai/deepseek-r1-distill-llama-8b")
    p.add_argument("--dataset",        choices=["math500", "aime2024", "gsm8k"],
                   default="math500")
    p.add_argument("--n_samples",      type=int, default=50)
    p.add_argument("--start_idx",      type=int, default=0,
                   help="First problem index; with --n_samples defines a shard [start, start+n).")
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--cache_sizes",    type=int, nargs="+",
                   default=[512, 1024, 2048, 4096],
                   help="KV cache budgets in tokens to sweep over")
    p.add_argument("--keep_recent_k",  type=int, default=128,
                   help="Tokens always kept in recency window (default: 128)")
    p.add_argument("--methods",        nargs="+",
                   default=sorted(ALL_METHODS),
                   choices=sorted(ALL_METHODS),
                   help="Eviction methods to benchmark")
    p.add_argument("--output",         type=Path, default=None,
                   help="Output JSON path (default: results/benchmark_<dataset>.json)")
    p.add_argument("--resume",         action="store_true",
                   help="Resume: skip (method, cache_size) pairs already in output file")
    p.add_argument("--attn_impl",      default="auto",
                   choices=["auto", "eager", "sdpa", "flash_attention_2"],
                   help="Attention implementation override. 'auto' uses eager when any "
                        "attention-based method is present, sdpa otherwise.")
    return p.parse_args()


def main():
    args = parse_args()

    output_path = args.output or Path(f"results/benchmark_{args.dataset}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine attention implementation
    needs_eager = any(m in ATTN_METHODS for m in args.methods)
    if args.attn_impl == "auto":
        attn_impl = "eager" if needs_eager else "sdpa"
    else:
        attn_impl = args.attn_impl
        if attn_impl != "eager" and needs_eager:
            print(f"WARNING: --attn_impl={attn_impl} but methods "
                  f"{ATTN_METHODS & set(args.methods)} require eager attention. "
                  f"Forcing eager.", file=sys.stderr)
            attn_impl = "eager"

    print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=_hf_cache())
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
        cache_dir=_hf_cache(),
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model loaded on {device} (attn: {attn_impl})")
    if device.type == "cpu" and torch.cuda.is_available():
        sys.exit(f"ERROR: model loaded on CPU despite CUDA being available — "
                 f"likely a CUDA driver mismatch on {os.uname().nodename}. "
                 f"Resubmit to get a different node.")

    print(f"Loading {args.dataset} ...")
    problems = load_problems(args.dataset, args.n_samples, args.start_idx)

    # Load existing results for resume
    existing: Dict = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f).get("results", {})
        print(f"  Resuming — {sum(len(v) for v in existing.values())} "
              f"(method, cache_size) pairs already done.")

    results: Dict = {m: {} for m in args.methods}
    for m, v in existing.items():
        if m in results:
            results[m] = v

    meta = {
        "model":          args.model,
        "dataset":        args.dataset,
        "n_samples":      len(problems),
        "max_new_tokens": args.max_new_tokens,
        "keep_recent_k":  args.keep_recent_k,
        "cache_sizes":    args.cache_sizes,
        "methods":        args.methods,
        "attn_impl":      attn_impl,
    }

    def _save():
        tmp = output_path.with_suffix(".tmp.json")
        with open(tmp, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        tmp.rename(output_path)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    def _is_complete(entry: dict) -> bool:
        """A result is complete only if at least one problem ran without error."""
        pp = entry.get("per_problem", [])
        return len(pp) > 0 and any(r.get("n_tokens_generated", 0) > 0 for r in pp)

    for method in args.methods:
        for cache_size in args.cache_sizes:
            key = str(cache_size)

            # "none" is the unlimited-cache baseline — cache_size is irrelevant.
            # Run it once (for the first cache_size), then copy to remaining slots.
            if method == "none":
                if _is_complete(results.get("none", {}).get(key, {})):
                    print(f"  Skipping none @ cache={cache_size} (already done)")
                    continue
                # Check if we already ran none for any other cache_size.
                existing_none = next(
                    (v for v in results.get("none", {}).values() if _is_complete(v)),
                    None,
                )
                if existing_none is not None:
                    results["none"][key] = existing_none
                    _save()
                    print(f"  none @ cache={cache_size}: reusing unlimited-cache result")
                    continue

                print(f"\n{'='*60}")
                print(f"  Method: none             Cache size: unlimited (no eviction)")
                print(f"{'='*60}")

            else:
                if _is_complete(results.get(method, {}).get(key, {})):
                    print(f"  Skipping {method} @ cache={cache_size} (already done)")
                    continue

                print(f"\n{'='*60}")
                print(f"  Method: {method:15s}  Cache size: {cache_size}")
                print(f"{'='*60}")

            per_problem = []
            n_correct = 0

            desc = "none@unlimited" if method == "none" else f"{method}@{cache_size}"
            for i, prob in enumerate(tqdm(problems, desc=desc)):
                eviction = make_eviction(method, cache_size, args.keep_recent_k)
                try:
                    res = run_one(model, tokenizer, prob, method, eviction,
                                  args.max_new_tokens, device)
                except Exception as exc:
                    print(f"\n  [ERROR] problem {i}: {exc}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    res = {"correct": False, "pred_answer": None, "n_tokens_generated": 0,
                           "error": str(exc)}
                    # Best-effort GPU recovery after a CUDA error.  The context
                    # may be corrupted; clearing cache + syncing gives the next
                    # problem the best chance of running cleanly.
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                finally:
                    # Always release the eviction object's tensor refs and flush
                    # the GPU allocator between problems, error or not.
                    del eviction
                    torch.cuda.empty_cache()

                if res.get("correct"):
                    n_correct += 1

                per_problem.append({
                    "problem":            prob["problem"][:80],
                    "ground_truth":       prob["ground_truth"],
                    "pred_answer":        res.get("pred_answer"),
                    "correct":            res.get("correct"),
                    "n_tokens_generated": res.get("n_tokens_generated", 0),
                    "wall_time_s":        res.get("wall_time_s"),
                    "peak_gpu_mb":        res.get("peak_gpu_mb"),
                    "error":              res.get("error"),
                })

            accuracy = n_correct / len(problems) if problems else 0.0
            print(f"  Accuracy: {n_correct}/{len(problems)} = {accuracy:.1%}")

            times = [r["wall_time_s"] for r in per_problem if r.get("wall_time_s") is not None]
            mems  = [r["peak_gpu_mb"]  for r in per_problem if r.get("peak_gpu_mb")  is not None]

            results[method][key] = {
                "accuracy":         accuracy,
                "n_correct":        n_correct,
                "n_total":          len(problems),
                "mean_wall_time_s": round(sum(times) / len(times), 2) if times else None,
                "mean_peak_gpu_mb": round(sum(mems)  / len(mems),  1) if mems  else None,
                "per_problem":      per_problem,
            }
            _save()

    print(f"\nResults written to {output_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Method':<16}", end="")
    for cs in args.cache_sizes:
        print(f"  {cs:>6}", end="")
    print()
    print("-" * (16 + 8 * len(args.cache_sizes)))
    for m in args.methods:
        print(f"{m:<16}", end="")
        for cs in args.cache_sizes:
            acc = results.get(m, {}).get(str(cs), {}).get("accuracy")
            print(f"  {acc:.1%}" if acc is not None else "     -", end="")
        print()


if __name__ == "__main__":
    main()
