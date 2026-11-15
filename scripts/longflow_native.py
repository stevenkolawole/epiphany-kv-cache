#!/usr/bin/env python3
"""
longflow_native.py

Runs LongFlow's released code (yisunlp/LongFlow, commit 89a68d3) on MATH-500 with this repo's prompt, answer
extraction and scorer, and writes a benchmark.py-style JSON so the cells pair with EpiKV cells problem by problem.

LongFlow's attention, cache and Triton kernel are imported unmodified from --longflow_dir; this file only drives
`model.generate` the way their example_multi_batch.py does (batch of one, fresh OursCache(cache_budget=K) per
problem, eager attention) and adds one API shim: their OursAttention.forward calls self.q_norm / self.k_norm,
which exist on Qwen3Attention but not on Qwen2Attention (R1-Distill-Qwen). For Qwen2 models those are set to
nn.Identity, which is what Qwen2 computes. Scoring and eviction logic are not touched.

Needs torch 2.4 / transformers 4.51 / triton (their pins). Usage:
  python scripts/longflow_native.py --longflow_dir <LongFlow> --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
      --n_samples 5 --cache_size 1024 --max_new_tokens 16384 --output out.json
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # prompt (build_prompt_ids), extract_boxed, answers_match, load_problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--longflow_dir", required=True)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--dataset", default="math500")
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--cache_size", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=16384)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                   help="float16 is what LongFlow's own scripts use")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    sys.path.insert(0, args.longflow_dir)
    from models.replace import replace_ours  # unmodified
    Cache = replace_ours()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert torch.cuda.is_available(), "NO CUDA: abort"
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=bm._hf_cache())
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), attn_implementation="eager",
        cache_dir=bm._hf_cache()).to("cuda").eval()
    device = next(model.parameters()).device
    assert device.type == "cuda", f"model on {device}: abort"

    shims = []
    for layer in model.model.layers:
        a = layer.self_attn
        if not hasattr(a, "q_norm"):
            a.q_norm = nn.Identity()
            a.k_norm = nn.Identity()
            if "q_norm/k_norm = nn.Identity (absent in Qwen2Attention)" not in shims:
                shims.append("q_norm/k_norm = nn.Identity (absent in Qwen2Attention)")
        if not hasattr(a, "sliding_window"):
            a.sliding_window = None
            if "sliding_window = None" not in shims:
                shims.append("sliding_window = None")
    print("shims:", shims, flush=True)

    problems = bm.load_problems(args.dataset, args.n_samples, args.start_idx)  # already sliced to [start_idx, start_idx+n)
    key = str(args.cache_size)
    per_problem = []
    if args.resume and args.output.exists():
        prev = json.load(open(args.output))["results"]["longflow_native"][key]
        if prev.get("partial"):
            per_problem = prev["per_problem"]
    meta = {"model": args.model, "dataset": args.dataset, "n_samples": len(problems), "start_idx": args.start_idx,
            "max_new_tokens": args.max_new_tokens, "cache_sizes": [args.cache_size], "methods": ["longflow_native"],
            "attn_impl": "eager", "dtype": args.dtype, "shims": shims,
            "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
            "gpu": torch.cuda.get_device_name(0)}

    def save(entry):
        tmp = args.output.with_suffix(".tmp.json")
        json.dump({"meta": meta, "results": {"longflow_native": {key: entry}}}, open(tmp, "w"), indent=2)
        tmp.rename(args.output)

    for i, prob in enumerate(problems):
        if i < len(per_problem):
            continue
        try:
            ids = bm.build_prompt_ids(tok, prob["problem"], device)
            cache = Cache(cache_budget=args.cache_size)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new_tokens,
                                     do_sample=False, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize(); wall = time.perf_counter() - t0
            gen = out[0, ids.shape[1]:].tolist()
            pred = bm.extract_boxed(tok.decode(gen, skip_special_tokens=True))
            res = {"pred_answer": pred, "correct": bm.answers_match(pred, prob["ground_truth"]),
                   "n_tokens_generated": len(gen), "wall_time_s": round(wall, 2),
                   "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1), "prompt_len": int(ids.shape[1]),
                   "hit_eos": bool(gen and gen[-1] == tok.eos_token_id)}
            del cache, out
        except Exception as exc:
            traceback.print_exc()
            res = {"pred_answer": None, "correct": False, "n_tokens_generated": 0, "error": str(exc)}
        torch.cuda.empty_cache()
        per_problem.append({"problem": prob["problem"][:80], "ground_truth": prob["ground_truth"],
                            "problem_idx": args.start_idx + i, **res})
        print(f"[{i}] correct={res['correct']} tok={res['n_tokens_generated']} t={res.get('wall_time_s')}", flush=True)
        save({"partial": True, "n_total": len(problems), "per_problem": per_problem})

    nc = sum(1 for r in per_problem if r["correct"])
    save({"accuracy": nc / len(problems), "n_correct": nc, "n_total": len(problems),
          "mean_wall_time_s": round(sum(r.get("wall_time_s") or 0 for r in per_problem) / len(problems), 2),
          "per_problem": per_problem})
    errs = [r for r in per_problem if r.get("error")]
    print(f"DONE {nc}/{len(problems)} errors={len(errs)}", flush=True)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
