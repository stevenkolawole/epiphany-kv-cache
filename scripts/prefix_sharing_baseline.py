"""Stock vLLM on a shared-prefix workload, prefix caching on and off.

One long document is the shared prefix; each request appends a different question and
generates one token, so the request time is the prefill time. With prefix caching on,
the requests after the first should hit the cached prefix blocks.
"""
import argparse, json, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
ap.add_argument("--prefix_tokens", type=int, default=4096)
ap.add_argument("--n_requests", type=int, default=8)
ap.add_argument("--prefix_caching", type=int, required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

from vllm import LLM, SamplingParams

llm = LLM(model=args.model, enforce_eager=True, gpu_memory_utilization=0.85,
          max_model_len=args.prefix_tokens + 1024, dtype="bfloat16", max_num_seqs=1,
          enable_prefix_caching=bool(args.prefix_caching), disable_log_stats=False)
tok = llm.get_tokenizer()

sentence = "The committee reviewed the budget for the northern district and recorded each decision. "
ids = tok(sentence * (args.prefix_tokens // 14 + 1), add_special_tokens=False)["input_ids"][:args.prefix_tokens]
prefix = tok.decode(ids)
questions = [f"Question {i}: what was decision number {i}?" for i in range(args.n_requests)]
sp = SamplingParams(temperature=0.0, max_tokens=1)

def counters():
    out = {}
    for m in llm.get_metrics():
        if m.name in ("vllm:prefix_cache_queries", "vllm:prefix_cache_hits"):
            out[m.name] = m.value
    return out

rows = []
for q in questions:
    t = time.time()
    r = llm.generate([prefix + "\n" + q], sp, use_tqdm=False)[0]
    rows.append({"seconds": time.time() - t, "prompt_tokens": len(r.prompt_token_ids),
                 "cached_tokens": r.num_cached_tokens})
res = {"prefix_caching": args.prefix_caching, "model": args.model, "rows": rows, "counters": counters()}
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps({"pc": args.prefix_caching, "first_s": round(rows[0]["seconds"], 3),
                  "rest_mean_s": round(sum(r["seconds"] for r in rows[1:]) / (len(rows) - 1), 3),
                  "cached_tokens": [r["cached_tokens"] for r in rows], "counters": res["counters"]}))
