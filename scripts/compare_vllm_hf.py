#!/usr/bin/env python3
"""Problem-by-problem comparison of the vLLM port against an HF reference.

Two result formats: the validator writes {"per_problem": [...]} at top level;
the harness writes {"results": {method: {K: {"per_problem": [...]}}}}. Rows
are matched by index (same --start_idx, same n) and by problem text as a
check. Agreement is the number the port's correctness rests on.
"""
import argparse
import json
import math


def load_validator(path):
    return json.load(open(path))["per_problem"]


def load_harness(path, method, K):
    return json.load(open(path))["results"][method][str(K)]["per_problem"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--method", default="kv_seg_hs")
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--cap", type=int, default=8192)
    args = ap.parse_args()

    v = load_validator(args.vllm)
    h = load_harness(args.hf, args.method, args.K)
    n = min(len(v), len(h))
    print(f"{'i':>2} {'vllm':>5} {'vtok':>5} {'comp':>4} | {'hf':>5} {'htok':>5}  problem")
    agree = wins = losses = 0
    vcap = hcap = 0
    for i in range(n):
        a, b = v[i], h[i]
        if a["problem"][:40] != b["problem"][:40]:
            print(f"{i:>2} PROBLEM MISMATCH: {a['problem'][:30]!r} vs {b['problem'][:30]!r}")
        va, hb = bool(a["correct"]), bool(b["correct"])
        agree += va == hb
        wins += va and not hb
        losses += hb and not va
        vcap += a["n_tokens_generated"] >= args.cap
        hcap += b["n_tokens_generated"] >= args.cap
        print(f"{i:>2} {str(va):>5} {a['n_tokens_generated']:>5} {len(a.get('compactions', [])):>4} | "
              f"{str(hb):>5} {b['n_tokens_generated']:>5}  {a['problem'][:40]}")
    k = wins + losses
    p = 1.0 if k == 0 else min(1.0, 2 * sum(math.comb(k, j) for j in range(min(wins, losses) + 1)) / 2 ** k)
    print(f"\nn={n} agree={agree} vllm-only-correct={wins} hf-only-correct={losses} "
          f"exact McNemar p={p:.3f}")
    print(f"vllm acc={sum(bool(x['correct']) for x in v[:n])/n:.3f} cap-hits={vcap}   "
          f"hf acc={sum(bool(x['correct']) for x in h[:n])/n:.3f} cap-hits={hcap}")


if __name__ == "__main__":
    main()
