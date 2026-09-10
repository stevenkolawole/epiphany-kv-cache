#!/usr/bin/env python3
"""Pool the AIME grid (aime2024/2025/2026 x shards) into one n=90 table.

Usage: pool_aime.py results/aime_pool [--methods m1 m2 ...]

Reads every <method>_<dataset>_k<K>_s<shard>.json written by benchmark.py,
concatenates per-problem records per (method, K), and prints accuracy,
cap-hit rate and a Wilson 95% interval, plus paired exact McNemar of every
method against the first one listed (default kv_seg_hs) at each K.
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b (A only), c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--ref", default="kv_seg_hs")
    ap.add_argument("--cap", type=int, default=None,
                    help="max_new_tokens; default read from file meta")
    args = ap.parse_args()

    pat = re.compile(r"^(?P<m>.+?)_(?P<ds>aime\d{4})_k(?P<k>\d+)_s(?P<s>\d+)\.json$")
    rows = defaultdict(dict)   # (method, K) -> {(ds, idx): (correct, gen)}
    caps = {}
    for f in sorted(Path(args.dir).glob("*.json")):
        m = pat.match(f.name)
        if not m:
            continue
        d = json.load(open(f))
        cap = args.cap or d.get("meta", {}).get("max_new_tokens", 8192)
        meth, ds, K, s = m["m"], m["ds"], m["k"], int(m["s"])
        res = d["results"].get(meth) or next(iter(d["results"].values()))
        cell = res.get(K) or res.get(int(K)) or next(iter(res.values()))
        for j, r in enumerate(cell["per_problem"]):
            key = (ds, s + j)
            rows[(meth, K)][key] = (bool(r["correct"]), int(r["n_tokens_generated"]) >= cap)
        caps[(meth, K)] = cap

    methods = args.methods or sorted({m for m, _ in rows})
    Ks = sorted({k for _, k in rows}, key=int)
    print(f"pooled AIME 2024/2025/2026, cap={sorted(set(caps.values()))}")
    print(f"{'method':22s} " + "  ".join(f"{'K=' + k:>26s}" for k in Ks))
    for meth in methods:
        line = f"{meth:22s} "
        for K in Ks:
            r = rows.get((meth, K), {})
            n = len(r)
            if n == 0:
                line += f"{'-':>26s}  "
                continue
            acc = sum(c for c, _ in r.values())
            cap = sum(h for _, h in r.values())
            lo, hi = wilson(acc, n)
            line += f"{100 * acc / n:5.1f} [{lo:4.1f},{hi:4.1f}] cap {100 * cap / n:4.1f} n={n:2d}  "
        print(line)
    print()
    print(f"paired vs {args.ref} (wins/losses of {args.ref}, exact McNemar):")
    for K in Ks:
        ref = rows.get((args.ref, K), {})
        if not ref:
            continue
        for meth in methods:
            if meth == args.ref:
                continue
            other = rows.get((meth, K), {})
            keys = sorted(set(ref) & set(other))
            if not keys:
                continue
            w = sum(ref[k][0] and not other[k][0] for k in keys)
            l = sum(other[k][0] and not ref[k][0] for k in keys)
            print(f"  K={K:>5s} {meth:22s} +{w}/-{l} on n={len(keys)}  p={mcnemar_exact(w, l):.3f}")


if __name__ == "__main__":
    main()
