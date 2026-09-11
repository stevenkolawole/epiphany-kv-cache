#!/usr/bin/env python3
"""Pool sharded benchmark.py result files and print accuracy per (tag, K).

    python scripts/summarize_ablation.py results/tau_logical results/bandabl_logical

A file's tag is its name up to the first "_math500" (or the whole stem);
shards of one tag (…_s0, _s25, …) are pooled by problem index. Prints
accuracy, n, mean generated tokens and mean wall time per budget, and the
per-problem correctness vectors so that pairs of tags can be compared on
exactly the same problems (McNemar-style counts).
"""
import glob
import json
import os
import sys
from collections import defaultdict


def load_dir(d):
    pooled = defaultdict(lambda: defaultdict(dict))  # tag -> K -> problem -> record
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        stem = os.path.basename(f)[:-5]
        tag = stem.split("_math500")[0]
        meta_start = json.load(open(f))
        res = meta_start["results"]
        start = int(meta_start["meta"].get("start_idx", 0)) if "start_idx" in meta_start["meta"] else None
        if start is None:  # infer from the shard suffix
            start = int(stem.rsplit("_s", 1)[1]) if "_s" in stem and stem.rsplit("_s", 1)[1].isdigit() else 0
        for method, per_k in res.items():
            for K, rec in per_k.items():
                for i, p in enumerate(rec["per_problem"]):
                    pooled[tag][int(K)][start + i] = p
    return pooled


def main():
    for d in sys.argv[1:]:
        print(f"== {d}")
        pooled = load_dir(d)
        for tag in sorted(pooled):
            for K in sorted(pooled[tag]):
                ps = pooled[tag][K]
                n = len(ps)
                acc = sum(bool(p["correct"]) for p in ps.values()) / n
                toks = sum(p.get("n_tokens_generated", 0) for p in ps.values()) / n
                wall = sum(p.get("wall_time_s", 0) for p in ps.values()) / n
                errs = sum(1 for p in ps.values() if p.get("error"))
                idx = sorted(ps)
                print(f"{tag:12s} K={K:<5d} acc={acc*100:5.1f}%  n={n:3d} [{idx[0]}..{idx[-1]}]  "
                      f"tokens={toks:6.0f}  wall={wall:6.1f}s  errors={errs}")
        # pairwise on shared problems, per K, against every other tag
        tags = sorted(pooled)
        for K in sorted({K for t in tags for K in pooled[t]}):
            have = [t for t in tags if K in pooled[t]]
            for a in have:
                for b in have:
                    if a >= b:
                        continue
                    shared = sorted(set(pooled[a][K]) & set(pooled[b][K]))
                    if len(shared) < 10:
                        continue
                    ca = [bool(pooled[a][K][i]["correct"]) for i in shared]
                    cb = [bool(pooled[b][K][i]["correct"]) for i in shared]
                    a_only = sum(x and not y for x, y in zip(ca, cb))
                    b_only = sum(y and not x for x, y in zip(ca, cb))
                    print(f"   K={K:<5d} {a:12s} vs {b:12s} on {len(shared):3d} shared: "
                          f"{sum(ca)/len(shared)*100:5.1f} vs {sum(cb)/len(shared)*100:5.1f}  "
                          f"(only-{a}: {a_only}, only-{b}: {b_only})")


if __name__ == "__main__":
    main()
