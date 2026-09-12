#!/usr/bin/env python3
"""For every finished tau=128 (and Band-A) Seg cell in a results dir, print
its accuracy and cap-hit rate next to the every-step Seg cell and the
no-eviction cell on the same problems (same file naming as the grids).

    python3 compare_tau_cells.py ~/results_qwen_aime_logical
    python3 compare_tau_cells.py ~/results_llama_logical
"""
import glob
import json
import os
import sys

d = sys.argv[1]


def load(path):
    j = json.load(open(path))
    out = {}
    for meth, per in j["results"].items():
        for K, r in per.items():
            pp = r["per_problem"]
            cap = j["meta"].get("max_new_tokens", 0)
            hits = sum(1 for p in pp if p.get("n_tokens_generated", 0) >= cap) if cap else None
            out[int(K)] = (r["accuracy"] * 100, r["n_total"], hits,
                           [bool(p["correct"]) for p in pp],
                           sum(p.get("n_tokens_generated", 0) for p in pp) / max(1, len(pp)))
    return out


def find(tag):
    p = os.path.join(d, tag + ".json")
    return load(p) if os.path.exists(p) else None


for f in sorted(glob.glob(os.path.join(d, "kv_seg_hs_tau128*.json"))):
    tag = os.path.basename(f)[:-5]
    import re
    # strip the variant suffix (tau128, tau128_aonly, tau128_qsim_<x>) to find the every-step Seg partner
    rest = re.sub(r"kv_seg_hs_tau128(?:_aonly|_qsim_[a-z0-9]+)?(?=_K|_aime)", "kv_seg_hs", tag)
    base = find(rest)
    none_tag = rest.replace("kv_seg_hs", "none")
    if "_K" in none_tag:                                         # MATH grids: none run once at K=1024
        none_tag = "none_K1024"
    else:
        none_tag = none_tag.replace("_k4096_", "_k8192_")        # AIME grids: none at 8192 only
    none = find(none_tag)
    new = load(f)
    for K, (acc, n, hits, corr, toks) in new.items():
        line = f"{tag:45s} K={K:<5d} acc={acc:5.1f}  n={n}  cap-hits={hits}  tokens={toks:6.0f}"
        if base and K in base:
            b = base[K]
            wins = sum(a and not c for a, c in zip(corr, b[3])); losses = sum(c and not a for a, c in zip(corr, b[3]))
            line += f" | tau=1: {b[0]:5.1f} cap={b[2]} (wins {wins} / losses {losses})"
        if none:
            nk = list(none.values())[0]
            line += f" | none: {nk[0]:5.1f} cap={nk[2]}"
        print(line)
