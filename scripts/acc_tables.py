#!/usr/bin/env python3
"""Accuracy tables for the EpiKV paper from the logical-position grids
(A100-80GB, transformers 4.57.1, greedy, MATH-500 problems 0-99 with an
8,192-token cap; AIME 2024/2025/2026 pooled, n=90, 16,384-token cap).

  --math     Table 2 rows (main methods), both models, K = 512..4096
  --family   family table rows (hidden-state and KV-vector variants)
  --aime     pooled AIME rows, both models, K = 4096 / 8192
  --latex    LaTeX rows instead of markdown
  --pairs    also print paired wins/losses and exact McNemar p against
             EpiKV-Seg (tau=128) per column

EpiKV-Seg means the shipped tau=128 form (files kv_seg_hs_tau128_K*,
*_tau128_* AIME cells); "EpiKV-Seg (tau=1)" is the every-step form.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool_aime import mcnemar_exact  # noqa: E402

R = Path(__file__).resolve().parent.parent / "results"
MATH_DIRS = {"Llama-8B": "llama_logical", "Qwen-7B": "qwen_logical"}
AIME_DIRS = {"Llama-8B": "aime_pool_logical", "Qwen-7B": "qwen_aime_pool_logical"}
KS_MATH = [512, 1024, 2048, 4096]
KS_AIME = [4096, 8192]

# (file tag, label, FA2)
MAIN = [("none", "none$^{\\star}$", True),
        ("kv_seg_hs_tau128", "\\methodseg~(ours)", True),
        ("lag_kv", "Lag-KV", True),
        ("r_kv", "R-KV", False),
        ("thinkv_faithful", "ThinKV", False),
        ("raas", "RaaS", False),
        ("longflow", "LongFlow", False),
        ("h2o", "H2O", False)]
FAMILY = [("kv_seg_hs_tau128", "\\methodseg~($\\tau{=}128$)", True),
          ("kv_seg_hs", "\\methodseg~($\\tau{=}1$)", True),
          ("hs_variance_detrend", "\\methodflat", True),
          ("hs_variance", "HS-variance", True),
          ("band_adaptive_hs", "Band-adaptive", True),
          ("kv_val", "KV-val", True),
          ("kv_key", "KV-key", True),
          ("lag_kv_key", "Lag-KV-key", True),
          ("lag_kv", "Lag-KV", True)]
AIME_TAG = {"kv_seg_hs_tau128": "kv_seg_hs_tau128", "kv_seg_hs": "kv_seg_hs"}


def math_cells(d, tag):
    """{K: [correct bools by problem]} for one method in a MATH grid dir."""
    out = {}
    if tag == "none":
        f = R / d / "none_K1024.json"
        if f.exists():
            pp = json.load(open(f))["results"]["none"]["1024"]["per_problem"]
            for K in KS_MATH:
                out[K] = [bool(p["correct"]) for p in pp]
        return out
    for K in KS_MATH:
        f = R / d / f"{tag}_K{K}.json"
        if not f.exists():
            continue
        j = json.load(open(f))
        meth = next(iter(j["results"]))
        out[K] = [bool(p["correct"]) for p in j["results"][meth][str(K)]["per_problem"]]
    return out


def aime_cells(d, tag):
    """{K: {problem key: correct}} pooled over the three years."""
    out = {}
    for f in glob.glob(str(R / d / f"{tag}_aime*_k*_s0.json")):
        m = re.search(r"_aime(\d{4})_k(\d+)_s0\.json$", f)
        if not m:
            continue
        year, K = m.group(1), int(m.group(2))
        j = json.load(open(f))
        meth = next(iter(j["results"]))
        pp = j["results"][meth][str(K)]["per_problem"]
        out.setdefault(K, {}).update({(year, i): bool(p["correct"]) for i, p in enumerate(pp)})
    if tag == "none" and 8192 in out:
        out[4096] = out[8192]
    return out


def pair(a, b):
    """wins/losses of a over b on shared problems, and exact McNemar p."""
    if isinstance(a, dict):
        keys = sorted(set(a) & set(b))
        w = sum(1 for k in keys if a[k] and not b[k]); l = sum(1 for k in keys if b[k] and not a[k])
    else:
        w = sum(1 for x, y in zip(a, b) if x and not y); l = sum(1 for x, y in zip(a, b) if y and not x)
    return w, l, mcnemar_exact(w, l)


def acc(v):
    vals = list(v.values()) if isinstance(v, dict) else v
    return 100.0 * sum(vals) / len(vals)


def table(rows, loader, dirs, Ks, latex, pairs):
    data = {model: {tag: loader(d, tag) for tag, _, _ in rows} for model, d in dirs.items()}
    # best per (model, K) among eviction rows
    best = {}
    for model in dirs:
        for K in Ks:
            vals = [acc(data[model][tag][K]) for tag, _, _ in rows if tag != "none" and K in data[model][tag]]
            best[(model, K)] = max(vals) if vals else None
    for tag, label, fa2 in rows:
        cells = []
        for model in dirs:
            for K in Ks:
                v = data[model][tag].get(K)
                if v is None:
                    cells.append("--"); continue
                s = f"{acc(v):.1f}"
                if tag != "none" and best[(model, K)] is not None and abs(acc(v) - best[(model, K)]) < 1e-9:
                    s = f"\\textbf{{{s}}}" if latex else f"**{s}**"
                if pairs and tag != "kv_seg_hs_tau128":
                    ref = data[model]["kv_seg_hs_tau128"].get(K)
                    if ref is not None:
                        w, l, p = pair(data[model]["kv_seg_hs_tau128"][K], v)
                        s += f" ({w}/{l}, p={p:.2f})"
                cells.append(s)
        mark = ("\\cmark" if fa2 else "\\xmark") if latex else ("y" if fa2 else "n")
        if latex:
            print(f"{label} & {mark} & " + " & ".join(cells) + " \\\\")
        else:
            print(f"| {label} | {mark} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math", action="store_true")
    ap.add_argument("--family", action="store_true")
    ap.add_argument("--aime", action="store_true")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--pairs", action="store_true")
    a = ap.parse_args()
    if a.math:
        print("% MATH-500, main rows; columns: Llama 512 1024 2048 4096 | Qwen 512 1024 2048 4096")
        table(MAIN, math_cells, MATH_DIRS, KS_MATH, a.latex, a.pairs)
    if a.family:
        print("% MATH-500, family rows")
        rows = [r for r in FAMILY if not (a.pairs and r[0] == "kv_seg_hs_tau128")] if False else FAMILY
        table(rows, math_cells, MATH_DIRS, KS_MATH, a.latex, a.pairs)
    if a.aime:
        print("% pooled AIME; columns: Llama 4096 8192 | Qwen 4096 8192")
        rows = [(t, l, f) for t, l, f in MAIN] + [r for r in FAMILY if r[0] not in ("kv_seg_hs_tau128", "lag_kv")]
        rows = [r for r in rows if r[0] not in ("r_kv", "longflow")] + [("r_kv", "R-KV", False), ("longflow", "LongFlow", False)]
        table(rows, aime_cells, AIME_DIRS, KS_AIME, a.latex, a.pairs)


if __name__ == "__main__":
    main()
