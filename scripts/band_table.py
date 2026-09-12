#!/usr/bin/env python3
"""Band ablation table (EpiKV paper, appendix), two blocks.

Block 1, the score's anatomy on the plain per-token scorer
(hs_variance_detrend, no segment tiering), Babel L40S / transformers 5.2.0,
Llama-8B MATH-500 problems 0-99, every-step eviction: A-B (deployed layers
10/21), A only, B only, alternative layer pairs 7/12 and 5/15.

Block 2, EpiKV-Seg proper, A-B against A only: Babel at tau=1 (tau1 vs
aonly_tau1), and the Lambda grids (transformers 4.57.1) at tau=128
(kv_seg_hs_tau128 vs kv_seg_hs_tau128_aonly) for both models.

    python3 scripts/band_table.py [--latex]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_ablation import load_dir  # noqa: E402

R = Path(__file__).resolve().parent.parent / "results"


def acc_wl(ps, ref):
    n = len(ps)
    acc = 100.0 * sum(bool(p["correct"]) for p in ps.values()) / n
    if ref is None:
        return acc, n, None, None
    shared = sorted(set(ps) & set(ref))
    w = sum(1 for i in shared if ps[i]["correct"] and not ref[i]["correct"])
    l = sum(1 for i in shared if ref[i]["correct"] and not ps[i]["correct"])
    return acc, n, w, l


def load_lambda(d, tag):
    """One Lambda grid file (single shard, n=100) -> {K: {i: record}}."""
    f = R / d / f"{tag}.json"
    if not f.exists():
        return {}
    j = json.load(open(f))
    out = {}
    for meth, per in j["results"].items():
        for K, r in per.items():
            out[int(K)] = {i: p for i, p in enumerate(r["per_problem"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    b = load_dir(str(R / "bandabl_logical"))
    t = load_dir(str(R / "tau_logical"))
    rows = []
    # Block 1: plain scorer, ref = diff
    for tag, label in (("diff", "A$-$B (10/21)"), ("a_only", "A only (10)"), ("b_only", "B only (21)"),
                       ("alt_7_12", "A$-$B, layers 7/12"), ("alt_5_15", "A$-$B, layers 5/15")):
        cells = []
        for K in (1024, 2048):
            acc, n, w, l = acc_wl(b[tag][K], None if tag == "diff" else b["diff"][K])
            cells.append((acc, w, l))
        rows.append(("plain scorer, Babel, $\\tau{=}1$", label, cells))
    # Block 2: Seg on Babel tau=1
    for tag, label in (("tau1", "A$-$B"), ("aonly_tau1", "A only")):
        cells = []
        for K in (1024, 2048):
            acc, n, w, l = acc_wl(t[tag][K], None if tag == "tau1" else t["tau1"][K])
            cells.append((acc, w, l))
        rows.append(("EpiKV-Seg, Babel, $\\tau{=}1$", label, cells))
    # Block 3: Seg on Lambda tau=128, both models, K=512..4096
    for d, model in (("llama_logical", "Llama-8B"), ("qwen_logical", "Qwen-7B")):
        base = {}
        for K in (512, 1024, 2048, 4096):
            base.update(load_lambda(d, f"kv_seg_hs_tau128_K{K}"))
        aonly = {}
        for K in (512, 1024, 2048, 4096):
            aonly.update(load_lambda(d, f"kv_seg_hs_tau128_aonly_K{K}"))
        for tag, src, label in (("base", base, "A$-$B"), ("aonly", aonly, "A only")):
            cells = []
            for K in (512, 1024, 2048, 4096):
                if K not in src:
                    cells.append(None)
                    continue
                acc, n, w, l = acc_wl(src[K], None if tag == "base" else base.get(K))
                cells.append((acc, w, l))
            rows.append(("EpiKV-Seg, Lambda, $\\tau{=}128$, " + model, label, cells))
    for block, label, cells in rows:
        def fmt(c):
            if c is None:
                return "--"
            acc, w, l = c
            return f"{acc:.0f}" if w is None else f"{acc:.0f} ({w}/{l})"
        if a.latex:
            print(f"{block} & {label} & " + " & ".join(fmt(c) for c in cells) + " \\\\")
        else:
            print(f"| {block} | {label} | " + " | ".join(fmt(c) for c in cells) + " |")


if __name__ == "__main__":
    main()
