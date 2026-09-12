#!/usr/bin/env python3
"""Compare every q.K per-query-relevance cell in results/qk_p4d against the
shipped EpiKV-Seg (tau=128) and the every-step form on the same problems.

    python3 scripts/compare_qk.py
"""
import glob
import json
import os
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def load(f):
    j = json.load(open(f))
    m = next(iter(j["results"]))
    K = next(iter(j["results"][m]))
    return K, j["results"][m][K]["per_problem"]


for f in sorted(glob.glob(str(R / "qk_p4d" / "kv_seg_hs_tau128_qk_*.json"))):
    K, pp = load(f)
    model = "qwen" if "qwen" in f else "llama"
    ref = load(R / f"{model}_logical" / f"kv_seg_hs_tau128_K{K}.json")[1]
    ref1 = load(R / f"{model}_logical" / f"kv_seg_hs_K{K}.json")[1]
    acc = sum(p["correct"] for p in pp)
    cap = sum(1 for p in pp if p["n_tokens_generated"] >= 8192)
    w = sum(1 for a, b in zip(pp, ref) if a["correct"] and not b["correct"])
    l = sum(1 for a, b in zip(pp, ref) if b["correct"] and not a["correct"])
    print(f"{os.path.basename(f)[:-5]:44s} acc={acc:3d} cap={cap:2d} | tau128 {sum(p['correct'] for p in ref)} "
          f"(wins {w} / losses {l}) | tau1 {sum(p['correct'] for p in ref1)}")
