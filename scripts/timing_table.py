#!/usr/bin/env python3
"""Equal-load long-context timing table (EpiKV paper): one process per
A100-80GB card, AIME-2024 (30 problems), K=8192, 16,384-token cap, greedy,
transformers 4.57.1 / torch 2.7 / flash-attn 2.8.3. Decode rate = generated
tokens / wall time summed over the 30 problems.

    python3 scripts/timing_table.py [--latex]
"""
import argparse
import glob
import json
import os
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
CAP = 16384
ROWS = [("none", "No eviction", "FA2"),
        ("kv_seg_hs_refresh_tau128", "EpiKV-Seg ($\\tau{=}128$)", "FA2"),
        ("kv_seg_hs_refresh_tau1", "EpiKV-Seg ($\\tau{=}1$)", "FA2"),
        ("hs_variance", "HS-variance", "FA2"),
        ("kv_key", "KV-key", "FA2"),
        ("raas", "RaaS", "eager"),
        ("h2o", "H2O", "eager"),
        ("r_kv", "R-KV", "eager"),
        ("longflow", "LongFlow", "eager"),
        ("thinkv_faithful", "ThinKV", "eager")]


def cell(d, tag):
    f = R / d / f"{tag}_aime2024_k8192.json"
    if not f.exists():
        return None
    j = json.load(open(f))
    meth = next(iter(j["results"]))
    r = j["results"][meth]["8192"]
    pp = r["per_problem"]
    toks = sum(p["n_tokens_generated"] for p in pp)
    wall = sum(p["wall_time_s"] for p in pp)
    return dict(acc=100 * r["accuracy"], cap=sum(1 for p in pp if p["n_tokens_generated"] >= CAP),
                tps=toks / wall, spp=wall / len(pp), tok=toks / len(pp), gb=r["mean_peak_gpu_mb"] / 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    if not a.latex:
        print("| method | kernel | Llama acc | cap | tok/s | s/prob | GB | Qwen acc | cap | tok/s | s/prob | GB |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, label, kern in ROWS:
        parts = []
        for d in ("timing_llama", "timing_qwen"):
            c = cell(d, tag)
            if c is None:
                parts += ["--"] * 5
            else:
                parts += [f"{c['acc']:.1f}", f"{c['cap']}", f"{c['tps']:.1f}", f"{c['spp']:.0f}", f"{c['gb']:.1f}"]
        if a.latex:
            print(f"{label} & {kern} & " + " & ".join(parts) + " \\\\")
        else:
            print(f"| {label} | {kern} | " + " | ".join(parts) + " |")


if __name__ == "__main__":
    main()
