#!/usr/bin/env python3
"""Paired comparison of the vLLM port against the HuggingFace harness on the
same MATH-500 problems: EpiKV-Seg at tau=128 with logical positions in both,
and the two no-eviction runs. Port shards: results/vllm_port/K<K>_s<start>_n<n>.json
(per_problem list); HF cells: results/llama_logical/kv_seg_hs_tau128_K<K>.json
and none_K1024.json.

    python3 scripts/port_vs_hf.py
"""
import glob
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
CAP = 8192


def port(K):
    """Shards K<K>_s<start>_n<n>.json plus the validation runs on problems
    0-19 (validate_K1024_n20.json, validate_none_n20.json)."""
    out = {}
    files = sorted(glob.glob(str(R / "vllm_port" / f"K{K}_s*_n*.json")))
    extra = {1024: "validate_K1024_n20.json", 1000000: "validate_none_n20.json"}.get(K)
    if extra and (R / "vllm_port" / extra).exists():
        files.append(str(R / "vllm_port" / extra))
    for f in files:
        m = re.search(r"_s(\d+)_n(\d+)\.json$", f)
        start = int(m.group(1)) if m else 0
        j = json.load(open(f))
        for i, p in enumerate(j["per_problem"]):
            idx = p.get("idx", p.get("problem_idx", start + i))
            out[idx] = p
    return out


def hf(name, method, K):
    j = json.load(open(R / "llama_logical" / f"{name}.json"))
    return {i: p for i, p in enumerate(j["results"][method][str(K)]["per_problem"])}


def correct(p):
    return bool(p.get("correct"))


def ntok(p):
    return p.get("n_tokens_generated", p.get("n_tokens", p.get("tokens", 0)))


def compare(label, a, b):
    shared = sorted(set(a) & set(b))
    w = sum(1 for i in shared if correct(a[i]) and not correct(b[i]))
    l = sum(1 for i in shared if correct(b[i]) and not correct(a[i]))
    acc_a = 100 * sum(correct(a[i]) for i in shared) / len(shared)
    acc_b = 100 * sum(correct(b[i]) for i in shared) / len(shared)
    cap_a = sum(1 for i in shared if ntok(a[i]) >= CAP)
    cap_b = sum(1 for i in shared if ntok(b[i]) >= CAP)
    same_len = sum(1 for i in shared if ntok(a[i]) == ntok(b[i]))
    print(f"{label}: n={len(shared)} port {acc_a:.0f} vs HF {acc_b:.0f}; port wins {w} / losses {l}; "
          f"cap-hits {cap_a} vs {cap_b}; identical lengths {same_len}")
    return shared


if __name__ == "__main__":
    p0 = port(1024)
    print("port K=1024 problems:", len(p0), "keys:", list(next(iter(p0.values())).keys()))
    compare("Seg tau=128 K=1024", p0, hf("kv_seg_hs_tau128_K1024", "kv_seg_hs", 1024))
    compare("Seg tau=128 K=2048", port(2048), hf("kv_seg_hs_tau128_K2048", "kv_seg_hs", 2048))
    compare("no eviction", port(1000000), hf("none_K1024", "none", 1024))
