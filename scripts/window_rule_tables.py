#!/usr/bin/env python3
"""Merge the window-rule grid's 100-problem pieces into cells and tabulate them against EpiKV.

Usage: python3 scripts/window_rule_tables.py <dir with cells/> [--n 500] [--cap 16384]

Reads <dir>/cells/<model>_<method>_K<K>_s<start>.json. A cell is complete when its pieces cover problems 0..n-1, every
piece is final (no "partial" flag), and no problem has an error or 0 generated tokens; incomplete cells are listed and
left out. For each complete cell: accuracy, loops (traces that reached the cap of --cap generated tokens), and against
EpiKV (kv_seg_hs_fill) at the same model and K, the paired wins (EpiKV right and the other wrong, and the reverse) with
the exact two-sided McNemar p-value (binomial test on the discordant pairs). Writes <dir>/TABLES.md and prints it.
"""
import argparse
import json
import re
from collections import defaultdict
from math import comb
from pathlib import Path

METHODS = ["kv_seg_hs_fill", "raas_epikv_window", "r_kv_epikv_window", "kvpress_snapkv_decoding"]
NAMES = {"kv_seg_hs_fill": "EpiKV", "raas_epikv_window": "RaaS rule on EpiKV window",
         "r_kv_epikv_window": "R-KV rule on EpiKV window", "kvpress_snapkv_decoding": "kvpress SnapKV DecodingPress"}
TAG = re.compile(r"^(llama|qwen)_(.+)_K(\d+)_s(\d+)$")


def mcnemar_exact(b: int, c: int) -> float:
    """Return the two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_cells(root: Path, n: int):
    pieces = defaultdict(dict)
    for f in sorted((root / "cells").glob("*.json")):
        if f.name.endswith(".tmp.json"):
            continue
        m = TAG.match(f.stem)
        if not m:
            continue
        model, method, K, s = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        blocks = json.load(open(f))["results"].get(method, {})
        pieces[(model, method, K)][s] = blocks.get(str(K), {})
    cells, bad = {}, []
    for key, by_start in sorted(pieces.items()):
        rows, why = [], None
        for s in sorted(by_start):
            b = by_start[s]
            pp = b.get("per_problem", [])
            if b.get("partial") or b.get("n_total") != len(pp):
                why = f"piece s{s} partial ({len(pp)}/{b.get('n_total')})"
                break
            if s != len(rows):
                why = f"gap before s{s}"
                break
            rows += pp
        if why is None and len(rows) != n:
            why = f"{len(rows)}/{n} problems"
        if why is None:
            errs = sum(1 for r in rows if r.get("error") or not r.get("n_tokens_generated"))
            if errs:
                why = f"{errs} problems errored or generated 0 tokens"
        if why:
            bad.append((key, why))
        else:
            cells[key] = rows
    return cells, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--cap", type=int, default=16384)
    a = ap.parse_args()
    cells, bad = load_cells(a.root, a.n)
    out = ["| model | K | method | accuracy | loops (hit cap) | EpiKV only | other only | McNemar p |",
           "|---|---|---|---|---|---|---|---|"]
    for model in ("llama", "qwen"):
        for K in (512, 1024, 2048, 4096):
            ref = cells.get((model, "kv_seg_hs_fill", K))
            for meth in METHODS:
                rows = cells.get((model, meth, K))
                if rows is None:
                    continue
                acc = sum(bool(r["correct"]) for r in rows) / len(rows)
                loops = sum(r["n_tokens_generated"] >= a.cap for r in rows)
                pair = ["-", "-", "-"]
                if ref is not None and meth != "kv_seg_hs_fill":
                    assert all(x["problem"] == y["problem"] for x, y in zip(ref, rows)), (model, meth, K)
                    b = sum(bool(x["correct"]) and not y["correct"] for x, y in zip(ref, rows))
                    c = sum(bool(y["correct"]) and not x["correct"] for x, y in zip(ref, rows))
                    pair = [str(b), str(c), f"{mcnemar_exact(b, c):.3g}"]
                out.append(f"| {model} | {K} | {NAMES[meth]} | {100 * acc:.1f} | {loops} | " + " | ".join(pair) + " |")
    if bad:
        out += ["", "Incomplete cells (left out):"] + [f"- {k[0]} {k[1]} K={k[2]}: {why}" for k, why in bad]
    text = "\n".join(out) + "\n"
    (a.root / "TABLES.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
