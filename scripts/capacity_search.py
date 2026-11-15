"""Find the largest decodable batch with one fresh process per probe (doubling, then bisection).

For stacks where an OOM inside a forward leaves memory held for the rest of the process: transformers 4.57.1
with output_hidden_states=True did this in baseline_costs.py capacity runs (EpiKV on Qwen3-8B: batch-1 peak
17.0 GiB after an OOM against 15.6 GiB fresh). Each probe runs `baseline_costs.py --mode capacity --probe_only`;
the probes and the result are written to --output as JSON.
Usage: capacity_search.py --python <venv python> --K 1024 --output out.json -- <baseline_costs.py args>
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def probe(a, extra, b):
    tmp = Path(a.output).with_suffix(f".probe{b}.json")
    cmd = [a.python, str(Path(__file__).parent / "baseline_costs.py"), "--mode", "capacity", "--probe_only",
           "--Ks", str(a.K), "--batches", str(b), "--output", str(tmp)] + extra
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cell = json.loads(tmp.read_text())["cells"].get(f"K{a.K}_B{b}") if tmp.exists() else None
    tmp.unlink(missing_ok=True)
    if cell is None:
        sys.exit(f"probe at batch {b} wrote no record")
    return cell["ok"], cell["peak_gb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--max_batch", type=int, default=1024)
    ap.add_argument("--output", required=True)
    a, extra = ap.parse_known_args()
    extra = [x for x in extra if x != "--"]
    tried = {}
    lo, hi, b = 0, None, 1
    while hi is None and b <= a.max_batch:
        ok, pk = probe(a, extra, b)
        tried[b] = pk if ok else "OOM"
        print(a.K, b, tried[b], flush=True)
        lo, hi, b = (b, None, b * 2) if ok else (lo, b, b)
    while hi is not None and hi - lo > 1:
        mid = (lo + hi) // 2
        ok, pk = probe(a, extra, mid)
        tried[mid] = pk if ok else "OOM"
        print(a.K, mid, tried[mid], flush=True)
        lo, hi = (mid, hi) if ok else (lo, mid)
    Path(a.output).write_text(json.dumps({"meta": {"mode": "capacity_fresh", "K": a.K, "args": extra},
                                          "cells": {f"K{a.K}": {"K_or_context": a.K, "max_batch": lo,
                                                                "first_oom": hi, "probes": tried}}}, indent=1))
    print(a.K, "MAX", lo, flush=True)


if __name__ == "__main__":
    main()
