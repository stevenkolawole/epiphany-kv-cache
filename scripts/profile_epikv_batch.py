"""Profile EpiKV's decode step and compaction at one batch size with torch.profiler; save op counts and times.

Builds the steady-state cache of baseline_costs.py at budget K (replicated over the batch, policy primed), runs
`--warm` normal steps unprofiled, then profiles `--steps` normal steps and, separately, the compaction step.
Writes one JSON with, per phase: wall ms, the top ops by self CPU time and by self CUDA time (name, calls,
self CPU ms, self CUDA ms), and the count of host-device syncs (aten::item / aten::_local_scalar_dense /
cudaStreamSynchronize / cudaMemcpyAsync calls). One process, one exclusive card.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_costs as C  # noqa: E402

SYNC_OPS = ("aten::item", "aten::_local_scalar_dense", "cudaStreamSynchronize", "cudaMemcpyAsync",
            "cudaDeviceSynchronize")


def summarize(prof, wall_ms, top=25):
    ev = prof.key_averages()

    def row(e):
        return {"name": e.key, "calls": e.count, "self_cpu_ms": round(e.self_cpu_time_total / 1e3, 3),
                "self_cuda_ms": round(e.self_device_time_total / 1e3, 3)}
    rows = [row(e) for e in ev]
    return {"wall_ms": wall_ms,
            "syncs": {n: next((r["calls"] for r in rows if r["name"] == n), 0) for n in SYNC_OPS},
            "kernel_launches": next((r["calls"] for r in rows if r["name"] == "cudaLaunchKernel"), 0),
            "top_cpu": sorted(rows, key=lambda r: -r["self_cpu_ms"])[:top],
            "top_cuda": sorted(rows, key=lambda r: -r["self_cuda_ms"])[:top]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="epikv", choices=["epikv", "epikv_batched"])
    ap.add_argument("--model", default=C.LLAMA)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    ns = argparse.Namespace(method=a.method, model=a.model, tokenizer=None, band_a_layer=None, band_b_layer=None,
                            restart_windows=False)
    model, tok, _ = C.load_model(ns)
    C._MODEL[0] = model
    ids = C.text_ids(tok, a.K + 2 * C.INTERVAL, next(model.parameters()).device)
    snap, hs = C.base_snapshot(ns, model, ids, a.K)
    cache, pol = C.setup_state(ns, model, ids, a.K, a.batch, 0, hs, snap, a.K)
    out = {"meta": {"method": a.method, "K": a.K, "batch": a.batch, "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__}}
    pos = a.K
    C.decode(ns, model, cache, ids, pos, a.warm, a.batch, pol)
    pos += a.warm
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=acts) as prof:
        t0 = time.perf_counter()
        recs = C.decode(ns, model, cache, ids, pos, a.steps, a.batch, pol)
        wall = (time.perf_counter() - t0) * 1e3
    assert not any(C.is_compaction(r) for r in recs)
    out["normal_steps"] = {"n": a.steps, "fwd_ms": [r["fwd_ms"] for r in recs], "pol_ms": [r["pol_ms"] for r in recs],
                           **summarize(prof, wall)}
    pos += a.steps
    left = C.INTERVAL - 1 - a.warm - a.steps
    C.decode(ns, model, cache, ids, pos, left, a.batch, pol)
    pos += left
    with profile(activities=acts) as prof:
        t0 = time.perf_counter()
        recs = C.decode(ns, model, cache, ids, pos, 1, a.batch, pol)
        wall = (time.perf_counter() - t0) * 1e3
    assert C.is_compaction(recs[0]), recs[0]
    out["compaction_step"] = {"fwd_ms": recs[0]["fwd_ms"], "pol_ms": recs[0]["pol_ms"], **summarize(prof, wall)}
    Path(a.output).write_text(json.dumps(out, indent=1))
    for k in ("normal_steps", "compaction_step"):
        o = out[k]
        print(k, "wall", round(o["wall_ms"], 1), "syncs", o["syncs"], "launches", o["kernel_launches"])
        for r in o["top_cpu"][:8]:
            print("  cpu", r)
    print("DONE")


if __name__ == "__main__":
    main()
