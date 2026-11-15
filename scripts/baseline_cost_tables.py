"""Turn the baseline_costs.py records in one folder into markdown tables (TABLES.md beside them).

Reads every <mode>_<method>[_<model>].json written by baseline_costs.py; skips smoke_* files. Values are medians
over the recorded repeats (steady: the three 128-step windows after the warmup window). Memory in GiB.
"""
import argparse
import json
import statistics as st
from pathlib import Path


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def f(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def oom(c):
    """OOM while building the cache and policy state means the cache itself does not fit, not the method's step."""
    return "OOM (cache does not fit)" if c.get("oom_phase") == "setup" else "OOM"


def load(folder):
    recs = []
    for p in sorted(Path(folder).glob("*.json")):
        if p.name.startswith("smoke_") or p.name == "analytical.json":
            continue
        d = json.loads(p.read_text())
        if "mode" in d.get("meta", {}) and isinstance(d.get("cells"), dict):   # skips profile and check records
            recs.append((p.stem, d))
    return recs


def steady(recs):
    rows = ["| run | K | B | ms/token | spread | normal step ms | compaction ms (step) | policy ms/token | "
            "resident GiB | peak GiB | max transient GiB | cache after compaction |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, d in recs:
        if d["meta"]["mode"] != "steady":
            continue
        for c in d["cells"].values():
            if "oom" in c:
                rows.append(f"| {name} | {c['K']} | {c['batch']} | {oom(c)} | | | | | | | | |")
                continue
            w = c["windows"]
            mt = [x["ms_per_token"] for x in w]
            comp = [x["compaction"][0] for x in w if x.get("compaction")]
            rows.append(
                f"| {name} | {c['K']} | {c['batch']} | {f(med(mt), 2)} | {f(max(mt) - min(mt), 2)} | "
                f"{f(med(x.get('normal_step_ms_median') for x in w), 2)} | {f(med(x['ms'] for x in comp))} | "
                f"{f(med(x['pol_ms_mean'] for x in w), 3)} | {f(w[0]['resident_gb'], 2)} | "
                f"{f(max(x['peak_gb'] for x in w), 2)} | {f(max(x['transient_gb'] for x in w), 3)} | "
                f"{'/'.join(sorted({str(x['to']) for x in comp})) or '-'} |")
    return rows


def compaction(recs):
    rows = ["| run | L | K | B | step ms (fwd + policy) | policy ms | transient GiB | resident GiB | from -> to |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, d in recs:
        if d["meta"]["mode"] != "compaction":
            continue
        for c in d["cells"].values():
            if "oom" in c or "oom_build" in c:
                rows.append(f"| {name} | {c['L']} | {c.get('K', '-')} | {c.get('batch', '-')} | "
                            f"{oom(c) if 'oom' in c else 'OOM building the cache'} | | | | |")
                continue
            r = c["repeats"]
            rows.append(f"| {name} | {c['L']} | {c['K']} | {c['batch']} | {f(med(x['ms'] for x in r))} | "
                        f"{f(med(x['pol_ms'] for x in r))} | {f(max(x['transient_gb'] for x in r), 3)} | "
                        f"{f(r[0]['resident_gb'], 2)} | {r[0]['from']} -> {r[0]['to']} |")
    return rows


def capacity(recs):
    rows = ["| run | K (none: context) | max batch | first OOM | peak GiB at max |", "|---|---|---|---|---|"]
    for name, d in recs:
        if d["meta"]["mode"] == "dms_capacity":     # DMS: prefill of the context is part of each probe
            for c in d["cells"].values():
                pk = c["probes"].get(str(c["max_batch"]), {}).get("peak_gb")
                rows.append(f"| {name} (with its prefill) | context {c['context']} | {c['max_batch']} | "
                            f"{c['first_oom']} | {f(pk, 2)} |")
            continue
        if d["meta"]["mode"] not in ("capacity", "capacity_fresh"):
            continue
        for c in d["cells"].values():
            pk = c["probes"].get(str(c["max_batch"]), c["probes"].get(c["max_batch"]))
            rows.append(f"| {name} | {c['K_or_context']} | {c['max_batch']} | {c['first_oom']} | {f(pk, 2)} |")
    return rows


def prefill(recs):
    rows = ["| run | L | ms | peak GiB | transient GiB | cache after |", "|---|---|---|---|---|---|"]
    for name, d in recs:
        if d["meta"]["mode"] != "prefill":
            continue
        for c in d["cells"].values():
            if "oom" in c:
                rows.append(f"| {name} | {c['L']} | OOM | | | |")
                continue
            r = c["repeats"]
            rows.append(f"| {name} | {c['L']} | {f(med(x['ms'] for x in r))} | {f(max(x['peak_gb'] for x in r), 2)} | "
                        f"{f(max(x['transient_gb'] for x in r), 2)} | {r[0]['cache_len']} |")
    return rows


def dms(recs):
    rows = ["| run | B | context at window end | ms/token | live KV entries (mean) | peak GiB |",
            "|---|---|---|---|---|---|"]
    for name, d in recs:
        if d["meta"]["mode"] != "dms_decode":
            continue
        for c in d["cells"].values():
            if "oom" in c:
                rows.append(f"| {name} | {c['batch']} | OOM | | | |")
                continue
            for w in c["windows"]:
                if w["context_end"] in (P + 1024, P + 2048, P + 4096, P + 8192) or w is c["windows"][-1]:
                    rows.append(f"| {name} | {c['batch']} | {w['context_end']} | {f(w['ms_per_token'], 2)} | "
                                f"{f(w['live_mean'], 0)} | {f(w['peak_gb'], 2)} |")
    return rows


P = 128


def headline(recs):
    """Per method at K in {1024, 4096} and batch {1, 16}: steady ms/token against no eviction at the same K and
    batch (whose cache grows from K to K+512 over the windows), the compaction's extra time over a normal step
    (H2O/RaaS evict every step: their step minus no eviction's step), the window's peak, and the max batch."""
    by = {name: d for name, d in recs}
    rows = ["| method | K | B | ms/token | / none | compaction extra ms | peak GiB | max batch at K |",
            "|---|---|---|---|---|---|---|---|"]
    for m in ("none", "epikv", "epikv_batched", "epikv_noq", "h2o", "raas", "rkv", "snapkv_decoding", "none_kvpressvenv",
              "none_qwen3", "epikv_qwen3"):
        # A _restart run (every window from the cache at K) supersedes one whose windows never reached K.
        d = by.get(f"steady_{m}_restart", by.get(f"steady_{m}"))
        if d is None:
            continue
        base = by.get("steady_none_qwen3" if m.endswith("qwen3") else
                      "steady_none_kvpressvenv" if m == "snapkv_decoding" else "steady_none")
        cap = by.get(f"capacity_{m}")
        fresh = {k: c for name, d2 in recs if name.startswith(f"capacity_fresh_{m}_K") for k, c in d2["cells"].items()}
        if fresh:   # one process per probe supersedes an in-process search (see capacity_search.py)
            cap = {"cells": fresh}
        for K in (1024, 4096):
            for b in (1, 16):
                c = d["cells"].get(f"K{K}_B{b}")
                if c is None:
                    continue
                mc = cap["cells"].get(f"K{K}", {}).get("max_batch") if cap else None
                if "oom" in c:
                    rows.append(f"| {m} | {K} | {b} | {oom(c)} | | | | {f(mc)} |")
                    continue
                w = c["windows"]
                mt = med(x["ms_per_token"] for x in w)
                bw = base["cells"][f"K{K}_B{b}"]["windows"]
                bn = med(x["ms_per_token"] for x in bw)
                comp = [x["compaction"][0]["ms"] - x["normal_step_ms_median"] for x in w
                        if x.get("compaction") and x.get("normal_step_ms_median")]
                if not comp and all(x["n_compactions"] == x["steps"] for x in w):
                    comp = [mt - bn]
                rows.append(f"| {m} | {K} | {b} | {f(mt, 2)} | {f(mt / bn, 3)} | {f(med(comp))} | "
                            f"{f(max(x['peak_gb'] for x in w), 2)} | {f(mc)} |")
    return rows


def throughput(recs):
    """Tokens/s per card at the largest batch that ran steady decode (tput_* runs, --first_fit)."""
    rows = ["| run | K (none: context) | batch | ms/step | spread | tokens/s per card | peak GiB | cache after compaction |",
            "|---|---|---|---|---|---|---|---|"]
    for name, d in recs:
        if not name.startswith("tput_"):
            continue
        cells = [c for c in d["cells"].values() if "oom" not in c]
        tried = sorted((c["batch"] for c in d["cells"].values() if "oom" in c), reverse=True)
        for c in cells:
            w = c["windows"]
            mt = [x["ms_per_token"] for x in w]
            comp = [x["compaction"][0]["to"] for x in w if x.get("compaction")]
            rows.append(f"| {name} | {c['K']} | {c['batch']} | {f(med(mt), 1)} | {f(max(mt) - min(mt), 1)} | "
                        f"{f(c['batch'] * 1000 / med(mt), 0)} | {f(max(x['peak_gb'] for x in w), 2)} | "
                        f"{'/'.join(sorted({str(t) for t in comp})) or '-'} |")
        if tried:
            rows.append(f"| {name} | | OOM at {', '.join(map(str, tried))} | | | | | |")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    a = ap.parse_args()
    recs = load(a.folder)
    parts = ["# Measured tables (generated by scripts/baseline_cost_tables.py; do not edit by hand)", ""]
    for title, fn in (("Headline (steady decode, Llama-8B unless marked)", headline), ("Throughput per card at the largest batch", throughput), ("Steady decode", steady), ("One compaction from length L", compaction),
                      ("Capacity", capacity), ("Prefill", prefill), ("DMS decode", dms)):
        parts += [f"## {title}", ""] + fn(recs) + [""]
    (Path(a.folder) / "TABLES.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
