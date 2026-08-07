#!/usr/bin/env python3
"""Regenerate the timing and memory tables from a results directory.

Every timing and memory number currently in the paper was measured before the
decode-loop cache fix, on Babel's mixed L40S/A6000/RTX pool, at n=30. Wall-clock
is not comparable across cards, so those tables mix hardware as well as code
state. The pooled-AIME runs on p4d replace both problems at once: one code
state, eight identical A100-40GB, n=90, and every per-problem record already
carries wall_time_s, n_tokens_generated and peak_gpu_mb.

Emits LaTeX, not numbers to copy. A transcription error (15525 for 15524) got
into a table once; generating the rows removes that failure mode.

Usage:
    python scripts/make_timing_tables.py --results <dir> --budget 8192
    python scripts/make_timing_tables.py --results <dir> --budget 8192 --memory

The results dir is a flat set of benchmark.py outputs (results/aime_pool on
p4d). Shards for the same (method, budget) are pooled.
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

# Paper names, per Table 1, which is the naming authority. Anything not listed
# is an internal variant and is skipped rather than printed with a harness key.
NAME = {
    "hs_variance_detrend": r"\methodflat",
    "kv_seg_hs":           r"\methodseg",
    "hs_variance":         "HS-variance",
    "band_adaptive_hs":    "Band-adaptive",
    "kv_val":              "KV-val",
    "kv_key":              "KV-key",
    "lag_kv_key":          "Lag-KV-key",
    "lag_kv":              "Lag-KV",
    "thinkv_faithful":     "ThinKV",
    "thinKV":              "ThinKV (simplified)",
    "h2o":                 "H2O",
    "raas":                "RaaS",
    "r_kv":                "R-KV",
    "longflow":            "LongFlow",
}
FA2 = {"hs_variance_detrend", "kv_seg_hs", "hs_variance", "band_adaptive_hs",
       "kv_val", "kv_key", "lag_kv_key", "lag_kv", "none"}


def collect(results_dir, budget):
    """Pool every shard by (method, budget). Returns method -> list of records."""
    rows = defaultdict(list)
    unreadable = []
    for f in sorted(Path(results_dir).glob("*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            unreadable.append(f.name)
            continue
        for m, by_k in d.get("results", {}).items():
            entry = by_k.get(str(budget))
            if not entry:
                continue
            for r in entry.get("per_problem", []):
                if isinstance(r.get("wall_time_s"), (int, float)) and r.get("n_tokens_generated"):
                    rows[m].append(r)
    return rows, unreadable


def decomposition(rows, budget, dataset):
    """Method | FA2 | s/1k tok | tokens | s/prob.

    Wall-clock per problem is tokens x per-token cost. Reporting only the
    product lets a method that breaks termination look slow for a reason that
    has nothing to do with its per-token cost, which is what the kernel sets.
    """
    stats = {}
    for m, recs in rows.items():
        if m != "none" and m not in NAME:
            continue
        tok = st.mean(r["n_tokens_generated"] for r in recs)
        sec = st.mean(r["wall_time_s"] for r in recs)
        stats[m] = (sec / tok * 1000, tok, sec, len(recs))

    if not stats:
        return "% no usable records at this budget\n"

    best_rate = min(v[0] for m, v in stats.items() if m != "none")
    best_sec  = min(v[2] for m, v in stats.items() if m != "none")
    best_tok  = min(v[1] for m, v in stats.items() if m != "none")

    def fmt(m, v):
        rate, tok, sec, n = v
        mark = r"\cmark" if m in FA2 else r"\xmark"
        label = "none" if m == "none" else NAME[m]
        b = lambda x, best: (r"\textbf{%s}" % x) if abs(float(x.replace("{,}", "").replace(",", "")) - best) < 1e-6 and m != "none" else x
        return "%s & %s & %s & %s & %s \\\\" % (
            label, mark,
            b("%.1f" % rate, best_rate),
            b("{:,.0f}".format(tok).replace(",", "{,}"), best_tok),
            b("%.0f" % sec, best_sec),
        )

    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\setlength{\tabcolsep}{5pt}",
           r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
           r"Method & FA2 & s/1k tok $\downarrow$ & tokens $\downarrow$ & s/prob $\downarrow$ \\",
           r"\midrule"]
    if "none" in stats:
        out.append(fmt("none", stats["none"]))
        out.append(r"\midrule")
    for m, v in sorted(((m, v) for m, v in stats.items() if m != "none"),
                       key=lambda kv: kv[1][0]):
        out.append(fmt(m, v))
    n = max(v[3] for v in stats.values())
    out += [r"\bottomrule", r"\end{tabular}",
            (r"\caption{%s at $K{=}%d$, decomposed: time per problem is tokens "
             r"generated $\times$ per-token cost. Per-token cost is what the "
             r"attention kernel sets; token count is what termination sets. "
             r"$n{=}%d$, measured on one platform (8$\times$A100-40GB). Rows "
             r"sorted by per-token cost.}" % (dataset, budget, n)),
            r"\label{tab:time-aime}", r"\end{table}"]
    return "\n".join(out)


def memory(rows, budget, dataset):
    """Peak GPU memory. Set by the cache budget rather than the policy, which
    is the claim this table has to be able to support or refute."""
    stats = {}
    for m, recs in rows.items():
        if m != "none" and m not in NAME:
            continue
        peaks = [r["peak_gpu_mb"] for r in recs if isinstance(r.get("peak_gpu_mb"), (int, float))]
        if peaks:
            stats[m] = (st.mean(peaks), len(peaks))
    if not stats:
        return "% no peak_gpu_mb records\n"

    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{@{}llr@{}}", r"\toprule",
           r"Method & FA2 & Peak MB $\downarrow$ \\", r"\midrule"]
    if "none" in stats:
        out.append("none & \\cmark & %.0f \\\\" % stats["none"][0])
        out.append(r"\midrule")
    for m, (mb, n) in sorted(((m, v) for m, v in stats.items() if m != "none"),
                             key=lambda kv: kv[1][0]):
        out.append("%s & %s & %.0f \\\\" % (NAME[m], r"\cmark" if m in FA2 else r"\xmark", mb))
    n = max(v[1] for v in stats.values())
    out += [r"\bottomrule", r"\end{tabular}",
            (r"\caption{Peak GPU memory on %s at $K{=}%d$, $n{=}%d$, one "
             r"platform (8$\times$A100-40GB).}" % (dataset, budget, n)),
            r"\label{tab:mem-aime}", r"\end{table}"]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dir of benchmark.py outputs")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--dataset", default="AIME 2024--2026 pooled")
    ap.add_argument("--memory", action="store_true", help="emit the memory table too")
    a = ap.parse_args()

    rows, unreadable = collect(a.results, a.budget)
    if unreadable:
        print("%% WARNING unreadable (mid-write?): %s" % ", ".join(unreadable))
    counts = {m: len(r) for m, r in sorted(rows.items())}
    print("%% pooled records per method at K=%d: %s" % (a.budget, counts))
    ns = set(counts.values())
    if len(ns) > 1:
        print("%% WARNING uneven coverage %s -- methods are not on the same problems" % sorted(ns))
    print()
    print(decomposition(rows, a.budget, a.dataset))
    if a.memory:
        print()
        print(memory(rows, a.budget, a.dataset))
