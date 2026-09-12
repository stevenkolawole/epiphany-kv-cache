#!/usr/bin/env python3
"""The tau ablation table (EpiKV paper, appendix): EpiKV-Seg evicting every
tau decode steps, on Babel L40S (transformers 5.2.0, one process per card),
Llama-8B, MATH-500 problems 0-99, greedy, 8,192-token cap.

Rows: none (no eviction; problems 25-99 only), score-only control
(tau = 1e9: scores every step, never evicts), tau = 1, 32, 128.
Columns per K: accuracy, paired wins/losses against tau=1 on the shared
problems, cap-hit rate, mean tokens generated, decode tokens/s per process
(sum of tokens / sum of wall), seconds per problem.

    python3 scripts/tau_table.py            # markdown to stdout
    python3 scripts/tau_table.py --latex    # LaTeX rows
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_ablation import load_dir  # noqa: E402

R = Path(__file__).resolve().parent.parent / "results"
CAP = 8192
ROWS = [("none", "no eviction"), ("scoreonly", "score only ($\\tau{=}\\infty$)"),
        ("tau1", "$\\tau{=}1$ (every step)"), ("tau32", "$\\tau{=}32$"), ("tau128", "$\\tau{=}128$")]


def stats(ps, ref=None):
    n = len(ps)
    acc = 100.0 * sum(bool(p["correct"]) for p in ps.values()) / n
    toks = sum(p["n_tokens_generated"] for p in ps.values())
    wall = sum(p["wall_time_s"] for p in ps.values())
    cap = 100.0 * sum(1 for p in ps.values() if p["n_tokens_generated"] >= CAP) / n
    out = dict(n=n, acc=acc, cap=cap, tok=toks / n, tps=toks / wall, spp=wall / n)
    if ref is not None:
        shared = sorted(set(ps) & set(ref))
        out["w"] = sum(1 for i in shared if ps[i]["correct"] and not ref[i]["correct"])
        out["l"] = sum(1 for i in shared if ref[i]["correct"] and not ps[i]["correct"])
        out["shared"] = len(shared)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    p = load_dir(str(R / "tau_logical"))
    b = load_dir(str(R / "bandabl_logical"))
    none = {}
    for t in ("none_s25", "none_s50", "none_s75"):
        for K in b[t]:
            none.setdefault(K, {}).update(b[t][K])
    p["none"] = none
    Ks = [512, 1024, 2048, 4096]
    lines = []
    for K in Ks:
        ref = p["tau1"][K]
        for tag, label in ROWS:
            if K not in p.get(tag, {}):
                continue
            s = stats(p[tag][K], ref if tag != "tau1" else None)
            wl = "--" if tag == "tau1" else f"{s['w']}/{s['l']}"
            if a.latex:
                lines.append(f"{K} & {label} & {s['acc']:.0f} & {wl} & {s['cap']:.0f} & {s['tok']:,.0f} & "
                             f"{s['tps']:.1f} & {s['spp']:.0f} \\\\")
            else:
                lines.append(f"| {K} | {label} | {s['acc']:.0f} (n={s['n']}) | {wl} | {s['cap']:.0f} | "
                             f"{s['tok']:,.0f} | {s['tps']:.1f} | {s['spp']:.0f} |")
        if a.latex:
            lines.append("\\midrule")
    if not a.latex:
        print("| K | row | acc % | W/L vs tau=1 | cap-hit % | tokens | tok/s | s/problem |")
        print("|---|---|---|---|---|---|---|---|")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
