#!/usr/bin/env python3
"""Pool the GPQA-Diamond shard files into one table.

Reads results_gpqa/gpqa_<model>_<policy>_K<K>_s<start>.json (one method, one cache
size each). A shard counts only when every policy in --policies has a complete record for
it under the same model, so every number below is paired on the same problems.

Prints per (model, K, policy): n, accuracy, cap hits, terminated wrong, and wins/losses
against the uncapped model on the same problems.
"""
import argparse, glob, json, os, re, collections

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default="results/gpqa_p5")
ap.add_argument("--policies", nargs="+", default=["none", "epikv", "seg", "raas"])
ap.add_argument("--cap", type=int, default=16384)
ap.add_argument("--latex", type=str, default=None, help="write appendix table rows to this file")
a = ap.parse_args()

pat = re.compile(r"gpqa_(?P<m>[a-z]+)_(?P<p>[a-z0-9]+)_K(?P<k>\d+)_s(?P<s>\d+)\.json$")
recs = {}  # (model, policy, K, start) -> per_problem list
for f in glob.glob(os.path.join(a.root, "gpqa_*.json")):
    mm = pat.search(os.path.basename(f))
    if not mm:
        continue
    d = json.load(open(f))
    for meth, cells in d["results"].items():
        for k, cell in cells.items():
            pp = cell.get("per_problem", [])
            if cell.get("partial") or not pp:
                continue
            recs[(mm["m"], mm["p"], int(mm["k"]), int(mm["s"]))] = pp

ks = sorted({k for (_, _, k, _) in recs})
LATEX = {}  # (model, K) -> {policy: (n, acc, cap, wrong, win, loss)}
for model in ("llama", "qwen"):
    for K in ks:
        shards = sorted({s for (m, p, k, s) in recs if m == model and k == K})
        # 'none' is stored under its own K (the first K in the grid); accept any K for it.
        def get(p, s):
            if p == "none":
                for (m2, p2, k2, s2), v in recs.items():
                    if (m2, p2, s2) == (model, "none", s):
                        return v
                return None
            return recs.get((model, p, K, s))
        complete = [s for s in shards if all(get(p, s) is not None for p in a.policies)]
        if not complete:
            continue
        print(f"\n{model}  K={K}  complete paired shards: {len(complete)}  "
              f"({sum(len(get('none', s)) for s in complete)} problems)")
        print(f"{'policy':8s} {'n':>4s} {'acc%':>6s} {'cap':>4s} {'wrong':>5s} {'win/loss vs none':>16s}")
        # A problem that errored (e.g. OOM) under any policy is dropped for every policy,
        # so the comparison stays paired; it is listed for a rerun.
        bad = set()
        for s in complete:
            for p in a.policies:
                for i, e in enumerate(get(p, s)):
                    if e.get("error"):
                        bad.add((s, i)); print(f"  errored: {model} {p} K={K} shard {s} idx {i}: {str(e['error'])[:60]}")
        keep = lambda s, pp: [e for i, e in enumerate(pp) if (s, i) not in bad]
        for p in a.policies:
            pp = [e for s in complete for e in keep(s, get(p, s))]
            base = [e for s in complete for e in keep(s, get("none", s))]
            n = len(pp); acc = sum(bool(e["correct"]) for e in pp)
            cap = sum(e["n_tokens_generated"] >= a.cap for e in pp)
            wrong = sum((not e["correct"]) and e["n_tokens_generated"] < a.cap for e in pp)
            win = sum(bool(x["correct"]) and not y["correct"] for x, y in zip(pp, base))
            loss = sum(bool(y["correct"]) and not x["correct"] for x, y in zip(pp, base))
            print(f"{p:8s} {n:4d} {100*acc/n:6.1f} {cap:4d} {wrong:5d} {win:7d}/{loss:<8d}")
            LATEX.setdefault((model, K), {})[p] = (n, 100*acc/n, cap, wrong, win, loss)
        # paired EpiKV against each other policy, exact McNemar (two-sided binomial on discordant pairs)
        from math import comb
        def mcnemar(w, l):
            n = w + l
            if n == 0: return 1.0
            k = min(w, l); pl = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
            return min(1.0, 2 * pl)
        if "epikv" in a.policies:
            e = [x for s in complete for x in keep(s, get("epikv", s))]
            for p in a.policies:
                if p == "epikv": continue
                o = [x for s in complete for x in keep(s, get(p, s))]
                w = sum(bool(x["correct"]) and not y["correct"] for x, y in zip(e, o))
                l = sum(bool(y["correct"]) and not x["correct"] for x, y in zip(e, o))
                wl = sum(bool(x["correct"]) and not y["correct"] and y["n_tokens_generated"] >= a.cap for x, y in zip(e, o))
                print(f"  epikv vs {p:6s}: wins {w:3d} ({wl} looped under {p}) / losses {l:3d}   p={mcnemar(w, l):.2f}")

if a.latex:
    NAME = {"none": "No eviction", "epikv": "\\rowcolor{epirow}\\methodq", "seg": "\\methodseg", "raas": "RaaS"}
    ks_l = sorted({k for (_, k) in LATEX})
    # only budgets paired on both models with the full problem set
    full = [k for k in ks_l if all((m, k) in LATEX and LATEX[(m, k)]["none"][0] >= 180 for m in ("llama", "qwen"))]
    with open(a.latex, "w") as f:
        f.write("% generated by scripts/pool_gpqa.py --latex from results/gpqa_p5; cells: cap hits, terminated wrong (accuracy: five_model_gpqa_rows.py)\n")
        f.write("% n per cell: " + ", ".join(f"{m} K={k}: {LATEX[(m,k)]['none'][0]}" for m in ("llama","qwen") for k in full) + "\n")
        for p in ("none", "epikv", "seg", "raas"):
            cells = []
            for m in ("llama", "qwen"):
                for k in full:
                    n, acc, cap, wrong, win, loss = LATEX[(m, k)][p]
                    cells.append(f"{cap} & {wrong}")
            f.write(NAME[p] + " & " + " & ".join(cells) + " \\\\\n")
    print("wrote", a.latex, "budgets", full)
