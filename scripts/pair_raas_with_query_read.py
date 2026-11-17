"""Original RaaS (eager attention) against RaaS's rule on EpiKV's query read, paired on the problems both ran.
Prints accuracy of each, wins/losses of the query-read run, and the exact McNemar p. Problems with an error or
no generated tokens are dropped and counted.   python scripts/pair_raas_with_query_read.py [results/rebuttal_raas_pair]"""
import collections, glob, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool_aime import mcnemar_exact  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "results/rebuttal_raas_pair"
cells, dropped = collections.defaultdict(dict), collections.Counter()
for f in sorted(glob.glob(f"{src}/rp_*.json")):
    m = re.match(r".*/rp_(llama|qwen)_(raas(?:_epikv_window)?)_K(\d+)_s(\d+)\.json", f)
    if not m:
        continue
    model, method, budget, shard = m.groups()
    for block in json.load(open(f))["results"].values():
        for v in block.values():
            if not isinstance(v, dict) or "per_problem" not in v:
                continue
            for i, p in enumerate(v["per_problem"]):
                if p.get("error") or not p.get("n_tokens_generated"):
                    dropped[(model, method, budget)] += 1
                else:
                    cells[(model, method, budget)][(int(shard), i)] = p["correct"]
for model in ("llama", "qwen"):
    for budget in sorted({k[2] for k in cells}, key=int):
        a, b = cells.get((model, "raas", budget), {}), cells.get((model, "raas_epikv_window", budget), {})
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        w = sum(b[q] and not a[q] for q in shared)
        l = sum(a[q] and not b[q] for q in shared)
        print(f"{model} K={budget} n={len(shared)} dropped={dropped[(model, 'raas', budget)] + dropped[(model, 'raas_epikv_window', budget)]}: "
              f"original {100 * sum(a[q] for q in shared) / len(shared):.1f} | on query read {100 * sum(b[q] for q in shared) / len(shared):.1f} | "
              f"wins/losses {w}/{l} p={mcnemar_exact(w, l):.3g}")
