"""Live KV-cache occupancy per policy and budget, from the kv_live record benchmark.py writes per problem:
peak over problems of the per-problem peak, and mean over problems of the per-problem mean. Complete, error-free
blocks only.   python scripts/occupancy_table.py [results/rebuttal_occupancy]"""
import glob, json, statistics as st, sys

src = sys.argv[1] if len(sys.argv) > 1 else "results/rebuttal_occupancy"
rows = {}
for f in sorted(glob.glob(f"{src}/*.json")):
    for m, b in json.load(open(f))["results"].items():
        for k, v in b.items():
            if not isinstance(v, dict) or "per_problem" not in v or v.get("partial"):
                continue
            pp = v["per_problem"]
            if any(p.get("error") for p in pp):
                continue
            live = [p["kv_live"] for p in pp]
            rows[(m, int(k))] = (len(pp), max(x["peak_all"] for x in live), st.mean(x["mean_all"] for x in live))
print("policy            K     n  peak  mean")
for (m, k), (n, peak, mean) in sorted(rows.items()):
    print(f"{m:16s} {k:5d} {n:5d} {peak:5d} {mean:6.0f}")
