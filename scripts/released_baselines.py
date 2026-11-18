"""Released-code baseline cells for the three added models (Qwen3-8B, Qwen3-14B, Nemotron-Nano-8B).

apply(cells, ds) removes the adapted R-KV and LongFlow cells from grid_tables.load()'s cells and puts in the R-KV cells
run from its released code (results/rebuttal_rkv_newmodels for MATH-500, results/rebuttal_rkv_newmodels_gpqa for
GPQA-Diamond). It returns {(model, row, K): mark} for every cell with no released number: "--" where the released code
does not support the model (LongFlow on Nemotron), "$\\circ$" where the released-code run is not complete.
A released cell counts only when every problem of the benchmark is present, none partial, none errored.
"""
import glob
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
TAG = {"qwen3": "q3_8b", "q14b": "q3_14b", "nemo": "nemo"}
N = {"math": 495, "gpqa": 198}
PENDING, UNSUPPORTED = "$\\circ$", "--"


def released_rkv(ds, m, K):
    pat = (R / "rebuttal_rkv_newmodels" / f"rn_{TAG[m]}_rkv_official_K{K}_s*.json") if ds == "math" else \
          (R / "rebuttal_rkv_newmodels_gpqa" / f"rn_gpqa_{TAG[m]}_rkv_official_K{K}_s*.json")
    out = {}
    for f in glob.glob(str(pat)):
        s = int(re.search(r"_s(\d+)\.json$", f).group(1))
        e = json.load(open(f))["results"]["rkv_official"][str(K)]
        if e.get("partial") or any(p.get("error") or not p.get("n_tokens_generated") for p in e["per_problem"]):
            return None
        for i, p in enumerate(e["per_problem"]):
            out[s + i] = p
    return out if len(out) == N[ds] else None


def apply(cells, ds, models, Ks):
    marks = {}
    for key in [k for k in cells if k[0] == ds and k[2] in ("rkv", "longflow")]:
        del cells[key]
    for m in models:
        for K in Ks:
            c = released_rkv(ds, m, K)
            if c is None:
                marks[(m, "rkv", K)] = PENDING
            else:
                cells[(ds, m, "rkv", K)] = c
            marks[(m, "longflow", K)] = UNSUPPORTED if m == "nemo" else PENDING
    return marks
