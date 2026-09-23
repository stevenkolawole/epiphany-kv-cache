#!/usr/bin/env python3
"""Termination accounting at n=495 (Sec. 4.4 exhibit): for each method at a budget K, how many
of its errors are generations that ran to the length cap (cap hits) versus terminated wrong
answers, on both models; and the decomposition of EpiKV's paired wins/losses over the
query-blind Seg into cap-hit flips.

Same per-problem records as scripts/n500_table.py (problems 0-99 from results/{llama,qwen}_logical
and results/win; 100-499 from results/math_n500_logical). A cap hit is a generation of
max_new_tokens (8,192) tokens.

  python scripts/termination_table.py            # markdown
  python scripts/termination_table.py --latex    # LaTeX rows
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acc_tables import MATH_DIRS, WIN, WIN_MODEL, Q_TAG  # noqa: E402

R = Path(__file__).resolve().parent.parent / "results"
N500 = R / "math_n500_logical"
CAP = 8192
ROWS = [("none", "none"), ("epikv_q", "\\methodq"), ("kv_seg_hs_tau128", "\\methodseg"),
        ("raas", "RaaS"), ("h2o", "H2O"), ("thinkv_faithful", "ThinKV")]


def recs(tag, K, model, d):
    if tag == "epikv_q":
        f = WIN / f"{Q_TAG}_K{K}_{model}.json"
    elif tag == "none":
        f = R / d / "none_K1024.json"
    else:
        f = R / d / f"{tag}_K{K}.json"
    j = json.load(open(f))
    m = next(iter(j["results"]))
    pp = list(j["results"][m][next(iter(j["results"][m]))]["per_problem"])
    Kf = 1024 if tag == "none" else K
    for S in (100, 200, 300, 400):
        j = json.load(open(N500 / f"{tag}_K{Kf}_s{S}_{model}.json"))
        m = next(iter(j["results"]))
        pp += j["results"][m][next(iter(j["results"][m]))]["per_problem"]
    return pp


def account(pp):
    err = [p for p in pp if not p["correct"]]
    cap = [p for p in err if p["n_tokens_generated"] >= CAP]
    return len(pp), len(err), len(cap), len(err) - len(cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    K = a.K
    data = {}
    for label, d in MATH_DIRS.items():
        model = "llama" if "llama" in d else "qwen"
        data[label] = {tag: recs(tag, K, model, d) for tag, _ in ROWS}
    models = list(MATH_DIRS)
    if a.latex:
        # Comments first: the rows file is \input inside a tabular, and anything after the last
        # row (a blank line, a comment) opens a new cell before \midrule and breaks the table.
        print(f"% termination accounting at K={K}; columns per model: errors, cap hits, terminated wrong")
        for label in models:
            print("% " + decomposition(data[label], K))
        for tag, tex in ROWS:
            cells = []
            for label in models:
                n, e, c, t = account(data[label][tag])
                cells += [str(e), str(c), str(t)]
            print(tex + " & " + " & ".join(cells) + " \\\\")
    else:
        print(f"\n### Termination accounting at K={K} (n = {len(data[models[0]]['none'])})\n")
        print("| method | " + " | ".join(f"{m} errors | cap hits | terminated wrong" for m in models) + " |")
        print("|---|" + "---|" * (3 * len(models)))
        for tag, tex in ROWS:
            cells = []
            for label in models:
                n, e, c, t = account(data[label][tag])
                cells += [str(e), str(c), str(t)]
            print(f"| {tag} | " + " | ".join(cells) + " |")
    if not a.latex:
        print()
        for label in models:
            print(decomposition(data[label], K))


def decomposition(d, K):
    """EpiKV's paired wins/losses over the query-blind Seg, split by whether the loser hit the cap."""
    e, s = d["epikv_q"], d["kv_seg_hs_tau128"]
    wins = [(x, y) for x, y in zip(e, s) if x["correct"] and not y["correct"]]
    losses = [(x, y) for x, y in zip(e, s) if y["correct"] and not x["correct"]]
    wc = sum(y["n_tokens_generated"] >= CAP for x, y in wins)
    lc = sum(x["n_tokens_generated"] >= CAP for x, y in losses)
    return (f"EpiKV over Seg at K={K}: {len(wins)} wins, of which {wc} were Seg cap hits; "
            f"{len(losses)} losses, of which {lc} are EpiKV cap hits")


if __name__ == "__main__":
    main()
