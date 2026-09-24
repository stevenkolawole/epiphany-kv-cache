#!/usr/bin/env python3
"""Print accuracy by budget for every result file in a directory.

Exists because inlining this over ssh keeps running into shell and f-string
quoting collisions, and a misquoted reader that prints nothing looks exactly
like a run that produced nothing.
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--per-problem", action="store_true",
                    help="also report cap-hit rate and mean tokens")
    args = ap.parse_args()

    for d in args.dirs:
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        if not files:
            print(f"{d}: (empty)")
            continue
        print(f"=== {d}: {len(files)} files ===")
        for f in files:
            try:
                rec = json.load(open(f))
            except Exception as exc:  # noqa: BLE001
                print(f"  {os.path.basename(f)}: UNREADABLE {exc}")
                continue
            meta = rec.get("meta", {})
            cap = meta.get("max_new_tokens")
            for method, by_k in rec.get("results", {}).items():
                cells = []
                for k, v in sorted(by_k.items(), key=lambda x: int(x[0])):
                    acc = 100.0 * v.get("accuracy", 0.0)
                    cell = f"{k}:{acc:.1f}"
                    if args.per_problem and v.get("per_problem"):
                        pp = v["per_problem"]
                        hits = sum(1 for p in pp
                                   if (p.get("n_tokens_generated") or 0) >= cap)
                        cell += f"(cap{100*hits/len(pp):.0f}%)"
                    cells.append(cell)
                n = meta.get("n_samples")
                print(f"  {method:22s} n={n:<4} {'  '.join(cells)}")


if __name__ == "__main__":
    main()
