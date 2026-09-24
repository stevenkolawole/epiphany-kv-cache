#!/usr/bin/env python3
"""Pooled, paired comparison of two methods over sharded result files.

Shards are pooled by problem text, not by averaging shard accuracies, so a
shard that ran a different number of problems cannot silently reweight the
result. The comparison is paired because both methods run the same problems,
and an exact McNemar on the discordant pairs is the right test for that design;
unpaired Wilson intervals throw away the pairing and will hide a real effect.

Also reports cap-hit, since the paper attributes tight-budget accuracy loss to
non-termination rather than token choice, and a change in accuracy that moves
with cap-hit means something different from one that does not.
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict


def exact_mcnemar(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def wilson(correct, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(dirs, methods):
    # (method, K) -> {problem: (correct, cap_hit)}
    per = defaultdict(dict)
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            rec = json.load(open(f))
            cap = rec["meta"]["max_new_tokens"]
            for method, by_k in rec.get("results", {}).items():
                if methods and method not in methods:
                    continue
                for k, v in by_k.items():
                    for p in v.get("per_problem", []):
                        per[(method, int(k))][p["problem"]] = (
                            bool(p.get("correct")),
                            (p.get("n_tokens_generated") or 0) >= cap,
                        )
    return per


def load_files(path_a, path_b, label_a, label_b):
    """Pair two result FILES that share a method name (ablation cells).

    The band and sparsity ablations all run `hs_variance_detrend` or
    `kv_seg_hs` under different flags, so the method key cannot tell the
    cells apart; the file can. Records are re-keyed by the given labels.
    """
    per = defaultdict(dict)
    for path, label in ((path_a, label_a), (path_b, label_b)):
        for f in sorted(glob.glob(path)):
            rec = json.load(open(f))
            cap = rec["meta"]["max_new_tokens"]
            for _method, by_k in rec.get("results", {}).items():
                for k, v in by_k.items():
                    for p in v.get("per_problem", []):
                        per[(label, int(k))][p["problem"]] = (
                            bool(p.get("correct")),
                            (p.get("n_tokens_generated") or 0) >= cap,
                        )
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--a", required=True, help="baseline method, or a label with --a-file")
    ap.add_argument("--b", required=True, help="variant method, or a label with --b-file")
    ap.add_argument("--a-file", help="glob of result files for the baseline cell")
    ap.add_argument("--b-file", help="glob of result files for the variant cell")
    args = ap.parse_args()

    if args.a_file or args.b_file:
        per = load_files(args.a_file, args.b_file, args.a, args.b)
    else:
        per = load(args.dirs, {args.a, args.b})
    budgets = sorted({k for (_, k) in per})
    print(f"{'K':>6} {'n':>5}  {args.a:>22} {args.b:>22}   paired")
    for K in budgets:
        A, B = per.get((args.a, K)), per.get((args.b, K))
        if not A or not B:
            continue
        common = sorted(set(A) & set(B))
        if not common:
            continue
        ac = sum(A[q][0] for q in common)
        bc = sum(B[q][0] for q in common)
        acap = sum(A[q][1] for q in common)
        bcap = sum(B[q][1] for q in common)
        wins = sum(1 for q in common if B[q][0] and not A[q][0])
        losses = sum(1 for q in common if A[q][0] and not B[q][0])
        p = exact_mcnemar(wins, losses)
        n = len(common)
        alo, ahi = wilson(ac, n)
        blo, bhi = wilson(bc, n)
        print(f"{K:>6} {n:>5}  "
              f"{100*ac/n:>6.1f} [{alo:>5.1f},{ahi:>5.1f}] cap{100*acap/n:>5.1f}  "
              f"{100*bc/n:>6.1f} [{blo:>5.1f},{bhi:>5.1f}] cap{100*bcap/n:>5.1f}   "
              f"{args.b}+{wins}/-{losses} p={p:.4f}")


if __name__ == "__main__":
    main()
