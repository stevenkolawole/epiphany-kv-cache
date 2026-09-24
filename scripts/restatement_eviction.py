#!/usr/bin/env python3
"""Were the blocks a looping model restates already evicted when it restated them?

Input: dump_retained.py --json_out files that carry per-token `evicted_at` (the decode
step at which the token left the cache; None = still cached at the end).

A restatement is the first reappearance of an n-token span (default n=16) that occurred
earlier in the generation. For each one we ask whether the earlier occurrence was out of
the cache at the moment of the reappearance (all of its tokens evicted, or any of them).

At a tight budget most of the history is evicted at any moment, so the answer means
something only against a base rate. For each restatement the base rate is the fraction
of same-length spans lying the same distance back from other steps of the trace (every
seventh step) that were fully evicted at that step. Reported per trace and pooled.
Tight loops (a span reappearing before it ends) are excluded; the question is about
re-derivations of blocks written long before.

    python3 scripts/restatement_eviction.py ~/exhibits_evt/p*_seg.json
"""
import json
import sys


def load(p):
    d = json.load(open(p))
    toks = d["tokens"]
    if "evicted_at" not in toks[0]:
        sys.exit(f"{p}: no evicted_at field (rerun dump_retained.py)")
    return d, [t["text"] for t in toks], [t["evicted_at"] for t in toks]


def gone(ev, lo, hi, step):
    """fraction of tokens [lo, hi) evicted strictly before decode step `step`"""
    n = sum(1 for e in ev[lo:hi] if e is not None and e < step)
    return n / (hi - lo)


def analyse(p, n=16):
    d, text, ev = load(p)
    first = {}
    rows = []
    seen_restated = set()
    for t in range(len(text) - n + 1):
        key = tuple(text[t:t + n])
        if key in first:
            src = first[key]
            if key not in seen_restated and t - src >= n:   # first reappearance, no overlap
                seen_restated.add(key)
                g = gone(ev, src, src + n, t)
                # base rate: every non-overlapping span at the same distance back, at other steps
                dist = t - src
                base = [gone(ev, u - dist, u - dist + n, u)
                        for u in range(dist, len(text) - n + 1, 7) if u != t]
                rows.append((t, src, g, sum(1 for b in base if b == 1.0) / max(1, len(base))))
        else:
            first[key] = t
    return d, rows


def main():
    n = 16
    pooled = []
    for p in sys.argv[1:]:
        d, rows = analyse(p, n)
        if not rows:
            print(f"{p}: no restatements"); continue
        full = sum(1 for r in rows if r[2] == 1.0) / len(rows)
        anyg = sum(1 for r in rows if r[2] > 0) / len(rows)
        base = sum(r[3] for r in rows) / len(rows)
        t0 = rows[0][0]
        print(f"p{d['problem_idx']:>3} gen={d['generated']:>5} restatements={len(rows):>4} "
              f"first_at={t0:>5}  src fully evicted={full:5.1%}  any evicted={anyg:5.1%}  "
              f"base rate (same distance)={base:5.1%}")
        pooled += rows
    if pooled:
        print(f"pooled: {len(pooled)} restatements, src fully evicted "
              f"{sum(1 for r in pooled if r[2] == 1.0) / len(pooled):.1%}, "
              f"base {sum(r[3] for r in pooled) / len(pooled):.1%}")


if __name__ == "__main__":
    main()
