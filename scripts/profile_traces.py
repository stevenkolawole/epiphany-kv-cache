#!/usr/bin/env python3
"""Profile retained-token dumps: generated tokens, evictions, kept-by-score,
never-ranked, and how many real words the score kept.

    python3 scripts/profile_traces.py <dir-with-K*_*.json>
"""
import glob
import json
import os
import sys

d = sys.argv[1]
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    j = json.load(open(f))
    t = j["tokens"]
    steps = j.get("eviction_steps", [])
    L = (steps[-1] - 128) if steps else -1
    sc = sum(1 for i, x in enumerate(t) if x["kept"] and not x["recency"] and i <= L)
    pale = sum(1 for i, x in enumerate(t) if x["kept"] and not x["recency"] and i > L)
    kept = [x["text"] for i, x in enumerate(t) if x["kept"] and not x["recency"] and i <= L]
    words = sum(1 for k in kept if k.strip().isalpha() and len(k.strip()) > 2)
    print(f"{os.path.basename(f):28s} gen={j['generated']:5d} retained={j['retained_generated']:4d} "
          f"evictions={len(steps):2d} first={steps[:1]} last={steps[-1:]} by_score={sc:4d} "
          f"never_ranked={pale:4d} words={words:4d} gt={str(j['ground_truth'])[:12]}")
