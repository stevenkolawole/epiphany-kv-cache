#!/usr/bin/env python3
"""Print the tag of every result file that is COMPLETE.

A result file exists from the first finished problem onward (that is how --resume
works), so existence is not completion. The harness marks each (method, K) block
``partial`` until the last problem is in; a cell is done when no block is partial
AND no problem errored.

Oct 6 21:30: benchmark.py catches a per-problem exception (CUDA OOM in practice),
records pred=None with 0 tokens -- scored as wrong -- and carries on, and after one
OOM the process usually fails every later problem too. Such a file ends with
partial=false and was counted done: 78 manifest cells, 43 with all 11 problems
errored. A problem with an error or 0 generated tokens is not a result.
Usage: list_complete.py DIR [DIR...]
"""
import glob, json, os, sys
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            r = json.load(open(f)).get("results", {})
            blocks = [v for m in r.values() for v in m.values()]
            ok = bool(blocks) and all(not v.get("partial", False) for v in blocks) and all(
                not p.get("error") and p.get("n_tokens_generated", 0) > 0
                for v in blocks for p in v.get("per_problem", []))
        except Exception:
            ok = False
        if ok:
            print(os.path.basename(f)[:-5])
