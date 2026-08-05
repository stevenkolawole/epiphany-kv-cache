"""Split the trace slice into N GPU shards balanced by estimated window count.

Occlusion cost scales with the number of 32-token windows, not the number of
traces, and that is wildly uneven here: most traces need under 100 windows while
one needs ~1129. Round-robin would leave one GPU running hours after the rest go
idle, so shards are packed longest-first onto whichever shard is currently
lightest (LPT scheduling).
"""
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
outdir = Path(sys.argv[2])
n_shards = int(sys.argv[3]) if len(sys.argv) > 3 else 8
WINDOW, STRIDE = 32, 16

traces = [json.loads(l) for l in open(src)]


def est_windows(t):
    # answer_start is only knowable with a tokenizer; the 64-token tail is the
    # same fallback find_answer_start() uses, and this is for balancing only.
    span = max(0, len(t["token_ids"]) - t["prompt_len"] - 64)
    return max(1, -(-span // STRIDE))


order = sorted(range(len(traces)), key=lambda i: -est_windows(traces[i]))
shards = [[] for _ in range(n_shards)]
load = [0] * n_shards
for i in order:
    j = load.index(min(load))
    shards[j].append(i)
    load[j] += est_windows(traces[i])

outdir.mkdir(parents=True, exist_ok=True)
for j, idxs in enumerate(shards):
    with open(outdir / f"shard{j}.jsonl", "w") as f:
        for i in sorted(idxs):          # keep original order within a shard
            f.write(json.dumps(traces[i]) + "\n")
    print(f"shard{j}: {len(idxs):2d} traces, ~{load[j]:5d} windows")
print(f"total ~{sum(load)} windows; max/min shard = {max(load)}/{min(load)} "
      f"({max(load)/max(min(load),1):.2f}x imbalance)")
