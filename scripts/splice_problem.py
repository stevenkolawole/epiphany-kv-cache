#!/usr/bin/env python3
"""Splice a single-problem rerun into a grid result file whose one problem
errored (CUDA OOM on a card shared by several processes), and record the
splice in the file's meta so the number stays traceable.

    python3 scripts/splice_problem.py results/llama_logical/none_K1024.json.ERRORED \
        results/llama_logical/none_K1024_p38.json results/llama_logical/none_K1024.json \
        --note "problem 38 rerun alone on box 3 card 4 (same stack) after CUDA OOM on box 1"

Greedy decoding with a fixed problem index makes the rerun the same
computation the grid would have done; only the host changed.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("fix")
    ap.add_argument("out")
    ap.add_argument("--note", default="")
    ap.add_argument("--start_idx", type=int, required=True,
                    help="the --start_idx the rerun was launched with (benchmark.py does not store it)")
    a = ap.parse_args()
    g = json.load(open(a.grid))
    f = json.load(open(a.fix))
    fix_start = a.start_idx
    spliced = []
    for meth, per in g["results"].items():
        for K, r in per.items():
            fr = f["results"][meth][K]
            pp = r["per_problem"]
            for j, p in enumerate(fr["per_problem"]):
                i = fix_start + j
                old = pp[i]
                if not (old.get("error") or not old.get("n_tokens_generated")):
                    raise SystemExit(f"problem {i} in the grid file did not error; refusing to overwrite")
                if old["problem"] != p["problem"]:
                    raise SystemExit(f"problem text mismatch at index {i}")
                pp[i] = p
                spliced.append(i)
            n = len(pp)
            c = sum(1 for p in pp if p.get("correct"))
            r["n_correct"] = c
            r["n_total"] = n
            r["accuracy"] = c / n
            times = [p["wall_time_s"] for p in pp if p.get("wall_time_s") is not None]
            mems = [p["peak_gpu_mb"] for p in pp if p.get("peak_gpu_mb") is not None]
            r["mean_wall_time_s"] = round(sum(times) / len(times), 2) if times else None
            r["mean_peak_gpu_mb"] = round(sum(mems) / len(mems), 1) if mems else None
            print(f"{meth} K={K}: {c}/{n} = {100*c/n:.1f}% (spliced problems {spliced})")
    g["meta"]["spliced_problems"] = spliced
    g["meta"]["splice_note"] = a.note
    g["meta"]["splice_source"] = a.fix
    json.dump(g, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
