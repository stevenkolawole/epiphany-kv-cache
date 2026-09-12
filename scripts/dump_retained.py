#!/usr/bin/env python3
"""Show what EpiKV-Seg actually keeps and drops on a full trace at a budget.

Two reviewers asked for this and it is the one thing the name "epiphany" rests
on: are the highest-scoring, retained tokens concluded steps, committed
intermediate results, and self-corrections, or are they something else? The
labels-vs-signal dump (inspect_traces.py) cannot answer it because it never runs
eviction; this does. It generates one problem greedily under kv_seg_hs at the
requested budget, records the keep mask the policy leaves behind at the end of
generation, and prints the generated text with retained tokens marked.

Output is plain text so it can be read, quoted, and pasted into an appendix.
Marking: retained tokens are wrapped as [tok]; evicted tokens are bare. The
recency window and prefill are always retained by construction and are labelled
as such so they are not mistaken for scored survivors.

    python3 scripts/dump_retained.py --cache_size 1024 --problem_idx 0 \
        > reports/retained_K1024_p0.txt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eviction import (EvictionConfig, KVSegHSEviction, H2OEviction,  # noqa: E402
                          DetrendendHSVarianceEviction)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402  (reuses the harness's prompt + loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/deepseek-r1-distill-llama-8b")
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--problem_idx", type=int, default=0)
    ap.add_argument("--cache_size", type=int, default=1024)
    ap.add_argument("--keep_recent_k", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--band_a_layer", type=int, default=10)
    ap.add_argument("--band_b_layer", type=int, default=21)
    # H2O is the contrast for the figure: attention-based eviction on the same
    # trace at the same budget. It needs eager attention and attention weights.
    ap.add_argument("--method", default="kv_seg_hs",
                    choices=["kv_seg_hs", "h2o", "hs_variance_detrend"])
    ap.add_argument("--attn_impl", default=None,
                    help="default: eager for h2o, flash_attention_2 otherwise")
    ap.add_argument("--json_out", default=None,
                    help="also write a sidecar for the figure generator: per generated "
                         "token its text, score at scoring time, and final kept/recency flags")
    ap.add_argument("--refresh_tau", type=int, default=1,
                    help="EpiKV-Seg: evict every tau decode steps (128 = the shipped / vLLM setting)")
    ap.add_argument("--logical_positions", action="store_true",
                    help="embed post-eviction tokens at their logical position (the corrected convention)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    needs_attn = args.method == "h2o"
    impl = args.attn_impl or ("eager" if needs_attn else "flash_attention_2")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=impl
    ).to(dev).eval()

    problems = bm.load_problems(args.dataset, n_samples=args.problem_idx + 1, start_idx=0)
    prob = problems[args.problem_idx]
    cfg = EvictionConfig(cache_size=args.cache_size, keep_recent_k=args.keep_recent_k)
    if args.method == "h2o":
        ev = H2OEviction(cfg)
    elif args.method == "hs_variance_detrend":
        ev = DetrendendHSVarianceEviction(cfg, band_a_layer=args.band_a_layer,
                                         band_b_layer=args.band_b_layer)
    else:
        ev = KVSegHSEviction(cfg, band_a_layer=args.band_a_layer, band_b_layer=args.band_b_layer,
                             refresh_tau=args.refresh_tau)

    prompt_ids = bm.build_prompt_ids(tok, prob["problem"], dev)
    prefill_len = prompt_ids.shape[1]
    # The hidden-state policies take prefill_len; H2O's reset() takes nothing.
    try:
        ev.reset(prefill_len=prefill_len)
    except TypeError:
        ev.reset()

    # Track survival by ORIGINAL position. The cache is physically compacted on
    # each eviction, so we keep a list mapping current cache slot -> original
    # absolute position and shrink it alongside the cache.
    alive = list(range(prefill_len))
    generated = []
    # Score of each generated token at the moment it was scored. The policy's
    # own _scores list is pruned on eviction (it keeps only survivors), so the
    # evicted tokens' scores would be lost without capturing them here.
    scores_at_time = []
    with torch.no_grad():
        out = model(prompt_ids, use_cache=True, output_hidden_states=True,
                    output_attentions=needs_attn)
        past = out.past_key_values
        if hasattr(ev, "set_prefill_end"):
            ev.set_prefill_end(out.hidden_states)
        next_id = out.logits[:, -1].argmax(-1, keepdim=True)
        for step in range(args.max_new_tokens):
            generated.append(int(next_id))
            alive.append(prefill_len + step)
            if int(next_id) == tok.eos_token_id:
                break
            pos_kw = {}
            if args.logical_positions:
                # same convention as benchmark.py --logical_positions: the new token
                # sits at prefill_len + step regardless of the cache's physical length
                p = torch.tensor([[prefill_len + step]], device=dev)
                pos_kw = {"position_ids": p, "cache_position": p[0]}
            out = model(next_id, past_key_values=past, use_cache=True,
                        output_hidden_states=True, output_attentions=needs_attn, **pos_kw)
            past = out.past_key_values
            legacy = past.to_legacy_cache() if hasattr(past, "to_legacy_cache") else past
            before = legacy[0][0].shape[2]
            signal = out.attentions if needs_attn else out.hidden_states
            new_legacy = ev.evict_past_key_values(legacy, signal)
            sc = getattr(ev, "_scores", None)
            scores_at_time.append(float(sc[-1]) if sc else float("nan"))
            after = new_legacy[0][0].shape[2]
            if after < before:
                # Recover which slots survived: KVSegHSEviction keeps prefill,
                # a recency tail, and scored survivors. Re-derive the mask from
                # the policy's own last decision so the report matches exactly.
                keep = ev.last_keep_mask.cpu().tolist()
                alive = [pos for pos, k in zip(alive, keep) if k]
                past = type(past).from_legacy_cache(new_legacy) if hasattr(past, "from_legacy_cache") else new_legacy
                if hasattr(past, "_seen_tokens"):
                    past._seen_tokens = after
            next_id = out.logits[:, -1].argmax(-1, keepdim=True)

    kept = set(alive)
    seq_len = prefill_len + len(generated)
    tail_start = seq_len - min(args.keep_recent_k, args.cache_size // 4)
    print(f"PROBLEM: {prob['problem'][:300]}")
    print(f"GROUND TRUTH: {prob.get('ground_truth') or prob.get('answer')}")
    print(f"generated={len(generated)} tokens, prefill={prefill_len}, K={args.cache_size}, "
          f"retained_generated={sum(1 for p in kept if p >= prefill_len)}")
    print("legend: [tok] retained by score | <tok> retained by recency window | tok evicted")
    print("-" * 80)
    pieces = []
    for i, tid in enumerate(generated):
        pos = prefill_len + i
        t = tok.decode([tid])
        if pos >= tail_start:
            pieces.append(f"<{t}>")
        elif pos in kept:
            pieces.append(f"[{t}]")
        else:
            pieces.append(t)
    print("".join(pieces))

    if args.json_out:
        import json
        rows = []
        for i, tid in enumerate(generated):
            pos = prefill_len + i
            rows.append({
                "i": i, "pos": pos, "text": tok.decode([tid]),
                "score": scores_at_time[i] if i < len(scores_at_time) else None,
                "kept": pos in kept, "recency": pos >= tail_start,
            })
        Path(args.json_out).write_text(json.dumps({
            "method": args.method, "problem_idx": args.problem_idx,
            "problem": prob["problem"], "ground_truth": str(prob.get("ground_truth") or prob.get("answer")),
            "cache_size": args.cache_size, "keep_recent": args.keep_recent_k,
            "prefill_len": prefill_len, "generated": len(generated),
            "retained_generated": sum(1 for p in kept if p >= prefill_len),
            "tokens": rows,
        }, indent=1))
        print(f"wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
