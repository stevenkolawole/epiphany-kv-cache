#!/usr/bin/env python3
"""Decode throughput of EpiKV-Seg eviction inside vLLM at fixed concurrency.

    python3 scripts/vllm_seg_throughput.py --batch 8 --cache_size 1024 --gen 2048 --out X.json
    python3 scripts/vllm_seg_throughput.py --batch 8 --cache_size 1000000 ...   # no eviction

Same engine, hooks and compaction path as vllm_seg_validate.py, but
`max_num_seqs=batch` and all prompts of a batch are submitted together, with
`ignore_eos` and `min_tokens` so every request decodes exactly `gen` tokens.
Reported: decode tokens per second for the batch, wall time, compactions, and
the minimum number of free KV blocks seen (peak cache occupancy).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "modal_infra"))

from src.vllm_seg_patch import (PhysicalLengths, SeqEvictState, apply_compaction,  # noqa: E402
                                compaction_plan, gather_compact_kv, patch_runner)
import benchmark as bm  # noqa: E402
from vllm_seg_validate import _find_runner, _scheduler, _key_variance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--cache_size", type=int, default=1024)
    ap.add_argument("--keep_recent", type=int, default=128)
    ap.add_argument("--tau", type=int, default=128)
    ap.add_argument("--gen", type=int, default=2048, help="exact decode tokens per request")
    ap.add_argument("--band_a", type=int, default=10)
    ap.add_argument("--band_b", type=int, default=21)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm_poc import EpiKVScoreTracker, get_decoder_layers

    llm = LLM(model=args.model, enforce_eager=True, gpu_memory_utilization=args.gpu_util,
              max_model_len=args.gen + 1024, dtype="bfloat16",
              max_num_seqs=args.batch, enable_prefix_caching=False)
    runner, path = _find_runner(llm)
    sched = _scheduler(llm)
    mgr = sched.kv_cache_manager.coordinator.single_type_managers[0]
    pool = sched.kv_cache_manager.block_pool
    block_size = mgr.block_size
    evict = args.cache_size < 10 ** 6
    tracker = None
    if evict:
        layers = get_decoder_layers(runner.model)
        hidden = runner.model.config.hidden_size
        tracker = EpiKVScoreTracker(runner.input_batch, hidden, runner.device,
                                    max_reqs=max(64, 4 * args.batch), score_cap=args.gen + 1024)
        layers[args.band_a].register_forward_hook(tracker.make_hook(0))
        layers[args.band_b].register_forward_hook(tracker.make_hook(1))
        pl = PhysicalLengths(verbose=False)
        patch_runner(runner, pl)

        # Scheduler side: without this the allocator keeps reserving blocks for
        # the LOGICAL length after a compaction (it regrows the request to L
        # on the next allocate_slots), so the pool never sees the saving. Both
        # entry points take "total tokens that need a slot"; subtract the
        # logical-minus-physical gap for compacted requests so the pool holds
        # ceil(physical / block_size) blocks, which is exactly what the packed
        # slot mapping (wrapA) writes into.
        _orig_need = mgr.get_num_blocks_to_allocate
        _orig_alloc = mgr.allocate_new_blocks

        def _gap(rid):
            if rid not in pl.phys:
                return 0
            return max(pl.logical.get(rid, 0) - pl.phys.get(rid, 0), 0)

        def _need(request_id, num_tokens, new_computed_blocks):
            return _orig_need(request_id, num_tokens - _gap(request_id), new_computed_blocks)

        def _alloc(request_id, num_tokens):
            return _orig_alloc(request_id, num_tokens - _gap(request_id))

        mgr.get_num_blocks_to_allocate = _need
        mgr.allocate_new_blocks = _alloc

    states, compactions = {}, []
    stats = {"min_free_blocks": pool.get_num_free_blocks(), "steps": 0}
    orig_exec = runner.execute_model

    def execute_model(scheduler_output, *a, **kw):
        out = orig_exec(scheduler_output, *a, **kw)
        stats["steps"] += 1
        stats["min_free_blocks"] = min(stats["min_free_blocks"], pool.get_num_free_blocks())
        if not evict:
            return out
        ib = runner.input_batch
        for i in range(ib.num_reqs):
            rid = ib.req_ids[i]
            req = sched.requests.get(rid)
            if req is None:
                continue
            st = states.get(rid)
            if st is None:
                st = SeqEvictState(prefill_len=req.num_prompt_tokens)
                st.alive_logical = list(range(req.num_computed_tokens))
                st.logical_len = len(st.alive_logical)
                states[rid] = st
            L = req.num_computed_tokens
            if L > st.logical_len:
                st.alive_logical.extend(range(st.logical_len, L))
                st.logical_len = L
            row = tracker._row(rid)
            n_dec = L - st.prefill_len
            if n_dec > 0:
                st.scores = tracker.scores[row, :n_dec].cpu().tolist()
            st.steps_since_refresh += 1
            P = len(st.alive_logical)
            if st.steps_since_refresh < args.tau or P <= args.cache_size:
                continue
            old_blocks = list(mgr.req_to_blocks[rid])
            old_ids = [b.block_id for b in old_blocks]
            key_stat = _key_variance(runner.kv_caches, old_ids, P, block_size,
                                     layers=(args.band_a, args.band_b))
            keep = compaction_plan(st, key_stat[st.prefill_len:], args.cache_size,
                                   args.keep_recent, block_size)
            if keep is None:
                st.steps_since_refresh = 0
                continue
            new_n = math.ceil(len(keep) / block_size)
            # Compaction gathers into fresh blocks before freeing the old ones,
            # so it needs new_n free blocks transiently. When the pool is
            # exhausted (hundreds of resident requests) defer to the next step
            # rather than raise; vLLM's own preemption handles the pressure.
            if pool.get_num_free_blocks() < new_n + 1:
                continue
            new_blocks = pool.get_new_blocks(new_n)
            new_ids = [b.block_id for b in new_blocks]
            gather_compact_kv(runner.kv_caches, old_ids, new_ids, keep, block_size)
            torch.cuda.synchronize()
            kept_ids = ib.token_ids_cpu[i, keep].copy()
            ib.token_ids_cpu[i, :len(kept_ids)] = kept_ids
            P_new = apply_compaction(st, keep)
            mgr.req_to_blocks[rid] = new_blocks
            pool.free_blocks(old_blocks)
            bt = ib.block_table[0] if hasattr(ib.block_table, "__getitem__") else ib.block_table
            bt.add_row(new_ids, i)
            rs = getattr(runner, "requests", {}).get(rid)
            if rs is not None:
                rs.block_ids = (list(new_ids),)
            pl.phys[rid] = P_new
            compactions.append({"req": rid, "from": P, "to": P_new})
            st.steps_since_refresh = 0
        return out

    runner.execute_model = execute_model

    tok = llm.get_tokenizer()
    problems = bm.load_problems(args.dataset, n_samples=args.batch, start_idx=args.start_idx)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p["problem"]}],
                                       tokenize=False, add_generation_prompt=True) for p in problems]
    sp = SamplingParams(max_tokens=args.gen, min_tokens=args.gen, ignore_eos=True, temperature=0)
    # warm-up: one short batch so engine start-up is not timed
    llm.generate(prompts[:1], SamplingParams(max_tokens=16, min_tokens=16, ignore_eos=True), use_tqdm=False)
    states.clear(); compactions.clear()
    if tracker is not None:
        tracker.reset(); pl.phys.clear(); pl.logical.clear()
    total_blocks = pool.get_num_free_blocks()
    stats["min_free_blocks"] = total_blocks
    torch.cuda.synchronize()
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    wall = time.time() - t0
    n_gen = sum(len(o.outputs[0].token_ids) for o in outs)
    peak_blocks = total_blocks - stats["min_free_blocks"]
    payload = {"meta": vars(args) | {"block_size": block_size, "runner": path},
               "wall_s": round(wall, 2), "decode_tokens": n_gen,
               "tokens_per_s": round(n_gen / wall, 1),
               "peak_kv_blocks": peak_blocks, "peak_kv_tokens": peak_blocks * block_size,
               "peak_kv_tokens_per_req": peak_blocks * block_size / args.batch,
               "compactions": len(compactions), "steps": stats["steps"]}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "meta"}), flush=True)


if __name__ == "__main__":
    main()
