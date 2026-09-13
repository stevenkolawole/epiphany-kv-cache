#!/usr/bin/env python3
"""
Run EpiKV-Seg inside vLLM and check it against the HuggingFace path.

This is the deliverable behind the paper's "drops into serving stacks" claim.
What we measured before was the SIGNAL's cost inside vLLM. This runs the
EVICTION inside vLLM: tau-amortised compaction on stock kernels, using the
planner in src/vllm_seg_evict.py and the two runner wrappers in
src/vllm_seg_patch.py. Every hook was located in the installed 0.11.2 source.

The check is accuracy agreement with the HF harness on the same problems at
the same budget under greedy decoding. Not bit-exact -- different kernels --
but the paired interval must contain zero. If it does not, the port is wrong.

    USE_TF=0 python3 scripts/vllm_seg_validate.py --cache_size 1024 --n 20
"""
import argparse
import faulthandler
import json
import math
import os
import sys
import time
from pathlib import Path

# The first run of this script hung silently for ten minutes with the GPU at
# zero and nothing in the log, and py-spy could not read the interpreter. A
# hang must never again be invisible: dump every thread's stack to stderr
# every four minutes, unconditionally.
_fh = int(os.environ.get("EPIKV_FH_INTERVAL", "0"))
if _fh > 0:
    faulthandler.dump_traceback_later(_fh, repeat=True, file=sys.stderr)

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "modal_infra"))

from src.vllm_seg_patch import (PhysicalLengths, SeqEvictState, apply_compaction,  # noqa: E402
                                compaction_plan, gather_compact_kv, patch_runner)
import benchmark as bm  # noqa: E402


def _find_runner(llm):
    # Located by scripts/vllm_find_runner.py on vLLM 0.11.2 with the
    # in-process engine. Try it directly; fall back to the BFS from the
    # signal-measurement harness if the layout moves again.
    try:
        r = llm.llm_engine.model_executor.driver_worker.worker.model_runner
        if hasattr(r, "input_batch") and hasattr(r, "kv_caches"):
            return r, "llm_engine.model_executor.driver_worker.worker.model_runner"
    except AttributeError:
        pass
    from vllm_poc import find_model_runner  # noqa: E402
    return find_model_runner(llm)


def _scheduler(llm):
    ec = llm.llm_engine.engine_core
    # vLLM 0.11.2 runs the engine core in a separate process by default
    # (SyncMPClient). The runner, scheduler and block pool then live in that
    # process and nothing here can reach them; the inspection script found
    # only a LoRA state object. The in-process engine (InprocClient) is
    # selected by VLLM_ENABLE_V1_MULTIPROCESSING=0 and is what the earlier
    # vLLM signal measurement used. Refuse to run without it rather than fail
    # deep inside a hook.
    if type(ec).__name__ != "InprocClient":
        raise SystemExit(
            f"engine_core is {type(ec).__name__}; need InprocClient. "
            "Set VLLM_ENABLE_V1_MULTIPROCESSING=0 before constructing LLM().")
    core = getattr(ec, "engine_core", ec)
    return core.scheduler


def _key_variance(kv_caches, block_ids, n_phys, block_size, layers=(10, 21)):
    """Per-position cached-key variance over head_dim, mean over the given
    layers and heads -- the same statistic KVSegHSEviction tiers segments by,
    read straight from the paged cache. Two layers keep it cheap."""
    idx = np.arange(n_phys)
    blk = torch.as_tensor(np.asarray(block_ids)[idx // block_size], device=kv_caches[0].device)
    slot = torch.as_tensor(idx % block_size, device=kv_caches[0].device)
    acc = None
    for l in layers:
        k = kv_caches[l][0, blk, slot].float()      # (P, H, D)
        v = k.var(dim=-1).mean(dim=-1)              # (P,)
        acc = v if acc is None else acc + v
    return (acc / len(layers)).cpu().numpy()


def _qk_relevance(kv_caches, layer, layer_idx, block_ids, n_phys, block_size, hs, positions):
    """Query relevance inside vLLM, the port of KVSegHSEviction._qk_relevance:
    the last w decode tokens' band-A hidden states (`hs`, (w, hidden), the
    input of `layer`) go through the layer's own input_layernorm and fused
    qkv projection (q part), are rotated at their logical `positions` by the
    layer's rotary module, and are dotted with the layer's cached keys of all
    n_phys physical slots (post-RoPE, gathered from the paged cache); causal
    mask, softmax per query and head, mean over queries and heads. Returns one
    value per physical slot. One (w x n_phys) matrix per compaction; the
    attention kernel is untouched."""
    attn = layer.self_attn
    dev = kv_caches[0].device
    idx = np.arange(n_phys)
    blk = torch.as_tensor(np.asarray(block_ids)[idx // block_size], device=dev)
    slot = torch.as_tensor(idx % block_size, device=dev)
    keys = kv_caches[layer_idx][0, blk, slot]                       # (P, kvh, d), post-RoPE
    w = hs.shape[0]
    x = layer.input_layernorm(hs.to(keys.dtype))
    qkv, _ = attn.qkv_proj(x)
    q, k, _v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    pos = torch.as_tensor(np.asarray(positions), device=dev, dtype=torch.long)
    q, _k = attn.rotary_emb(pos, q, k)
    nh, kvh, d = attn.num_heads, attn.num_kv_heads, attn.head_dim
    q = q.view(w, nh, d).transpose(0, 1).float()                    # (nh, w, d)
    kr = keys.repeat_interleave(nh // kvh, dim=1).transpose(0, 1).float()  # (nh, P, d)
    logits = torch.matmul(q, kr.transpose(-1, -2)) * attn.scaling    # (nh, w, P)
    s = torch.arange(n_phys, device=dev)
    qslot = n_phys - w + torch.arange(w, device=dev)
    allowed = s[None, :] <= qslot[:, None]
    logits = logits.masked_fill(~allowed[None], float("-inf"))
    rel = torch.softmax(logits, dim=-1).mean(dim=(0, 1))            # (P,)
    # One-shot diagnostic dump of the first relevance computation, for the
    # harness-vs-vLLM diff (EPIKV_REL_DUMP=<path.json>); same fields as
    # KVSegHSEviction._qk_relevance.
    dump = os.environ.get("EPIKV_REL_DUMP")
    if dump and not getattr(_qk_relevance, "_dumped", False):
        _qk_relevance._dumped = True
        with open(dump, "w") as f:
            json.dump({"impl": "vllm", "layer": int(layer_idx), "seq": int(n_phys), "w": int(w),
                       "query_pos": [int(p) for p in positions],
                       "hs_absmean": hs.float().abs().mean(dim=-1).tolist(),
                       "q_norm": q.float().norm(dim=-1).mean(dim=0).tolist(),
                       "k_absmean": keys.float().abs().mean(dim=(1, 2)).tolist()[:16],
                       # full key vectors at slots 2..5 for layers L-1, L, L+1, transposed to
                       # (kvh, 4, d) like the harness dump (layer-identity check)
                       "keys_2to5": {str(l): kv_caches[l][0, blk[2:6], slot[2:6]].float().transpose(0, 1).cpu().tolist()
                                     for l in (layer_idx - 1, layer_idx, layer_idx + 1)},
                       "rel": rel.float().tolist()}, f)
    return rel.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--cache_size", type=int, default=1024)
    ap.add_argument("--keep_recent", type=int, default=128)
    ap.add_argument("--tau", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--band_a", type=int, default=10)
    ap.add_argument("--band_b", type=int, default=21)
    ap.add_argument("--out", default=str(Path.home() / "vllm_seg_validate.json"))
    ap.add_argument("--gpu_util", type=float, default=0.6,
                    help="vLLM gpu_memory_utilization; lower it to co-locate a second engine")
    ap.add_argument("--query_mode", choices=["hs", "qk"], default="hs",
                    help="hs: rank by the hidden-state score (tiering only); qk: rank by query relevance "
                         "from the cached keys (the shipped configuration since 2026-09-12)")
    ap.add_argument("--query_layer", type=int, default=None, help="layer whose keys are scored (default band_a+1)")
    ap.add_argument("--query_window", type=int, default=8, help="number of most recent queries averaged")
    ap.add_argument("--fill", action="store_true", help="spend the budget the tier caps leave unspent")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm_poc import EpiKVScoreTracker, get_decoder_layers

    # enable_prefix_caching=False is not optional here. With it on (the 0.11.2
    # default) the scheduler hashes every full block after each step and may
    # share or reclaim it across requests; compaction frees and rewrites
    # blocks underneath that bookkeeping. Eviction and prefix sharing are
    # incompatible by construction, and the first bisect hung on the step
    # right after a clean compaction, which is exactly where that collides.
    llm = LLM(model=args.model, enforce_eager=True, gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_new_tokens + 1024, dtype="bfloat16",
              max_num_seqs=1, enable_prefix_caching=False)
    runner, path = _find_runner(llm)
    sched = _scheduler(llm)
    mgr = sched.kv_cache_manager.coordinator.single_type_managers[0]
    pool = sched.kv_cache_manager.block_pool
    block_size = mgr.block_size
    layers = get_decoder_layers(runner.model)
    hidden = runner.model.config.hidden_size
    # The tracker hands each request a row from a fixed free list and never
    # gives it back; with max_reqs=4 the fourth problem crashed on an empty
    # pool. Rows are recycled below after every generate() call.
    tracker = EpiKVScoreTracker(runner.input_batch, hidden, runner.device,
                                max_reqs=64, score_cap=args.max_new_tokens + 1024)
    layers[args.band_a].register_forward_hook(tracker.make_hook(0))
    layers[args.band_b].register_forward_hook(tracker.make_hook(1))
    pl = PhysicalLengths(verbose=True)
    patch_runner(runner, pl)
    print(f"[vllm-seg] runner at {path}; block_size={block_size}; hooks on {args.band_a},{args.band_b}",
          flush=True)

    states = {}
    compactions = []

    orig_exec = runner.execute_model

    def execute_model_inner(scheduler_output, *a, **kw):
        out = orig_exec(scheduler_output, *a, **kw)
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
                states[rid] = st
            # Logical length as the scheduler sees it after this step. Append
            # exactly the NEW logical positions. The first version padded the
            # physical list until its length equalled L, which after a
            # compaction silently regrew it from 342 back to the logical
            # length with fabricated positions; the second compaction then saw
            # P == L and inconsistent arrays.
            L = req.num_computed_tokens
            if not hasattr(st, "logical_len"):
                st.logical_len = len(st.alive_logical)
            if L > st.logical_len:
                st.alive_logical.extend(range(st.logical_len, L))
                st.logical_len = L
            # Scores: tracker row holds one score per decode token, logical order.
            row = tracker._row(rid)
            n_dec = L - st.prefill_len
            if n_dec > 0:
                st.scores = tracker.scores[row, :n_dec].cpu().tolist()
            st.steps_since_refresh += 1
            P = len(st.alive_logical)
            if st.steps_since_refresh < args.tau or P <= args.cache_size:
                continue
            # --- tau boundary and over budget: compact ---
            # Phase prints: the first run of this path hung silently. The last
            # phase printed before a hang is the phase that hangs.
            print(f"[compact] {rid} L={L} P={P} phase=plan", flush=True)
            old_blocks = list(mgr.req_to_blocks[rid])
            old_ids = [b.block_id for b in old_blocks]
            key_stat = _key_variance(runner.kv_caches, old_ids, P, block_size,
                                     layers=(args.band_a, args.band_b))
            override = None
            if args.query_mode == "qk" and n_dec > 0:
                qL = args.query_layer if args.query_layer is not None else args.band_a + 1
                w = min(args.query_window, n_dec, tracker.HBANK)
                trow = tracker._row(rid)
                ns = int(tracker.nscore[trow])
                bank_idx = torch.tensor([(ns - w + j) % tracker.HBANK for j in range(w)],
                                        device=tracker.hbank.device)
                hs_w = tracker.hbank[trow].index_select(0, bank_idx)   # (w, hidden), chronological
                rel = _qk_relevance(runner.kv_caches, layers[qL], qL, old_ids, P, block_size,
                                    hs_w, st.alive_logical[-w:])
                dec = rel[st.prefill_len:]
                override = (dec - dec.mean()) / (dec.std() + 1e-6)
                # diagnostic: record the token ids behind the first dumped relevance vector
                _dump = os.environ.get("EPIKV_REL_DUMP")
                if _dump and os.path.exists(_dump) and not getattr(main, "_tok_dumped", False):
                    main._tok_dumped = True
                    _d = json.load(open(_dump))
                    _d["token_ids"] = [int(t) for t in ib.token_ids_cpu[i, :P]]
                    _d["prefill_len"] = int(st.prefill_len)
                    json.dump(_d, open(_dump, "w"))
            keep = compaction_plan(st, key_stat[st.prefill_len:], args.cache_size,
                                   args.keep_recent, block_size,
                                   scores_override=override, fill_budget=args.fill)
            if keep is None:
                st.steps_since_refresh = 0
                continue
            new_n = math.ceil(len(keep) / block_size)
            print(f"[compact] {rid} keep={len(keep)} new_blocks={new_n} old_blocks={len(old_blocks)} "
                  f"pool_free={pool.get_num_free_blocks()} phase=alloc", flush=True)
            new_blocks = pool.get_new_blocks(new_n)
            new_ids = [b.block_id for b in new_blocks]
            print(f"[compact] {rid} phase=gather", flush=True)
            gather_compact_kv(runner.kv_caches, old_ids, new_ids, keep, block_size)
            torch.cuda.synchronize()
            # Pack the runner's token-id buffer the same way the cache was
            # packed: `keep` indexes the current PHYSICAL layout. Only the first
            # P entries are physical before this compaction (either never
            # compacted, so physical == logical, or kept packed by wrapA).
            kept_ids = ib.token_ids_cpu[i, keep].copy()
            ib.token_ids_cpu[i, :len(kept_ids)] = kept_ids
            P_new = apply_compaction(st, keep)
            print(f"[compact] {rid} P_new={P_new} phase=rewrite", flush=True)
            # Scheduler side: hand it the physical block list; it regrows to
            # logical size on the next allocate_slots (documented over-allocation).
            mgr.req_to_blocks[rid] = new_blocks
            pool.free_blocks(old_blocks)
            # Runner side: rewrite the row, AND the per-request block list the
            # runner extends every step (gpu_model_runner l.832) and rebuilds
            # rows from on reorder (gpu_input_batch l.343). Leaving it stale
            # keeps the freed blocks referenced.
            bt = ib.block_table[0] if hasattr(ib.block_table, "__getitem__") else ib.block_table
            bt.add_row(new_ids, i)
            rs = getattr(runner, "requests", {}).get(rid)
            if rs is not None:
                rs.block_ids = (list(new_ids),)
            else:
                print(f"[compact] WARNING no runner.requests[{rid}] to update", flush=True)
            pl.phys[rid] = P_new
            spin["since_compact"] = 0
            print(f"[compact] {rid} phase=done", flush=True)
            compactions.append({"req": rid, "step_logical": L, "from": P, "to": P_new,
                                "freed_blocks": len(old_blocks) - new_n})
            st.steps_since_refresh = 0
        return out

    spin = {"since_compact": -1, "last_num_tokens": {}, "stall": 0}

    def execute_model(scheduler_output, *a, **kw):
        # Outer instrumentation. The first post-compaction run livelocked:
        # 2.3 million steps in two minutes, GPU idle, no token emitted, the
        # scheduler re-issuing the same step forever because the sampled token
        # was being discarded. Two defences: print the request's logical
        # bookkeeping for the five steps after any compaction, and abort if
        # num_tokens fails to advance for 50 consecutive steps.
        r = execute_model_inner(scheduler_output, *a, **kw)
        ib = runner.input_batch
        stalled = True
        for i in range(ib.num_reqs):
            rid = ib.req_ids[i]
            req = sched.requests.get(rid)
            rs = getattr(runner, "requests", {}).get(rid)
            nt = req.num_tokens if req is not None else -1
            if nt != spin["last_num_tokens"].get(rid):
                stalled = False
            spin["last_num_tokens"][rid] = nt
            if rid in pl.phys and spin["since_compact"] < 5:
                print(f"[state] rid={rid} sched.num_tokens={nt} "
                      f"sched.num_computed={getattr(req, 'num_computed_tokens', -1)} "
                      f"runner.num_tokens={getattr(rs, 'num_tokens', -1)} "
                      f"runner.num_computed={getattr(rs, 'num_computed_tokens', -1)} "
                      f"cpu_buf={int(ib.num_computed_tokens_cpu[i])} phys={pl.phys.get(rid)} "
                      f"row_blocks={int(ib.block_table[0].num_blocks_per_row[i])} "
                      f"sched_blocks={len(mgr.req_to_blocks.get(rid, []))}", flush=True)
        if spin["since_compact"] >= 0:
            spin["since_compact"] += 1
        # Batch- and scheduler-level view for the ten steps after a compaction.
        # The previous run showed the request leaving the runner's batch after
        # step two while the scheduler kept it, and the engine then spun with
        # num_reqs == 0 -- which the old guard treated as "not stalled".
        if 0 <= spin["since_compact"] <= 10:
            running = [getattr(q, "request_id", "?") for q in getattr(sched, "running", [])]
            waiting = len(getattr(sched, "waiting", []))
            fin = list(getattr(sched, "finished_req_ids", []))[:4]
            print(f"[batch] step+{spin['since_compact']} num_reqs={ib.num_reqs} "
                  f"req_ids={list(ib.req_ids)[:ib.num_reqs]} sched.running={running} "
                  f"sched.waiting={waiting} finished={fin} "
                  f"sched_out_finished={list(getattr(scheduler_output, 'finished_req_ids', []))[:4]}",
                  flush=True)
        if ib.num_reqs == 0:
            stalled = True          # an empty batch that repeats IS the failure mode
        spin["stall"] = spin["stall"] + 1 if stalled else 0
        if spin["stall"] >= 50:
            raise SystemExit("[step] ABORT: 50 consecutive steps with no progress "
                             "(num_tokens flat or empty batch) after compaction; see [batch]/[state] lines")
        return r

    runner.execute_model = execute_model

    tok = llm.get_tokenizer()
    problems = bm.load_problems(args.dataset, n_samples=args.n, start_idx=args.start_idx)
    sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0)
    records = []
    t0 = time.time()
    for i, prob in enumerate(problems):
        prompt = tok.apply_chat_template([{"role": "user", "content": prob["problem"]}],
                                         tokenize=False, add_generation_prompt=True)
        t1 = time.time()
        out = llm.generate([prompt], sp, use_tqdm=False)[0]
        text = out.outputs[0].text
        # Same extraction and matching as the HF harness, so agreement is
        # about the eviction path and not about the grader.
        pred = bm.extract_boxed(text)
        gt = prob.get("ground_truth") or prob.get("answer")
        correct = bool(bm.answers_match(pred, gt)) if pred is not None else False
        n_gen = len(out.outputs[0].token_ids)
        records.append({"problem": prob["problem"][:80], "ground_truth": gt, "pred": pred,
                        "correct": correct, "n_tokens_generated": n_gen,
                        "wall_s": round(time.time() - t1, 2),
                        "compactions": [c for c in compactions if c["req"] == out.request_id]})
        print(f"[vllm-seg] {i:3d} gen={n_gen:5d} correct={correct} "
              f"compactions={len(records[-1]['compactions'])} ({records[-1]['wall_s']}s)", flush=True)
        # per-request state is dead after the request finishes
        states.pop(out.request_id, None)
        pl.phys.pop(out.request_id, None)
        pl.logical.pop(out.request_id, None)
        spin["last_num_tokens"].pop(out.request_id, None)
        spin["since_compact"] = -1
        # Recycle the tracker's row pool; one request in flight at a time.
        tracker.reset()

    acc = [r["correct"] for r in records if r["correct"] is not None]
    payload = {"meta": {"model": args.model, "dataset": args.dataset, "n": len(records),
                        "cache_size": args.cache_size, "keep_recent": args.keep_recent,
                        "tau": args.tau, "max_new_tokens": args.max_new_tokens,
                        "band": [args.band_a, args.band_b], "block_size": block_size,
                        "query_mode": args.query_mode, "query_window": args.query_window,
                        "fill": args.fill,
                        "vllm_runner_path": path, "wall_s": round(time.time() - t0)},
               "accuracy": (sum(acc) / len(acc)) if acc else None,
               "per_problem": records}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[vllm-seg] accuracy={payload['accuracy']} over {len(acc)}; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
