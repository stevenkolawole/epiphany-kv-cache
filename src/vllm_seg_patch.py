"""
EpiKV-Seg inside vLLM 0.11.2: tau-amortised compaction on stock kernels.

Read the docstring of vllm_seg_evict.py first for why compaction and not a
kernel. This module is the part that touches vLLM, and every hook below was
located in the installed 0.11.2 source rather than assumed:

  vllm/v1/worker/gpu_model_runner.py
    l.1198  positions_np = num_computed_tokens_cpu[req_indices] + arange
    l.1275  block_table.compute_slot_mapping(req_indices, positions_np)
    l.1309  seq_lens = num_computed_tokens_cpu + num_scheduled_tokens
  vllm/v1/worker/block_table.py
    add_row(block_ids, row_idx)         -- full row rewrite, what compaction needs
  vllm/v1/core/single_type_kv_cache_manager.py
    remove_skipped_blocks()             -- the native mid-sequence free pattern
  vllm/v1/attention/backends/flash_attn.py
    get_kv_cache_shape -> (2, num_blocks, block_size, num_kv_heads, head_size)

The one real problem, and its one-line fix
-------------------------------------------
vLLM derives THREE things from a single counter, num_computed_tokens:
  (a) the slot each new token is written to      (slot mapping, l.1275)
  (b) how many cached tokens attention sees        (seq_lens, l.1309)
  (c) the RoPE position of each new token          (positions, l.1198)

After compacting a request from logical length L down to physical length P,
(a) and (b) must see P -- the cache really is P tokens long now -- but (c)
must keep counting from L, or every later token gets the wrong rotation.
vLLM's sliding-window path never hits this because it only drops a prefix and
the kernel is told a window; the counter stays logical and the mapping holds.

So: set num_computed_tokens_cpu[r] = P, and carry rope_offset[r] = L - P.
Slot mapping is computed from the un-offset positions (l.1275), then the
offset is added to the positions buffer before the model forward. One
subtraction at compaction time, one addition per step. Nothing else in the
runner needs to know.

RoPE on the cached keys is already applied at write time, so moving a key to a
new block does not disturb its encoding. That is why gather-compaction is
correct here and why the HuggingFace path needed the _seen_tokens fixup.

What compaction does, per request, at a tau boundary
-----------------------------------------------------
  1. scores from the layer hooks (EpiKVScoreTracker, modal_infra/vllm_poc.py)
     and per-position key variance read from the paged cache
  2. keep = plan_segment_eviction(...)            pure, tested, no vLLM
  3. new_blocks = pool.allocate(ceil(P / block_size))
  4. for each layer: kv[:, new_blk, new_slot] = kv[:, old_blk, old_slot]
     -- one index_select per layer over (2, N, B, H, D); no custom kernel
  5. block_table.add_row(new_blocks, row); free old blocks to the pool
  6. num_computed_tokens[r] = P;  rope_offset[r] += L - P

Cost: a gather of P tokens x layers every tau=128 steps. ThinKV avoids even
that with a PagedAttention fork; we pay it to stay on the stock kernel.

Validation before any number is reported
----------------------------------------
Same problems, same budget, same greedy decoding, HF path vs this path. Not
bit-exact (different kernels, different reduction order) but accuracy on the
matched set must agree within the paired interval. If it does not, the port is
wrong, not the method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .vllm_seg_evict import plan_segment_eviction


@dataclass
class SeqEvictState:
    """Per-request state the runner does not track: logical vs physical length."""

    prefill_len: int
    rope_offset: int = 0            # logical - physical, grows with each compaction
    scores: List[float] = field(default_factory=list)   # per decode position, logical order
    alive_logical: List[int] = field(default_factory=list)  # logical pos of each physical slot
    steps_since_refresh: int = 0


def gather_compact_kv(
    kv_caches: List[torch.Tensor],
    old_block_ids: List[int],
    new_block_ids: List[int],
    keep_physical: np.ndarray,
    block_size: int,
) -> None:
    """Move retained (block, slot) entries into fresh blocks, densely, in order.

    kv_caches[layer] is (2, num_blocks, block_size, H, D). Retained physical
    positions are mapped old (blk, slot) -> new (blk, slot) by rank. This is an
    index_select per layer: no kernel, no fusion, and it runs once per tau
    steps rather than once per token.
    """
    keep_physical = np.asarray(keep_physical, dtype=np.int64)
    src_blk = np.asarray(old_block_ids)[keep_physical // block_size]
    src_slot = keep_physical % block_size
    dst_idx = np.arange(len(keep_physical))
    dst_blk = np.asarray(new_block_ids)[dst_idx // block_size]
    dst_slot = dst_idx % block_size

    for kv in kv_caches:
        dev = kv.device
        sb = torch.as_tensor(src_blk, device=dev)
        ss = torch.as_tensor(src_slot, device=dev)
        db = torch.as_tensor(dst_blk, device=dev)
        ds = torch.as_tensor(dst_slot, device=dev)
        # Read all sources first, then write: src and dst blocks are disjoint
        # by construction (new blocks come from the free pool), so no aliasing.
        gathered = kv[:, sb, ss]            # (2, P, H, D)
        kv[:, db, ds] = gathered


def compaction_plan(
    state: SeqEvictState,
    key_stat: np.ndarray,
    cache_size: int,
    keep_recent: int,
    block_size: int,
) -> Optional[np.ndarray]:
    """Decide the physical positions that survive, or None if nothing to do.

    Scores are held in logical order; the planner reasons in logical positions;
    we translate back to physical slots via alive_logical. Prefill is always
    kept, so its physical and logical positions coincide.
    """
    n_phys = len(state.alive_logical)
    if n_phys <= cache_size:
        return None
    # Planner wants scores per decode position for the *current physical* cache.
    phys_scores = np.array(
        [state.scores[lp - state.prefill_len] for lp in state.alive_logical[state.prefill_len:]],
        dtype=np.float32,
    )
    plan = plan_segment_eviction(
        scores=phys_scores,
        key_stat=key_stat,
        prefill_len=state.prefill_len,
        cache_size=cache_size,
        keep_recent=keep_recent,
        fill_budget=False,          # Section 18: filling makes it worse
    )
    return plan.keep                # physical positions, sorted


def apply_compaction(
    state: SeqEvictState,
    keep_physical: np.ndarray,
) -> int:
    """Update per-request bookkeeping after the bytes have moved. Returns P."""
    logical_before = len(state.alive_logical)
    state.alive_logical = [state.alive_logical[i] for i in keep_physical]
    P = len(state.alive_logical)
    state.rope_offset += logical_before - P
    return P


# ----------------------------------------------------------------------------
# Runner patches. Applied once, after LLM() construction, to the live runner.
#
# Two pure wrappers, no line-level surgery:
#   A. before _prepare_inputs: present the PHYSICAL length to the runner for
#      any compacted request, so slot mapping (l.1275) and seq_lens (l.1309)
#      address the cache as it actually is. The scheduler overwrites this
#      array every step with the logical value, so A re-applies every step.
#   B. around model.forward: positions derived in A are physical too, and RoPE
#      must be logical, so add (logical - physical) per token before the model
#      reads them. The per-token request index is captured in A.
#
# The scheduler is left entirely logical. Its one visible side effect is that
# allocate_slots keeps req_to_blocks at logical size, so it over-allocates tail
# blocks after a compaction. Harmless for correctness -- the runner never writes
# past the physical length -- and it means MEMORY numbers from this path are not
# yet meaningful. Accuracy validation first; the scheduler arithmetic patch for
# memory accounting is a documented follow-up.
# ----------------------------------------------------------------------------

class PhysicalLengths:
    """Per-request physical (compacted) length; absent means uncompacted."""

    def __init__(self, verbose: bool = False):
        self.phys: Dict[str, int] = {}
        self.logical: Dict[str, int] = {}
        self.restarted: set = set()   # requests preempted after a compaction
        self._step_req_indices: Optional[np.ndarray] = None
        self._step_offsets: Optional[np.ndarray] = None
        self.verbose = verbose


def patch_runner(runner, pl: PhysicalLengths) -> None:
    orig_prepare = runner._prepare_inputs
    ib = runner.input_batch

    def _prepare_inputs(scheduler_output, *a, **kw):
        # Runs after _update_states (which wrote the new token id at the
        # LOGICAL index and reset num_computed_tokens_cpu to logical) and
        # before everything downstream reads positions. Four consumers derive
        # from num_computed_tokens_cpu, and they want different things:
        #   slot mapping     -> physical   (where the new KV entry is written)
        #   seq_lens         -> physical   (how much cache attention reads)
        #   token gathering  -> physical, IF token_ids_cpu is packed physically
        #   discard mask     -> compares seq_len to req.num_tokens; must not fire
        # RoPE wants logical and is handled in wrapB.
        num_reqs = ib.num_reqs
        tokens = scheduler_output.num_scheduled_tokens
        counts = np.array([tokens[ib.req_ids[i]] for i in range(num_reqs)], dtype=np.int64)
        req_indices = np.repeat(np.arange(num_reqs), counts)
        offs = np.zeros(num_reqs, dtype=np.int64)
        restore = []
        for i in range(num_reqs):
            rid = ib.req_ids[i]
            if rid not in pl.phys:
                continue
            logical = int(ib.num_computed_tokens_cpu[i])
            phys = pl.phys[rid]
            n = int(counts[i])
            # A preempted request comes back as a fresh prefill: its logical
            # count restarts below our physical count and it is scheduled in
            # a multi-token chunk. Its cache is recomputed from scratch, so the
            # physical layout is logical again; drop our bookkeeping and let
            # it run uncompacted from here (the caller sees rid leave pl.phys).
            if logical < phys or n > 1:
                del pl.phys[rid]
                pl.logical.pop(rid, None)
                pl.restarted.add(rid)
                continue
            pl.logical[rid] = logical
            offs[i] = logical - phys
            # Keep token_ids_cpu packed: the token(s) _update_states just wrote
            # at the logical index must also sit at the physical index, since
            # positions_np (physical) is what gathers input_ids.
            ib.token_ids_cpu[i, phys:phys + n] = ib.token_ids_cpu[i, logical:logical + n]
            ib.num_computed_tokens_cpu[i] = phys                 # A
            restore.append(i)
            # The new token is now physically appended.
            pl.phys[rid] = phys + n
        pl._step_req_indices = req_indices
        pl._step_offsets = offs
        if pl.verbose and offs.any():
            print(f"[wrapA] enter phys(after)={dict(pl.phys)} offs={offs.tolist()}", flush=True)
        out = orig_prepare(scheduler_output, *a, **kw)
        # The discard mask inside _prepare_inputs compares seq_len (physical)
        # against requests[rid].num_tokens (logical, and a read-only property),
        # so every compacted request is flagged "not ready to sample" and its
        # logits are dropped -- which is exactly the livelock seen earlier. The
        # mask's result lives in a buffer the sampler reads later in the same
        # step, so remove our rows from it here rather than fight the property.
        if restore and getattr(runner, "num_discarded_requests", 0) > 0:
            buf = runner.discard_request_indices
            cur = buf.np[: runner.num_discarded_requests]
            ours = set(restore)
            kept = np.array([x for x in cur if int(x) not in ours], dtype=cur.dtype)
            runner.num_discarded_requests = len(kept)
            buf.np[: len(kept)] = kept
            buf.copy_to_gpu(len(kept))
            if pl.verbose:
                print(f"[wrapA] discard mask: removed {len(cur) - len(kept)} of ours, "
                      f"{len(kept)} remain", flush=True)
        if pl.verbose and offs.any():
            print("[wrapA] exit", flush=True)
        return out

    runner._prepare_inputs = _prepare_inputs

    model = runner.model
    orig_forward = model.forward

    def forward(*args, **kw):                                      # B
        pos = kw.get("positions", args[1] if len(args) > 1 else None)
        active = pos is not None and pl._step_offsets is not None and pl._step_offsets.any()
        if active:
            n = pos.shape[0]
            off = torch.as_tensor(pl._step_offsets[pl._step_req_indices[:n]],
                                  device=pos.device, dtype=pos.dtype)
            pos = pos + off
            if "positions" in kw:
                kw["positions"] = pos
            else:
                args = (args[0], pos) + tuple(args[2:])
            if pl.verbose:
                print(f"[wrapB] enter n={n} pos0={int(pos[0])}", flush=True)
        out = orig_forward(*args, **kw)
        if active and pl.verbose:
            print("[wrapB] exit", flush=True)
        return out

    model.forward = forward
