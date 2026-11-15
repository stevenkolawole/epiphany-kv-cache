"""EpiKV for a batch of sequences, making exactly the decisions one KVSegHSEviction per sequence makes.

Each sequence keeps its own KVSegHSEviction (src/eviction.py, unmodified), which holds the state and makes every
compaction decision with its own code. This class changes only how the work is issued:
  - Per decode step, the hidden-state differences of both bands are computed for every sequence with the same
    (1, d) tensor shapes the original uses, then read back with ONE host transfer instead of two per sequence;
    the rolling z-scores and list updates are the original's Python arithmetic, in the same order.
  - At a compaction, each sequence's keep mask is captured from its own evict_past_key_values, and the cache is
    gathered one layer at a time and written back in place, so the old and the kept cache never coexist: the
    transient is one layer's kept copy instead of the whole kept cache.
  - Only the last query_sim_window banked hidden states stay as tensors; older bank entries become None. The
    policy reads the bank's length and its last window only (query_mode "qk"), and the window is always inside
    the recent tokens every compaction keeps, so decisions are unchanged and 16 KiB per cached token is freed.
Supported configuration: tau_mode "steps", band_mode "diff", no value reserve, and query_mode "qk" or no query.
The batch must share one cache length (every sequence keeps exactly the budget; asserted at each gather).
"""
from typing import List, Optional, Sequence

import torch

import eviction as E


def trim_bank(pol) -> None:
    """Replace all but the last query_sim_window banked hidden states with None (length preserved)."""
    bank = pol._hs_bank
    for i in range(max(0, len(bank) - pol.query_sim_window)):
        if bank[i] is None:
            continue
        bank[i] = None


class BatchedEpiKV:
    """Drive one KVSegHSEviction per sequence over a batched DynamicCache, compacting it in place."""

    def __init__(self, pols: Sequence):
        p = pols[0]
        if p.tau_mode != "steps":
            raise ValueError("BatchedEpiKV requires tau_mode 'steps'")
        if p.query_sim_weight > 0 and p.query_mode != "qk":
            raise ValueError("BatchedEpiKV requires query_mode 'qk' when query_sim_weight > 0")
        if p.band_mode != "diff" or p.value_reserve:
            raise ValueError("BatchedEpiKV requires band_mode 'diff' and value_reserve 0")
        self.pols: List = list(pols)
        for q in self.pols:
            trim_bank(q)
        self.last_masks: Optional[List[torch.Tensor]] = None

    def _check_due(self) -> bool:
        due = {q._since_compaction + 1 >= q.refresh_tau for q in self.pols}
        assert len(due) == 1, "sequences disagree on the compaction schedule"
        return due.pop()

    def step(self, cache, hidden_states) -> bool:
        """Account for one decode step; compact `cache` in place if the policy fires. Returns True on compaction."""
        self.last_masks = None
        if self._check_due():
            return self._evict(cache, hidden_states)
        p0 = self.pols[0]
        ha = hidden_states[p0.band_a_layer + 1][:, -1, :].detach()
        hb = hidden_states[p0.band_b_layer + 1][:, -1, :].detach()
        zero = torch.zeros((), dtype=ha.dtype, device=ha.device)
        diffs = []
        for b, q in enumerate(self.pols):
            for h, prev in ((ha, q._prev_hs_a), (hb, q._prev_hs_b)):
                # Same shapes and ops as eviction._hs_diff, so the bf16 result is the same number.
                diffs.append((h[b:b + 1] - prev).norm(dim=-1).mean() if prev is not None else zero)
        vals = torch.stack(diffs).tolist()
        bank_rows = ha.float() if p0.query_sim_weight > 0 else None
        for b, q in enumerate(self.pols):
            da = vals[2 * b] if q._prev_hs_a is not None else 0.0
            db = vals[2 * b + 1] if q._prev_hs_b is not None else 0.0
            q._prev_hs_a, q._prev_hs_b = ha[b:b + 1], hb[b:b + 1]
            q._buf_a.append(da)
            q._buf_b.append(db)
            q._step += 1
            if q.query_sim_weight > 0:
                q._hs_bank.append(bank_rows[b])
                q._pos_bank.append(q._prefill_len + q._step - 1)
                w = q.query_sim_window
                if len(q._hs_bank) > w:
                    q._hs_bank[-w - 1] = None
            z_a = E._rolling_z_score(q._buf_a, q.window)
            z_b = E._rolling_z_score(q._buf_b, q.window)
            q._scores.append(z_a - z_b)
            q._since_compaction += 1
        return False

    def _evict(self, cache, hidden_states) -> bool:
        masks: List[torch.Tensor] = []
        orig = E._gather_kept
        E._gather_kept = lambda pkv, m: (masks.append(m), pkv)[1]
        try:
            kv = [(layer.keys, layer.values) for layer in cache.layers]
            for b, q in enumerate(self.pols):
                q.evict_past_key_values(tuple((k[b:b + 1], v[b:b + 1]) for k, v in kv),
                                        tuple(h[b:b + 1] for h in hidden_states))
                if q.query_sim_weight > 0:
                    trim_bank(q)
            del kv
        finally:
            E._gather_kept = orig
        if not masks:
            return False
        assert len(masks) == len(self.pols), "sequences disagreed on whether to compact"
        idx = torch.stack([m.nonzero().squeeze(1) for m in masks])        # (B, kept); equal counts or stack fails
        g = idx[:, None, :, None]
        for layer in cache.layers:
            # One layer at a time, so the old layer is freed before the next kept copy is made.
            k = layer.keys
            layer.keys = k.gather(2, g.expand(-1, k.shape[1], -1, k.shape[3]))
            del k
            v = layer.values
            layer.values = v.gather(2, g.expand(-1, v.shape[1], -1, v.shape[3]))
            del v
        self.last_masks = masks
        return True
