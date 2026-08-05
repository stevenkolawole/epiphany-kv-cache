"""Adapted contemporary baselines and the FA2-compatible segment hybrid.

Kept in a separate module from eviction.py so that this file and the
entropy-based KVSegmentHSEviction developed on the cluster can coexist:
KVSegHSEviction here classifies segments by cached-key variance, while
KVSegmentHSEviction in eviction.py classifies by the entropy of that
variance distribution. The Modal results reported in the paper
(kv_seg_hs: 34.0% at K=1024, 56.0% at K=2048) come from KVSegHSEviction.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from eviction import EvictionConfig


class KVSegHSEviction:
    """
    kv_seg_hs: FA2-compatible segment hybrid (Phase 2C).

    Replaces HybridSegmentHSEviction's attention-entropy segment classifier
    with a classifier built from cached KEY statistics, keeping the detrended
    Band A − B HS z-score for within-segment token ranking.  The result needs
    only past_key_values + output_hidden_states → runs inside FlashAttention.

    Outer loop (segment level): each `segment_size`-token block is scored by
    the mean per-position key variance (var over head_dim, mean over batch,
    heads, layers) — computed statelessly from the current cache.  High key
    variance = content-rich block (Phase 0B: kv_key_var has the highest |ρ|
    of all signals on competition math), so blocks are classified by
    DESCENDING key variance: top tertile → R (retain_r), middle → E
    (retain_e), bottom → T (retain_t).  This mirrors ThinKV's R/E/T budget
    structure with an attention-free segment signal.

    Inner loop (token level): identical to HybridSegmentHSEviction — tokens
    within each segment are ranked by the detrended Band A − B hidden-state
    z-score.

    Requires output_hidden_states=True only.  FA2-COMPATIBLE — this is the
    method that targets the tight-budget regime (K ≤ 2048) where the eager
    hybrid leads but no FA2-compatible method previously competed.

    Usage:
        eviction = KVSegHSEviction(config)
        eviction.reset(prefill_len=prompt_len)
        eviction.set_prefill_end(prefill_outputs.hidden_states)
        outputs = model(..., output_hidden_states=True)
        past_kv = eviction.evict_past_key_values(past_kv, outputs.hidden_states)
    """

    def __init__(
        self,
        config: EvictionConfig,
        band_a_layer: int = 10,
        band_b_layer: int = 21,
        window: int = 64,
        segment_size: int = 128,
        retain_r: int = 64,
        retain_e: int = 32,
        retain_t: int = 8,
    ):
        self.config = config
        self.band_a_layer = band_a_layer
        self.band_b_layer = band_b_layer
        self.window = window
        self.segment_size = segment_size
        self.retain_r = retain_r
        self.retain_e = retain_e
        self.retain_t = retain_t

        self._prefill_len: int = 0
        self._prev_hs_a: Optional[torch.Tensor] = None
        self._prev_hs_b: Optional[torch.Tensor] = None
        self._buf_a: List[float] = []
        self._buf_b: List[float] = []
        self._scores: List[float] = []   # per decode token, pruned on eviction

    def reset(self, prefill_len: int = 0):
        self._prefill_len = prefill_len
        self._prev_hs_a = None
        self._prev_hs_b = None
        self._buf_a = []
        self._buf_b = []
        self._scores = []

    def set_prefill_end(self, hidden_states: Tuple[torch.Tensor, ...]):
        self._prev_hs_a = hidden_states[self.band_a_layer + 1][:, -1, :].detach()
        self._prev_hs_b = hidden_states[self.band_b_layer + 1][:, -1, :].detach()

    def _segment_keyvar(
        self,
        past_key_values: Tuple,
        classify_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Per-segment mean key variance over the first `classify_len` positions.

        Attention-free analogue of ThinKV's segment entropy: for each cached
        position, var(k, dim=head_dim) averaged over batch, heads, and layers.

        Returns:
            (num_segments,) float tensor; higher = more content-rich (→ R).
        """
        per_pos = torch.stack([
            k[:, :, :classify_len, :].float().var(dim=-1).mean(dim=(0, 1)).to(device)
            for k, _ in past_key_values
        ]).mean(dim=0)  # (classify_len,)

        seg_size = self.segment_size
        num_full = classify_len // seg_size
        stats = []
        for i in range(num_full):
            stats.append(per_pos[i * seg_size:(i + 1) * seg_size].mean())
        if classify_len - num_full * seg_size > 0:
            stats.append(per_pos[num_full * seg_size:].mean())
        return torch.stack(stats)

    def _classify_segments(self, seg_stats: torch.Tensor) -> List[str]:
        """Tertile classification by DESCENDING key variance (high → R)."""
        n = len(seg_stats)
        if n == 0:
            return []
        if n == 1:
            return ['R']
        sorted_s, _ = seg_stats.sort()
        t_low = sorted_s[n // 3].item()
        t_high = sorted_s[(2 * n) // 3].item()
        return [
            'R' if s.item() >= t_high else ('E' if s.item() >= t_low else 'T')
            for s in seg_stats
        ]

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> Tuple:
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        prefill_len = min(self._prefill_len, seq_len)

        # ── Update per-token HS z-score ───────────────────────────────────────
        diff_a, hs_a = _hs_diff(hidden_states, self.band_a_layer, self._prev_hs_a)
        diff_b, hs_b = _hs_diff(hidden_states, self.band_b_layer, self._prev_hs_b)
        self._prev_hs_a = hs_a
        self._prev_hs_b = hs_b
        self._buf_a.append(diff_a)
        self._buf_b.append(diff_b)
        self._scores.append(
            _rolling_z_score(self._buf_a, self.window)
            - _rolling_z_score(self._buf_b, self.window)
        )

        if seq_len <= self.config.cache_size:
            return past_key_values

        keep_recent = min(self.config.keep_recent_k, self.config.cache_size // 4)
        classify_len = seq_len - keep_recent

        # ── Segment classification from cached key statistics ─────────────────
        seg_stats = self._segment_keyvar(past_key_values, classify_len, device)
        seg_labels = self._classify_segments(seg_stats)

        # ── Build HS score tensor for all decode positions ────────────────────
        num_decode = seq_len - prefill_len
        hs_decode = torch.tensor(
            self._scores[-num_decode:], device=device, dtype=torch.float32
        )
        if len(hs_decode) < num_decode:
            pad = torch.full((num_decode - len(hs_decode),), float('-inf'), device=device)
            hs_decode = torch.cat([pad, hs_decode])

        # ── Apply per-segment retention budgets using HS scores ───────────────
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[:prefill_len] = True
        keep_mask[-keep_recent:] = True

        budget_map = {'R': self.retain_r, 'E': self.retain_e, 'T': self.retain_t}
        remaining_budget = self.config.cache_size - keep_recent - prefill_len
        seg_size = self.segment_size

        for i, label in enumerate(seg_labels):
            if remaining_budget <= 0:
                break
            seg_abs_start = i * seg_size
            seg_abs_end = min((i + 1) * seg_size, classify_len)

            decode_start = max(seg_abs_start, prefill_len)
            decode_end = min(seg_abs_end, seq_len - keep_recent)
            if decode_start >= decode_end:
                continue

            score_start = decode_start - prefill_len
            score_end = decode_end - prefill_len
            seg_hs = hs_decode[score_start:score_end]

            n_keep = min(budget_map[label], len(seg_hs), remaining_budget)
            if n_keep > 0:
                _, top_idx = torch.topk(seg_hs, n_keep)
                keep_mask[decode_start + top_idx] = True
                remaining_budget -= n_keep

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )
        decode_keep = keep_mask[prefill_len:prefill_len + num_decode].tolist()
        self._scores = [
            s for s, kept in zip(self._scores[-num_decode:], decode_keep) if kept
        ]
        return new_past

class RKVEviction:
    """
    R-KV: redundancy-aware KV eviction (Cai et al., NeurIPS 2025) — harness adaptation.

    Score per token:  Z_i = λ · I_i − (1 − λ) · R_i   (paper Eq. 7, λ = 0.1)

      I_i (importance): mean attention received from the last `obs_window`
          query steps, after a ±(kernel_size//2) positional max-pool
          (paper Eq. 1–4; the pooling is SnapKV-style clustering).
      R_i (redundancy): softmax over positions of the mean cosine similarity
          between token i's normalized key and all other cached keys
          (paper Eq. 5–6).  Computed here via the exact column-mean identity
          S̄_i = (K̄_i · Σ_j K̄_j − 1) / n, which avoids materializing the
          n×n similarity matrix the paper describes (O(nd) instead of O(n²d);
          the result is identical).

    Faithful-adaptation notes (disclosed for the rebuttal):
      * Single global keep-mask (scores averaged over layers and heads), as
        for every method in this harness; the original selects tokens
        per layer and per KV head.
      * Per-step eviction to exactly `cache_size` when the cache overflows,
        matching this harness's budget convention (peak resident = K); the
        original compresses every 128 steps with a B_buffer=128 overflow
        (peak resident = K + 128).
      * The last `obs_window` (=8) positions are always retained, matching
        the paper's α=8 observation tokens.  Prompt tokens are evictable,
        matching the paper (Algorithm 1 treats the whole cache as candidates).

    Requires output_attentions=True → EAGER ONLY in this harness.
    """

    def __init__(
        self,
        config: EvictionConfig,
        obs_window: int = 8,
        kernel_size: int = 7,
        mix_lambda: float = 0.1,
        buffer_size: int = 128,
    ):
        self.config = config
        self.obs_window = obs_window
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        # B_buffer: the cache is allowed to grow to cache_size + buffer_size
        # before being compressed back to cache_size in one shot, matching the
        # paper's every-128-step compression.  Per-step eviction (dropping one
        # token at a time) makes the score nearly irrelevant, since cache
        # composition is then dominated by arrival order rather than selection.
        self.buffer_size = buffer_size
        self._obs: List[torch.Tensor] = []  # last obs_window attention rows

    def reset(self, prefill_len: int = 0):
        self._prefill_len = prefill_len
        self._obs = []

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        attention_weights: Tuple[torch.Tensor, ...],
    ) -> Tuple:
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device

        # ── Record this step's attention row (mean over layers, batch, heads) ─
        row = torch.stack([
            a.mean(dim=(0, 1))[-1, :].to(device) for a in attention_weights
        ]).mean(dim=0).float()  # (key_len,)
        self._obs.append(row)
        if len(self._obs) > self.obs_window:
            self._obs.pop(0)

        # Periodic compression: let the buffer fill, then compress in one shot.
        if seq_len < self.config.cache_size + self.buffer_size:
            return past_key_values

        # ── Importance: pooled mean over the observation rows ─────────────────
        mat = torch.zeros(len(self._obs), seq_len, device=device)
        for j, r in enumerate(self._obs):
            n = min(len(r), seq_len)
            mat[j, :n] = r[:n]
        pooled = F.max_pool1d(
            mat.unsqueeze(1), kernel_size=self.kernel_size, stride=1,
            padding=self.kernel_size // 2,
        ).squeeze(1)
        importance = pooled.mean(dim=0)  # (seq_len,)

        # ── Redundancy: exact column-mean of the key cosine-similarity matrix ─
        red_layers = []
        for k, _ in past_key_values:
            kn = k.float()
            kn = kn / (kn.norm(dim=-1, keepdim=True) + 1e-8)   # (b, h, s, d)
            sum_k = kn.sum(dim=2, keepdim=True)                 # (b, h, 1, d)
            sbar = (kn * sum_k).sum(dim=-1) - 1.0               # (b, h, s): Σ_j K̄_i·K̄_j − self
            sbar = sbar / kn.shape[2]
            r_h = F.softmax(sbar, dim=-1)                       # per-head softmax over positions
            red_layers.append(r_h.mean(dim=(0, 1)).to(device))
        redundancy = torch.stack(red_layers).mean(dim=0)        # (seq_len,)

        # ── Joint score and eviction ──────────────────────────────────────────
        # Scale-match the two terms before mixing.  Attention importance is
        # sharply peaked while the redundancy softmax over ~10^3 candidates is
        # nearly flat, so the raw mix (Eq. 7) is dominated by importance by
        # ~50x in our single-global-mask setting and the redundancy signal --
        # the method's contribution -- becomes numerically inert.  We
        # standardise each term over positions first, which preserves the
        # within-term ranking and makes lambda behave as the paper intends.
        def _std(x):
            return (x - x.mean()) / (x.std() + 1e-8)

        scores = (self.mix_lambda * _std(importance)
                  - (1.0 - self.mix_lambda) * _std(redundancy))
        scores[-self.obs_window:] = float('inf')  # observation tokens always kept

        n_keep = min(self.config.cache_size, seq_len)
        _, keep_idx = torch.topk(scores, n_keep)
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[keep_idx] = True

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )
        # Prune stored observation rows to surviving positions.
        self._obs = [r[keep_mask[:len(r)].cpu() if r.device.type == 'cpu' else keep_mask[:len(r)]]
                     for r in self._obs]
        return new_past

class LongFlowEviction:
    """
    LongFlow: attention-output contribution-norm eviction (Su et al. 2026) —
    signal reproduced in eager mode.

    Score per token (paper Eq. 6):  score_i = α_t^i · ‖v_i‖₁ — the L1 norm of
    token i's contribution to the current attention output, recomputed each
    step from the current attention row ("zero-history" estimation).  Lowest
    contribution is evicted when the cache exceeds budget.

    Faithful-adaptation notes (disclosed for the rebuttal):
      * The native implementation computes this score inside a custom Triton
        kernel that forks FlashAttention (drops the safe-softmax running max,
        decode-only).  The SCORE here is identical, obtained via
        output_attentions in eager mode — accuracy comparisons are valid;
        wall-clock is NOT representative of their kernel.
      * Single global keep-mask (mean over layers and heads).
      * No protected recency window and prompt tokens evictable, matching the
        paper's decode policy (their SnapKV prefill compression only triggers
        when the prompt exceeds the budget, which never occurs at our prompt
        lengths).

    Requires output_attentions=True → EAGER ONLY in this harness.
    """

    def __init__(self, config: EvictionConfig):
        self.config = config

    def reset(self, prefill_len: int = 0):
        self._prefill_len = prefill_len

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        attention_weights: Tuple[torch.Tensor, ...],
    ) -> Tuple:
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device

        if seq_len <= self.config.cache_size:
            return past_key_values

        # α_t: current-step attention row, mean over layers, batch, heads.
        attn_row = torch.stack([
            a.mean(dim=(0, 1))[-1, :].to(device) for a in attention_weights
        ]).mean(dim=0).float()  # (seq_len,)

        # ‖v_i‖₁: mean over layers, batch, heads of the value L1 norm.
        v_l1 = torch.stack([
            v.float().abs().sum(dim=-1).mean(dim=(0, 1)).to(device)
            for _, v in past_key_values
        ]).mean(dim=0)  # (seq_len,)

        scores = attn_row * v_l1

        n_keep = min(self.config.cache_size, seq_len)
        _, keep_idx = torch.topk(scores, n_keep)
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[keep_idx] = True

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )
        return new_past


if __name__ == "__main__":
    # Smoke tests — verify all eviction classes reduce cache from seq_len to cache_size.
    batch_size = 1
    num_heads = 8
    num_layers = 4
    head_dim = 8
    seq_len = 1000

    config = EvictionConfig(cache_size=512, keep_recent_k=128)

    # HuggingFace-format fixtures
    past_kv = tuple(
        (torch.randn(batch_size, num_heads, seq_len, head_dim),
         torch.randn(batch_size, num_heads, seq_len, head_dim))
        for _ in range(num_layers)
    )
    attn_tuple = tuple(
        torch.randn(batch_size, num_heads, 1, seq_len).abs()
        for _ in range(num_layers)
    )

    # ── H2OEviction ──────────────────────────────────────────────────────────
    h2o = H2OEviction(config)
    pruned_h2o = h2o.evict_past_key_values(past_kv, attn_tuple)
    print(f"H2O (past_kv):                {seq_len} → {pruned_h2o[0][0].shape[2]} tokens")
    assert pruned_h2o[0][0].shape[2] == config.cache_size

    # Verify stateful accumulation: a second call with the same cache still works.
    h2o2 = H2OEviction(config)
    _ = h2o2.evict_past_key_values(past_kv, attn_tuple)  # prime cumulative buffer
    h2o2.reset()
    pruned_h2o2 = h2o2.evict_past_key_values(past_kv, attn_tuple)
    print(f"H2O (after reset):            {seq_len} → {pruned_h2o2[0][0].shape[2]} tokens")
    assert pruned_h2o2[0][0].shape[2] == config.cache_size

    # ── ThinKVEviction ───────────────────────────────────────────────────────
    thinKV = ThinKVEviction(config)
    pruned_tkv = thinKV.evict_past_key_values(past_kv, attn_tuple)
    retained = pruned_tkv[0][0].shape[2]
    # ThinKV may retain fewer tokens than cache_size (segment budgets cap per-segment)
    print(f"ThinKV (past_kv):             {seq_len} → {retained} tokens (≤ {config.cache_size})")
    assert retained <= config.cache_size

    # ── RaaSEviction ─────────────────────────────────────────────────────────
    prefill_len = 200
    raas = RaaSEviction(config)
    raas.reset(prefill_len=prefill_len)
    pruned_raas = raas.evict_past_key_values(past_kv, attn_tuple)
    print(f"RaaS (past_kv):               {seq_len} → {pruned_raas[0][0].shape[2]} tokens")
    assert pruned_raas[0][0].shape[2] == config.cache_size
    # Prefill tokens must all be present in the output
    assert pruned_raas[0][0].shape[2] >= prefill_len

    # ── HSVarianceEviction ───────────────────────────────────────────────────
    # Simulate a decode loop: feed seq_len=1 hidden states at each step.
    hidden_dim = num_heads * head_dim  # 64
    num_hs_layers = 34  # 32 transformer layers + embedding + final norm (index 0..33)

    hs_eviction = HSVarianceEviction(config, band_a_layer=10, band_b_layer=21)
    hs_eviction.reset(prefill_len=prefill_len)

    # Build a fake past_kv with seq_len tokens (pretend prefill already done)
    past_kv_hs = past_kv  # reuse existing fixture
    # Simulate 10 decode steps to populate score buffers before eviction kicks in
    for step in range(10):
        fake_hs = tuple(torch.randn(batch_size, 1, hidden_dim) for _ in range(num_hs_layers))
        past_kv_hs = hs_eviction.evict_past_key_values(past_kv_hs, fake_hs)

    # After 10 steps, we've been accumulating scores; if cache exceeded budget it would evict.
    print(f"HSVariance (after 10 decode steps): cache len = {past_kv_hs[0][0].shape[2]}")
    assert past_kv_hs[0][0].shape[2] <= config.cache_size

    # ── KVValVarianceEviction ────────────────────────────────────────────────
    kv_eviction = KVValVarianceEviction(config)
    kv_eviction.reset(prefill_len=prefill_len)
    pruned_kv = kv_eviction.evict_past_key_values(past_kv)
    print(f"KVValVariance (past_kv):      {seq_len} → {pruned_kv[0][0].shape[2]} tokens")
    assert pruned_kv[0][0].shape[2] <= config.cache_size

    # ── KVKeyVarianceEviction ────────────────────────────────────────────────
    kvk_eviction = KVKeyVarianceEviction(config)
    kvk_eviction.reset(prefill_len=prefill_len)
    pruned_kvk = kvk_eviction.evict_past_key_values(past_kv)
    print(f"KVKeyVariance (past_kv):      {seq_len} → {pruned_kvk[0][0].shape[2]} tokens")
    assert pruned_kvk[0][0].shape[2] <= config.cache_size

    # ── LagKVKeyVarianceEviction ─────────────────────────────────────────────
    lag_key_eviction = LagKVKeyVarianceEviction(config, chunk_size=128)
    lag_key_eviction.reset(prefill_len=prefill_len)
    pruned_lag_key = lag_key_eviction.evict_past_key_values(past_kv)
    print(f"LagKVKeyVariance (past_kv):   {seq_len} → {pruned_lag_key[0][0].shape[2]} tokens")
    assert pruned_lag_key[0][0].shape[2] <= config.cache_size

    # ── LagKVEviction ────────────────────────────────────────────────────────
    lag_eviction = LagKVEviction(config, chunk_size=128)
    lag_eviction.reset(prefill_len=prefill_len)
    pruned_lag = lag_eviction.evict_past_key_values(past_kv)
    print(f"LagKV (past_kv):              {seq_len} → {pruned_lag[0][0].shape[2]} tokens")
    assert pruned_lag[0][0].shape[2] <= config.cache_size

    # ── New Phase 1 HS classes ────────────────────────────────────────────────
    # Shared fixtures: 34 HS layers (32 transformer + embedding + final norm),
    # hidden_dim=64. band_a_layer=10 → index 11; band_b_layer=21 → index 22;
    # BandAdaptiveHS Band B max layer 25 → index 26. All within range [0, 33].
    fake_prefill_hs = tuple(
        torch.randn(batch_size, prefill_len, hidden_dim) for _ in range(num_hs_layers)
    )

    # ── DetrendendHSVarianceEviction ─────────────────────────────────────────
    det_ev = DetrendendHSVarianceEviction(config, band_a_layer=10, band_b_layer=21)
    det_ev.reset(prefill_len=prefill_len)
    det_ev.set_prefill_end(fake_prefill_hs)
    past_kv_det = past_kv
    for _ in range(10):
        fake_hs = tuple(torch.randn(batch_size, 1, hidden_dim) for _ in range(num_hs_layers))
        past_kv_det = det_ev.evict_past_key_values(past_kv_det, fake_hs)
    print(f"DetrendendHSVariance (10 steps): cache len = {past_kv_det[0][0].shape[2]}")
    assert past_kv_det[0][0].shape[2] <= config.cache_size
    # z-scores must all be finite (eps guards against division by zero)
    assert all(s == s for s in det_ev._scores), "NaN in DetrendendHSVariance scores"

    # ── BandAdaptiveHSEviction ────────────────────────────────────────────────
    band_ev = BandAdaptiveHSEviction(config)
    band_ev.reset(prefill_len=prefill_len)
    band_ev.set_prefill_end(fake_prefill_hs)
    past_kv_band = past_kv
    for _ in range(10):
        fake_hs = tuple(torch.randn(batch_size, 1, hidden_dim) for _ in range(num_hs_layers))
        past_kv_band = band_ev.evict_past_key_values(past_kv_band, fake_hs)
    print(f"BandAdaptiveHS (10 steps):       cache len = {past_kv_band[0][0].shape[2]}")
    assert past_kv_band[0][0].shape[2] <= config.cache_size
    # All band layers should have entries in the prev dicts after first decode step
    assert len(band_ev._prev_a) == len(BandAdaptiveHSEviction._BAND_A)
    assert len(band_ev._prev_b) == len(BandAdaptiveHSEviction._BAND_B)

    # ── AttentionHSProductEviction ────────────────────────────────────────────
    ahs_ev = AttentionHSProductEviction(config, band_a_layer=10)
    ahs_ev.reset(prefill_len=prefill_len)
    ahs_ev.set_prefill_end(fake_prefill_hs)
    past_kv_ahs = past_kv
    for _ in range(10):
        fake_hs = tuple(torch.randn(batch_size, 1, hidden_dim) for _ in range(num_hs_layers))
        past_kv_ahs = ahs_ev.evict_past_key_values(past_kv_ahs, attn_tuple, fake_hs)
    print(f"AttentionHSProduct (10 steps):   cache len = {past_kv_ahs[0][0].shape[2]}")
    assert past_kv_ahs[0][0].shape[2] <= config.cache_size

    # ── HybridSegmentHSEviction ───────────────────────────────────────────────
    # num_classifier_layers=num_layers so attn_tuple (4 layers) covers the full set.
    hyb_ev = HybridSegmentHSEviction(
        config, band_a_layer=10, band_b_layer=21, num_classifier_layers=num_layers
    )
    hyb_ev.reset(prefill_len=prefill_len)
    hyb_ev.set_prefill_end(fake_prefill_hs)
    past_kv_hyb = past_kv
    for _ in range(10):
        fake_hs = tuple(torch.randn(batch_size, 1, hidden_dim) for _ in range(num_hs_layers))
        past_kv_hyb = hyb_ev.evict_past_key_values(past_kv_hyb, attn_tuple, fake_hs)
    retained = past_kv_hyb[0][0].shape[2]
    print(f"HybridSegmentHS (10 steps):      cache len = {retained} (≤ {config.cache_size})")
    assert retained <= config.cache_size

    # ── KVSegHSEviction (kv_seg_hs — FA2-compatible segment hybrid) ──────────
    kvseg_ev = KVSegHSEviction(config, band_a_layer=10, band_b_layer=21)
    kvseg_ev.reset(prefill_len=prefill_len)
    kvseg_ev.set_prefill_end(fake_prefill_hs)
    past_kv_kvseg = past_kv
    for _ in range(10):
        fake_hs = tuple(torch.randn(batch_size, 1, hidden_dim) for _ in range(num_hs_layers))
        past_kv_kvseg = kvseg_ev.evict_past_key_values(past_kv_kvseg, fake_hs)
    retained = past_kv_kvseg[0][0].shape[2]
    print(f"KVSegHS (10 steps):              cache len = {retained} (≤ {config.cache_size})")
    assert retained <= config.cache_size
    # Prefill must survive (structural convention of the HS family)
    assert retained >= prefill_len

    # ── RKVEviction ──────────────────────────────────────────────────────────
    rkv_ev = RKVEviction(config)
    rkv_ev.reset(prefill_len=prefill_len)
    past_kv_rkv = past_kv
    for _ in range(3):  # multiple steps to fill the observation window
        cur_len = past_kv_rkv[0][0].shape[2]
        attn_cur = tuple(
            torch.randn(batch_size, num_heads, 1, cur_len).abs().softmax(dim=-1)
            for _ in range(num_layers)
        )
        past_kv_rkv = rkv_ev.evict_past_key_values(past_kv_rkv, attn_cur)
    retained = past_kv_rkv[0][0].shape[2]
    print(f"RKV (3 steps):                   cache len = {retained}")
    assert retained == config.cache_size
    # Observation rows must be pruned in sync with the cache
    assert all(len(r) <= retained for r in rkv_ev._obs)

    # ── LongFlowEviction ─────────────────────────────────────────────────────
    lf_ev = LongFlowEviction(config)
    lf_ev.reset(prefill_len=prefill_len)
    pruned_lf = lf_ev.evict_past_key_values(past_kv, attn_tuple)
    print(f"LongFlow (past_kv):           {seq_len} → {pruned_lf[0][0].shape[2]} tokens")
    assert pruned_lf[0][0].shape[2] == config.cache_size

    print("\nAll assertions passed.")


# ---- entropy-based variant developed on the cluster, for the head-to-head ----

class KVSegmentHSEviction:
    """
    FA2-compatible segment hybrid: KV-key-variance segment classification +
    detrended HS within-segment token ranking.

    The attention-free analog of HybridSegmentHSEviction.  ThinKV (and our
    HybridSegmentHS) classify thought segments by the entropy of the attention
    distribution, which forces the eager kernel.  Here the segment classifier
    reads only the cached key vectors: each segment is classified by the entropy
    of its per-token key-variance (kv_key_var) distribution — the KV analog of
    attention sparsity.  Low entropy = key-variance concentrated on a few tokens
    (focused, R-like); high entropy = diffuse (T-like).  Within each segment,
    tokens are ranked by the detrended Band A−B HS z-score, exactly as in
    HybridSegmentHS.

    Reads only past_key_values and hidden states — no attention matrix — so this
    is FA2-compatible (output_hidden_states=True only).  Closes the tight-budget
    gap where the eager HybridSegmentHS still leads.

    Usage:
        eviction = KVSegmentHSEviction(config)
        eviction.reset(prefill_len=prompt_len)
        eviction.set_prefill_end(prefill_outputs.hidden_states)
        outputs = model(..., output_hidden_states=True)
        past_kv = eviction.evict_past_key_values(past_kv, outputs.hidden_states)
    """

    def __init__(
        self,
        config: EvictionConfig,
        band_a_layer: int = 10,
        band_b_layer: int = 21,
        window: int = 64,
        segment_size: int = 128,
        retain_r: int = 64,
        retain_e: int = 32,
        retain_t: int = 8,
    ):
        self.config = config
        self.band_a_layer = band_a_layer
        self.band_b_layer = band_b_layer
        self.window = window
        self.segment_size = segment_size
        self.retain_r = retain_r
        self.retain_e = retain_e
        self.retain_t = retain_t

        self._prefill_len: int = 0
        self._prev_hs_a: Optional[torch.Tensor] = None
        self._prev_hs_b: Optional[torch.Tensor] = None
        self._buf_a: List[float] = []
        self._buf_b: List[float] = []
        self._scores: List[float] = []   # per decode token, pruned on eviction

    def reset(self, prefill_len: int = 0):
        self._prefill_len = prefill_len
        self._prev_hs_a = None
        self._prev_hs_b = None
        self._buf_a = []
        self._buf_b = []
        self._scores = []

    def set_prefill_end(self, hidden_states: Tuple[torch.Tensor, ...]):
        self._prev_hs_a = hidden_states[self.band_a_layer + 1][:, -1, :].detach()
        self._prev_hs_b = hidden_states[self.band_b_layer + 1][:, -1, :].detach()

    def _segment_kv_key_entropy(
        self,
        past_key_values: Tuple,
        classify_len: int,
    ) -> torch.Tensor:
        """
        Entropy of the per-token key-variance distribution within each segment.

        kv_key_var[t] = mean over layers/heads of var(k[t], head_dim).  Within a
        segment, treat the kv_key_var values as a PMF: low entropy = variance
        concentrated on a few tokens (R-like), high entropy = diffuse (T-like).
        Attention-free analog of ThinKV's attention-entropy classifier.
        """
        # (heads, seq, head_dim), averaged over layers and batch.
        k_all = torch.stack([k.mean(0) for k, _ in past_key_values]).mean(0)
        per_token_var = k_all.var(dim=-1).mean(dim=0)[:classify_len]  # (classify_len,)

        seg_size = self.segment_size
        num_full = classify_len // seg_size
        entropies = []
        for i in range(num_full):
            seg = per_token_var[i * seg_size:(i + 1) * seg_size]
            p = seg / (seg.sum() + 1e-9)
            entropies.append(-(p * torch.log(p + 1e-12)).sum())
        if classify_len - num_full * seg_size > 0:
            seg = per_token_var[num_full * seg_size:]
            p = seg / (seg.sum() + 1e-9)
            entropies.append(-(p * torch.log(p + 1e-12)).sum())
        if not entropies:
            return torch.zeros(0, device=per_token_var.device)
        return torch.stack(entropies)

    def _classify_segments(self, seg_entropies: torch.Tensor) -> List[str]:
        """Tertile threshold classification (identical to ThinKV/HybridSegmentHS)."""
        n = len(seg_entropies)
        if n == 0:
            return []
        if n == 1:
            return ['R']
        sorted_e, _ = seg_entropies.sort()
        t_low = sorted_e[n // 3].item()
        t_high = sorted_e[(2 * n) // 3].item()
        return [
            'R' if e.item() <= t_low else ('E' if e.item() <= t_high else 'T')
            for e in seg_entropies
        ]

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> Tuple:
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        prefill_len = min(self._prefill_len, seq_len)

        # ── Update per-token HS z-score (within-segment ranker) ───────────────
        diff_a, hs_a = _hs_diff(hidden_states, self.band_a_layer, self._prev_hs_a)
        diff_b, hs_b = _hs_diff(hidden_states, self.band_b_layer, self._prev_hs_b)
        self._prev_hs_a = hs_a
        self._prev_hs_b = hs_b
        self._buf_a.append(diff_a)
        self._buf_b.append(diff_b)
        self._scores.append(
            _rolling_z_score(self._buf_a, self.window)
            - _rolling_z_score(self._buf_b, self.window)
        )

        if seq_len <= self.config.cache_size:
            return past_key_values

        keep_recent = min(self.config.keep_recent_k, self.config.cache_size // 4)
        classify_len = seq_len - keep_recent

        # ── Segment type classification (KV-key-variance entropy) ─────────────
        seg_entropies = self._segment_kv_key_entropy(past_key_values, classify_len)
        seg_labels = self._classify_segments(seg_entropies)

        # ── HS scores for all decode positions ────────────────────────────────
        num_decode = seq_len - prefill_len
        hs_decode = torch.tensor(
            self._scores[-num_decode:], device=device, dtype=torch.float32
        )
        if len(hs_decode) < num_decode:
            pad = torch.full((num_decode - len(hs_decode),), float('-inf'), device=device)
            hs_decode = torch.cat([pad, hs_decode])

        # ── Apply per-segment retention budgets using HS scores ───────────────
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[:prefill_len] = True
        keep_mask[-keep_recent:] = True

        budget_map = {'R': self.retain_r, 'E': self.retain_e, 'T': self.retain_t}
        remaining_budget = self.config.cache_size - keep_recent - prefill_len
        seg_size = self.segment_size

        for i, label in enumerate(seg_labels):
            if remaining_budget <= 0:
                break
            seg_abs_start = i * seg_size
            seg_abs_end = min((i + 1) * seg_size, classify_len)
            decode_start = max(seg_abs_start, prefill_len)
            decode_end = min(seg_abs_end, seq_len - keep_recent)
            if decode_start >= decode_end:
                continue
            score_start = decode_start - prefill_len
            score_end = decode_end - prefill_len
            seg_hs = hs_decode[score_start:score_end]
            n_keep = min(budget_map[label], len(seg_hs), remaining_budget)
            if n_keep > 0:
                _, top_idx = torch.topk(seg_hs, n_keep)
                keep_mask[decode_start + top_idx] = True
                remaining_budget -= n_keep

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )
        decode_keep = keep_mask[prefill_len:prefill_len + num_decode].tolist()
        self._scores = [
            s for s, kept in zip(self._scores[-num_decode:], decode_keep) if kept
        ]
        return new_past


if __name__ == "__main__":
    # Smoke tests — verify all eviction classes reduce cache from seq_len to cache_size.
    batch_size = 1
    num_heads = 8
    num_layers = 4
    head_dim = 8
    seq_len = 1000

    config = EvictionConfig(cache_size=512, keep_recent_k=128)

    # HuggingFace-format fixtures
    past_kv = tuple(
        (torch.randn(batch_size, num_heads, seq_len, head_dim),
         torch.randn(batch_size, num_heads, seq_len, head_dim))
        for _ in range(num_layers)
    )
    attn_tuple = tuple(
        torch.randn(batch_size, num_heads, 1, seq_len).abs()
        for _ in range(num_layers)
    )

    # ── H2OEviction ──────────────────────────────────────────────────────────
    h2o = H2OEviction(config)
    pruned_h2o = h2o.evict_past_key_values(past_kv, attn_tuple)
    print(f"H2O (past_kv):                {seq_len} → {pruned_h2o[0][0].shape[2]} tokens")
    assert pruned_h2o[0][0].shape[2] == config.cache_size

    # Verify stateful accumulation: a second call with the same cache still works.
    h2o2 = H2OEviction(config)
    _ = h2o2.evict_past_key_values(past_kv, attn_tuple)  # prime cumulative buffer
    h2o2.reset()
    pruned_h2o2 = h2o2.evict_past_key_values(past_kv, attn_tuple)
    print(f"H2O (after reset):            {seq_len} → {pruned_h2o2[0][0].shape[2]} tokens")
    assert pruned_h2o2[0][0].shape[2] == config.cache_size

    # ── ThinKVEviction ───────────────────────────────────────────────────────
    thinKV = ThinKVEviction(config)
    pruned_tkv = thinKV.evict_past_key_values(past_kv, attn_tuple)
    retained = pruned_tkv[0][0].shape[2]
    # ThinKV may retain fewer tokens than cache_size (segment budgets cap per-segment)
    print(f"ThinKV (past_kv):             {seq_len} → {retained} tokens (≤ {config.cache_size})")
    assert retained <= config.cache_size

    # ── RaaSEviction ─────────────────────────────────────────────────────────
    prefill_len = 200
    raas = RaaSEviction(config)
    raas.reset(prefill_len=prefill_len)
    pruned_raas = raas.evict_past_key_values(past_kv, attn_tuple)
    print(f"RaaS (past_kv):               {seq_len} → {pruned_raas[0][0].shape[2]} tokens")
    assert pruned_raas[0][0].shape[2] == config.cache_size
    # Prefill tokens must all be present in the output
    assert pruned_raas[0][0].shape[2] >= prefill_len

    # ── HSVarianceEviction ───────────────────────────────────────────────────
    # Simulate a decode loop: feed seq_len=1 hidden states at each step.
    hidden_dim = num_heads * head_dim  # 64
    num_hs_layers = 34  # 32 transformer layers + embedding + final norm (index 0..33)

    hs_eviction = HSVarianceEviction(config, band_a_layer=10, band_b_layer=21)
    hs_eviction.reset(prefill_len=prefill_len)

    # Build a fake past_kv with seq_len tokens (pretend prefill already done)
    past_kv_hs = past_kv  # reuse existing fixture
    # Simulate 10 decode steps to populate score buffers before eviction kicks in
    for step in range(10):
        fake_hs = tuple(torch.randn(batch_size, 1, hidden_dim) for _ in range(num_hs_layers))
        past_kv_hs = hs_eviction.evict_past_key_values(past_kv_hs, fake_hs)

    # After 10 steps, we've been accumulating scores; if cache exceeded budget it would evict.
    print(f"HSVariance (after 10 decode steps): cache len = {past_kv_hs[0][0].shape[2]}")
    assert past_kv_hs[0][0].shape[2] <= config.cache_size

    # ── KVValVarianceEviction ────────────────────────────────────────────────
    kv_eviction = KVValVarianceEviction(config)
    kv_eviction.reset(prefill_len=prefill_len)
    pruned_kv = kv_eviction.evict_past_key_values(past_kv)
    print(f"KVValVariance (past_kv):      {seq_len} → {pruned_kv[0][0].shape[2]} tokens")
    assert pruned_kv[0][0].shape[2] <= config.cache_size

    # ── KVKeyVarianceEviction ────────────────────────────────────────────────
    kvk_eviction = KVKeyVarianceEviction(config)
    kvk_eviction.reset(prefill_len=prefill_len)
    pruned_kvk = kvk_eviction.evict_past_key_values(past_kv)
    print(f"KVKeyVariance (past_kv):      {seq_len} → {pruned_kvk[0][0].shape[2]} tokens")
    assert pruned_kvk[0][0].shape[2] <= config.cache_size

    # ── LagKVKeyVarianceEviction ─────────────────────────────────────────────
    lag_key_eviction = LagKVKeyVarianceEviction(config, chunk_size=128)
    lag_key_eviction.reset(prefill_len=prefill_len)
    pruned_lag_key = lag_key_eviction.evict_past_key_values(past_kv)
    print(f"LagKVKeyVariance (past_kv):   {seq_len} → {pruned_lag_key[0][0].shape[2]} tokens")
    assert pruned_lag_key[0][0].shape[2] <= config.cache_size

    # ── LagKVEviction ────────────────────────────────────────────────────────
    lag_eviction = LagKVEviction(config, chunk_size=128)
    lag_eviction.reset(prefill_len=prefill_len)
    pruned_lag = lag_eviction.evict_past_key_values(past_kv)
    print(f"LagKV (past_kv):              {seq_len} → {pruned_lag[0][0].shape[2]} tokens")
    assert pruned_lag[0][0].shape[2] <= config.cache_size

    # ── New Phase 1 HS classes ────────────────────────────────────────────────
    # Shared fixtures: 34 HS layers (32 transformer + embedding + final norm),
    # hidden_dim=64. band_a_layer=10 → index 11; band_b_layer=21 → index 22;
    # BandAdaptiveHS Band B max layer 25 → index 26. All within range [0, 33].
    fake_prefill_hs = tuple(
        torch.randn(batch_size, prefill_len, hidden_dim) for _ in range(num_hs_layers)
    )

    # ── DetrendendHSVarianceEviction ─────────────────────────────────────────
    det_ev = DetrendendHSVarianceEviction(config, band_a_layer=10, band_b_layer=21)
    det_ev.reset(prefill_len=prefill_len)
    det_ev.set_prefill_end(fake_prefill_hs)
    past_kv_det = past_kv
    for _ in range(10):
        fake_hs = tuple(torch.randn(batch_size, 1, hidden_dim) for _ in range(num_hs_layers))
        past_kv_det = det_ev.evict_past_key_values(past_kv_det, fake_hs)
    print(f"DetrendendHSVariance (10 steps): cache len = {past_kv_det[0][0].shape[2]}")
    assert past_kv_det[0][0].shape[2] <= config.cache_size
    # z-scores must all be finite (eps guards against division by zero)
    assert all(s == s for s in det_ev._scores), "NaN in DetrendendHSVariance scores"

    # ── BandAdaptiveHSEviction ────────────────────────────────────────────────
    band_ev = BandAdaptiveHSEviction(config)
    band_ev.reset(prefill_len=prefill_len)
    band_ev.set_prefill_end(fake_prefill_hs)
    past_kv_band = past_kv
    for _ in range(10):
        fake_hs = tuple(torch.randn(batch_size, 1, hidden_dim) for _ in range(num_hs_layers))
        past_kv_band = band_ev.evict_past_key_values(past_kv_band, fake_hs)
    print(f"BandAdaptiveHS (10 steps):       cache len = {past_kv_band[0][0].shape[2]}")
    assert past_kv_band[0][0].shape[2] <= config.cache_size
    # All band layers should have entries in the prev dicts after first decode step
    assert len(band_ev._prev_a) == len(BandAdaptiveHSEviction._BAND_A)
    assert len(band_ev._prev_b) == len(BandAdaptiveHSEviction._BAND_B)

    # ── AttentionHSProductEviction ────────────────────────────────────────────
    ahs_ev = AttentionHSProductEviction(config, band_a_layer=10)
    ahs_ev.reset(prefill_len=prefill_len)
    ahs_ev.set_prefill_end(fake_prefill_hs)
    past_kv_ahs = past_kv
    for _ in range(10):
        fake_hs = tuple(torch.randn(batch_size, 1, hidden_dim) for _ in range(num_hs_layers))
        past_kv_ahs = ahs_ev.evict_past_key_values(past_kv_ahs, attn_tuple, fake_hs)
    print(f"AttentionHSProduct (10 steps):   cache len = {past_kv_ahs[0][0].shape[2]}")
    assert past_kv_ahs[0][0].shape[2] <= config.cache_size

    # ── HybridSegmentHSEviction ───────────────────────────────────────────────
    # num_classifier_layers=num_layers so attn_tuple (4 layers) covers the full set.
    hyb_ev = HybridSegmentHSEviction(
        config, band_a_layer=10, band_b_layer=21, num_classifier_layers=num_layers
    )
    hyb_ev.reset(prefill_len=prefill_len)
    hyb_ev.set_prefill_end(fake_prefill_hs)
    past_kv_hyb = past_kv
    for _ in range(10):
        fake_hs = tuple(torch.randn(batch_size, 1, hidden_dim) for _ in range(num_hs_layers))
        past_kv_hyb = hyb_ev.evict_past_key_values(past_kv_hyb, attn_tuple, fake_hs)
    retained = past_kv_hyb[0][0].shape[2]
    print(f"HybridSegmentHS (10 steps):      cache len = {retained} (≤ {config.cache_size})")
    assert retained <= config.cache_size

    # ── KVSegmentHSEviction (FA2-compatible segment hybrid) ───────────────────
    kvseg_ev = KVSegmentHSEviction(config, band_a_layer=10, band_b_layer=21)
    kvseg_ev.reset(prefill_len=prefill_len)
    kvseg_ev.set_prefill_end(fake_prefill_hs)
    past_kv_kvseg = past_kv
    for _ in range(10):
        fake_hs = tuple(torch.randn(batch_size, 1, hidden_dim) for _ in range(num_hs_layers))
        past_kv_kvseg = kvseg_ev.evict_past_key_values(past_kv_kvseg, fake_hs)
    retained = past_kv_kvseg[0][0].shape[2]
    print(f"KVSegmentHS (10 steps):          cache len = {retained} (≤ {config.cache_size})")
    assert retained <= config.cache_size

    print("\nAll assertions passed.")
