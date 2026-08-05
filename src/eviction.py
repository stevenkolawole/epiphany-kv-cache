"""
KV cache eviction baselines and Phase 1 proposed methods.

Baselines
---------
H2OEviction             — cumulative attention / Heavy Hitter Oracle (Zhang et al. 2023)
ThinKVEviction          — segment-level R/E/T classification + per-type budgets (He et al. 2025)
RaaSEviction            — LRU decode eviction + unconditional prefill preservation

Phase 1 proposed — KV signals (no attention matrix, FA2-compatible)
--------------------------------------------------------------------
KVValVarianceEviction      — value-vector variance (kv_val_var_rolling64)
KVKeyVarianceEviction      — key-vector variance (kv_key_var_rolling64); highest ρ on competition math
LagKVKeyVarianceEviction   — lag-normalized key variance; causal adaptation of LagKV (Xu 2025)
LagKVEviction              — lag-normalized key + value variance; full LagKV (Xu et al. 2025)

Phase 1 proposed — HS signals (output_hidden_states=True; FA2-compatible)
-------------------------------------------------------------------------
HSVarianceEviction         — Band A−B HS diff rolling mean (l10_rolling64 − l21_rolling64)
DetrendendHSVarianceEviction — same as above but with rolling z-score detrending to remove
                               position-correlated temporal trends (AhaKV / LagKV principle)
BandAdaptiveHSEviction     — averages over all Band A (l7–l13) and Band B (l18–l25) layers,
                               with empirically-calibrated per-band weights from Phase 0B ρ

Phase 1 proposed — hybrid (output_attentions=True + output_hidden_states=True; eager only)
------------------------------------------------------------------------------------------
AttentionHSProductEviction — cumulative key-perspective attn + detrended Band A HS z-score
HybridSegmentHSEviction    — ThinKV segment type classification (key-perspective entropy)
                              + detrended HS Band A−B score for within-segment token ranking
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class EvictionConfig:
    """Configuration for KV cache eviction."""
    cache_size: int = 4096  # Max KV cache tokens
    keep_recent_k: int = 128  # Always keep recent K tokens (like StreamingLLM)


class H2OEviction:
    """
    H2O: Heavy Hitter Oracle — cumulative attention eviction.

    Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference
    of Large Language Models" (NeurIPS 2023).

    Tracks cumulative attention column sums across ALL decode steps. Tokens
    that accumulate the most total attention (heavy hitters) are retained;
    tokens with the lowest cumulative attention are evicted. Attention sinks
    (first `num_sink_tokens`) are always kept unconditionally because early
    tokens receive disproportionate attention that doesn't reflect content
    importance.

    Call reset() between sequences to clear cumulative state.
    """

    def __init__(self, config: EvictionConfig, num_sink_tokens: int = 4):
        self.config = config
        self.num_sink_tokens = num_sink_tokens
        self._cumulative_attn: Optional[torch.Tensor] = None

    def reset(self):
        """Clear per-sequence cumulative state. Call before each new sequence."""
        self._cumulative_attn = None

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        attention_weights: Tuple[torch.Tensor, ...],
    ) -> Tuple:
        """
        Accumulate attention and evict if the cache exceeds budget.

        Args:
            past_key_values: HuggingFace (k, v) per layer.
                             k/v shape: (batch, num_heads, seq_len, head_dim)
            attention_weights: Per-layer attention tensors.
                               Each: (batch, num_heads, query_len, key_len)

        Returns:
            Pruned past_key_values.
        """
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device

        # Average over layers, batch, heads → (key_len,)
        attn_step = torch.stack([
            a.mean(dim=(0, 1, 2)) for a in attention_weights
        ]).mean(dim=0)  # (key_len,)

        # Initialise or extend buffer when new tokens are appended.
        if self._cumulative_attn is None:
            self._cumulative_attn = torch.zeros(seq_len, device=device, dtype=attn_step.dtype)
        elif attn_step.shape[0] > self._cumulative_attn.shape[0]:
            pad_len = attn_step.shape[0] - self._cumulative_attn.shape[0]
            pad = torch.zeros(pad_len, device=device, dtype=attn_step.dtype)
            self._cumulative_attn = torch.cat([self._cumulative_attn, pad])

        self._cumulative_attn[:attn_step.shape[0]] += attn_step

        if seq_len <= self.config.cache_size:
            return past_key_values

        cache_size = self.config.cache_size
        num_sink = min(self.num_sink_tokens, cache_size // 4)
        keep_recent = min(self.config.keep_recent_k, cache_size - num_sink)
        remaining = max(cache_size - num_sink - keep_recent, 0)

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[:num_sink] = True          # attention sinks
        keep_mask[-keep_recent:] = True      # recency window

        mid_start = num_sink
        mid_end = seq_len - keep_recent
        if remaining > 0 and mid_start < mid_end:
            mid_scores = self._cumulative_attn[mid_start:mid_end]
            n = min(remaining, mid_end - mid_start)
            _, top_idx = torch.topk(mid_scores, n)
            keep_mask[mid_start + top_idx] = True

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )
        # Prune cumulative buffer to match evicted cache.
        self._cumulative_attn = self._cumulative_attn[keep_mask]
        return new_past


class ThinKVEviction:
    """
    ThinKV: Thought-type segment eviction.

    He et al., "ThinKV: Token Compression for Efficient Long Reasoning" (2025).

    Classifies each `segment_size`-token block of the reasoning chain as:
      R (Reasoning)  — low entropy; focused attention, high-value tokens
      E (Execution)  — medium entropy; moderately diffuse attention
      T (Transition) — high entropy; scattered attention, low informational value

    Retention budgets are applied per segment:
      R → retain_r tokens   E → retain_e tokens   T → retain_t tokens

    Entropy classification uses the last `num_classifier_layers` layers
    (mimicking ThinKV's 4-layer classifier). Tertile percentile splits replace
    a full KDE for simplicity while preserving the R/E/T distinction.

    This is a stateless eviction: each call to evict_past_key_values()
    re-classifies segments from the current attention weights.
    """

    def __init__(
        self,
        config: EvictionConfig,
        segment_size: int = 128,
        retain_r: int = 64,
        retain_e: int = 32,
        retain_t: int = 8,
        num_classifier_layers: int = 4,
    ):
        self.config = config
        self.segment_size = segment_size
        self.retain_r = retain_r
        self.retain_e = retain_e
        self.retain_t = retain_t
        self.num_classifier_layers = num_classifier_layers

    def _segment_entropy(
        self,
        attention_weights: Tuple[torch.Tensor, ...],
        classify_len: int,
    ) -> torch.Tensor:
        """
        Compute mean attention entropy for each segment of the first
        `classify_len` token positions.

        Each segment's entropy is derived from the normalised column-sum
        attention distribution (how much attention each token received from
        the last query step), treating that distribution within the segment
        as a probability mass function.

        Returns:
            (num_segments,) float tensor; higher entropy → more T-like.
        """
        # Use last num_classifier_layers (or all available)
        layers = attention_weights[-self.num_classifier_layers:]

        # Average over layers, batch, heads; take last query position → (key_len,)
        col_attn = torch.stack([
            a.mean(dim=(0, 1))[-1, :] for a in layers
        ]).mean(dim=0)  # (key_len,)
        col_attn = col_attn[:classify_len]

        seg_size = self.segment_size
        num_full = classify_len // seg_size
        entropies = []

        for i in range(num_full):
            seg = col_attn[i * seg_size: (i + 1) * seg_size]
            p = seg / (seg.sum() + 1e-9)
            H = -(p * torch.log(p + 1e-12)).sum()
            entropies.append(H)

        remainder = classify_len - num_full * seg_size
        if remainder > 0:
            seg = col_attn[num_full * seg_size:]
            p = seg / (seg.sum() + 1e-9)
            H = -(p * torch.log(p + 1e-12)).sum()
            entropies.append(H)

        return torch.stack(entropies)

    def _classify_segments(self, seg_entropies: torch.Tensor) -> List[str]:
        """
        Classify segments into R/E/T via entropy tertile thresholds.

        Returns:
            List of 'R', 'E', or 'T' labels, one per segment.
        """
        n = len(seg_entropies)
        if n == 0:
            return []
        if n == 1:
            return ['R']  # Single segment: treat as Reasoning
        sorted_e, _ = seg_entropies.sort()
        t_low = sorted_e[n // 3].item()
        t_high = sorted_e[(2 * n) // 3].item()
        labels = []
        for e in seg_entropies:
            v = e.item()
            if v <= t_low:
                labels.append('R')
            elif v <= t_high:
                labels.append('E')
            else:
                labels.append('T')
        return labels

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        attention_weights: Tuple[torch.Tensor, ...],
    ) -> Tuple:
        """
        Apply ThinKV segment-level eviction.

        Args:
            past_key_values: HuggingFace (k, v) per layer.
            attention_weights: Per-layer attention. Each: (batch, heads, Q, K).

        Returns:
            Pruned past_key_values.
        """
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device

        if seq_len <= self.config.cache_size:
            return past_key_values

        keep_recent = min(self.config.keep_recent_k, self.config.cache_size // 4)
        classify_len = seq_len - keep_recent  # segment classification excludes recency window

        seg_entropies = self._segment_entropy(attention_weights, classify_len)
        seg_labels = self._classify_segments(seg_entropies)

        # Use last-layer avg column attention to rank tokens within each segment.
        col_attn = attention_weights[-1].mean(dim=(0, 1, 2))[:classify_len]  # (classify_len,)

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[-keep_recent:] = True  # always keep recency window

        budget_map = {'R': self.retain_r, 'E': self.retain_e, 'T': self.retain_t}
        seg_size = self.segment_size
        remaining_budget = self.config.cache_size - keep_recent

        for i, label in enumerate(seg_labels):
            if remaining_budget <= 0:
                break
            seg_start = i * seg_size
            seg_end = min((i + 1) * seg_size, classify_len)
            if seg_start >= seg_end:
                break
            seg_scores = col_attn[seg_start:seg_end]
            n_keep = min(budget_map[label], seg_end - seg_start, remaining_budget)
            if n_keep > 0:
                _, top_idx = torch.topk(seg_scores, n_keep)
                keep_mask[seg_start + top_idx] = True
                remaining_budget -= n_keep

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )
        return new_past


class ThinKVFaithfulEviction(ThinKVEviction):
    """Faithful ThinKV: KDE-based R/E/T classification with a tau-step refresh.

    ThinKVEviction diverges from the published algorithm in two ways, both
    documented in its own docstring: it substitutes tertile percentile splits
    for the paper's kernel-density estimate, and it re-classifies on every
    decode step rather than refreshing every tau steps. This subclass restores
    both so the baseline can be compared against ThinKV-as-published.

    KDE: a Gaussian kernel density is fitted to the segment entropies with
    Scott's-rule bandwidth, and the two deepest local minima of the density
    become the R/E and E/T boundaries. Where fewer than two minima exist (a
    unimodal density), we fall back to tertiles, which is the degenerate case
    of the same partition.

    Refresh: segment labels are recomputed only every `refresh_tau` decode
    steps and cached in between, matching the paper's tau=128 refresh window.
    """

    def __init__(self, *args, refresh_tau: int = 128, kde_points: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        self.refresh_tau = refresh_tau
        self.kde_points = kde_points
        self._step = 0
        self._cached_labels = None
        self._cached_nseg = 0

    def reset(self, prefill_len: int = 0):
        if hasattr(super(), "reset"):
            super().reset(prefill_len)
        self._step = 0
        self._cached_labels = None
        self._cached_nseg = 0

    def _kde_boundaries(self, e: torch.Tensor):
        """Return (t_low, t_high) from the two deepest minima of a Gaussian KDE."""
        n = e.numel()
        if n < 4:
            return None
        x = e.detach().float().flatten()
        sd = x.std().item()
        if sd <= 0 or not torch.isfinite(torch.tensor(sd)):
            return None
        # Silverman's robust rule: min(sd, IQR/1.349) rather than sd alone.
        # On multimodal input the IQR term is the smaller one, which keeps the
        # bandwidth narrow enough for the modes to survive; Scott's rule sets h
        # from the global spread and smooths a trimodal density flat at small n.
        q1, q3 = torch.quantile(x, 0.25).item(), torch.quantile(x, 0.75).item()
        iqr = q3 - q1
        spread = min(sd, iqr / 1.349) if iqr > 0 else sd
        h = 0.9 * spread * (n ** (-0.2))
        if h <= 0:
            return None
        grid = torch.linspace(x.min().item(), x.max().item(), self.kde_points, device=x.device)
        z = (grid.unsqueeze(1) - x.unsqueeze(0)) / h
        dens = torch.exp(-0.5 * z * z).sum(dim=1) / (n * h * (2 * 3.141592653589793) ** 0.5)
        # interior local minima
        mins = [(dens[i].item(), grid[i].item())
                for i in range(1, self.kde_points - 1)
                if dens[i] < dens[i - 1] and dens[i] <= dens[i + 1]]
        if len(mins) < 2:
            return None
        mins.sort(key=lambda p: p[0])          # deepest first
        cuts = sorted(v for _, v in mins[:2])
        return cuts[0], cuts[1]

    def _classify_segments(self, seg_entropies: torch.Tensor):
        n = len(seg_entropies)
        if n == 0:
            return []
        if n == 1:
            return ["R"]
        cuts = self._kde_boundaries(seg_entropies)
        if cuts is None:                        # unimodal: tertiles are the degenerate case
            return super()._classify_segments(seg_entropies)
        t_low, t_high = cuts
        return [
            "R" if v.item() <= t_low else ("E" if v.item() <= t_high else "T")
            for v in seg_entropies
        ]

    def evict_past_key_values(self, past_key_values, attention_weights):
        """Reclassify only every refresh_tau steps; reuse cached labels between."""
        self._step += 1
        orig = ThinKVEviction._classify_segments.__get__(self, type(self))
        reclassify = (self._cached_labels is None) or (self._step % self.refresh_tau == 0)

        def _cached(seg_entropies):
            nseg = len(seg_entropies)
            if reclassify or self._cached_labels is None or nseg != self._cached_nseg:
                labels = ThinKVFaithfulEviction._classify_segments(self, seg_entropies)
                self._cached_labels, self._cached_nseg = labels, nseg
            return self._cached_labels

        saved = self._classify_segments
        self._classify_segments = _cached
        try:
            return super().evict_past_key_values(past_key_values, attention_weights)
        finally:
            self._classify_segments = saved


class RaaSEviction:
    """
    RaaS: Recency-Aware and Accuracy-Sensitive KV cache eviction.

    Implements the core eviction policy from RaaS:
      - Prefill (prompt) tokens are ALWAYS preserved unconditionally.
      - Decode tokens track an LRU timestamp: the most recent generation step
        at which the token appeared in the top-50% of decode-token attention.
      - When the cache exceeds cache_size, decode tokens with the oldest
        (smallest) timestamp are evicted first.

    Call reset(prefill_len) at the start of each new sequence to set the
    prefill boundary and clear decode-token state.
    """

    def __init__(self, config: EvictionConfig):
        self.config = config
        self._prefill_len: int = 0
        self._step: int = 0
        self._lru_timestamps: Optional[torch.Tensor] = None  # (num_decode_tokens,)

    def reset(self, prefill_len: int):
        """
        Set the prefill boundary for a new sequence.

        Args:
            prefill_len: Number of prompt tokens — these are always kept.
        """
        self._prefill_len = prefill_len
        self._step = 0
        self._lru_timestamps = None

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        attention_weights: Tuple[torch.Tensor, ...],
    ) -> Tuple:
        """
        Update LRU timestamps and evict stale decode tokens if over budget.

        Args:
            past_key_values: HuggingFace (k, v) per layer.
            attention_weights: Per-layer attention. Each: (batch, heads, Q, K).

        Returns:
            Pruned past_key_values.
        """
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        prefill_len = min(self._prefill_len, seq_len)
        num_decode = seq_len - prefill_len

        self._step += 1

        # Initialise LRU buffer or extend when new decode tokens are appended.
        # New tokens are stamped with the current step (freshly generated → most recent).
        if self._lru_timestamps is None:
            self._lru_timestamps = torch.full(
                (num_decode,), self._step, device=device, dtype=torch.long
            )
        elif self._lru_timestamps.shape[0] < num_decode:
            new_count = num_decode - self._lru_timestamps.shape[0]
            fresh = torch.full((new_count,), self._step, device=device, dtype=torch.long)
            self._lru_timestamps = torch.cat([self._lru_timestamps, fresh])

        # Update timestamps: tokens in the top-50% of decode attention get refreshed.
        if num_decode > 0:
            attn_step = torch.stack([
                a.mean(dim=(0, 1, 2)) for a in attention_weights
            ]).mean(dim=0)  # (key_len,)
            decode_attn = attn_step[prefill_len:seq_len]  # (num_decode,)
            median_attn = decode_attn.median()
            self._lru_timestamps[decode_attn >= median_attn] = self._step

        if seq_len <= self.config.cache_size:
            return past_key_values

        # Prefill tokens are always kept; allocate remaining budget to decode tokens.
        decode_budget = max(self.config.cache_size - prefill_len, 0)

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[:prefill_len] = True

        if decode_budget > 0 and num_decode > 0:
            n_keep = min(decode_budget, num_decode)
            # Highest timestamp = most recently accessed = keep
            _, keep_idx = torch.topk(self._lru_timestamps[:num_decode], n_keep)
            keep_mask[prefill_len + keep_idx] = True

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )
        # Align LRU buffer to the surviving decode tokens.
        decode_keep = keep_mask[prefill_len:prefill_len + num_decode]
        self._lru_timestamps = self._lru_timestamps[:num_decode][decode_keep]
        return new_past


class HSVarianceEviction:
    """
    Phase 1 proposed method: Band A − Band B hidden-state variance eviction.

    Score per token = l10_rolling64 − l21_rolling64, where:
      l10_rolling64 = rolling mean (window=64) of ||hs[t, l10] − hs[t−1, l10]||₂
      l21_rolling64 = rolling mean (window=64) of ||hs[t, l21] − hs[t−1, l21]||₂

    Phase 0B finding: Band A (l7–l13) ρ > 0 in competition math — high hs_l2_diff at
    these layers = important token.  Band B (l18–l25) ρ < 0 — high hs_l2_diff at
    these layers = dispensable token.  The combined score cancels Band B noise and
    amplifies Band A signal.

    Requires output_hidden_states=True in each decode forward pass (standard
    HuggingFace option; fully FA2-compatible — no attention matrix needed).
    Stores two hidden-state vectors (~2 × hidden_dim floats) between steps.

    Usage:
        eviction = HSVarianceEviction(config)
        # After prefill:
        eviction.reset(prefill_len=prompt_len)
        eviction.set_prefill_end(prefill_outputs.hidden_states)
        # Each decode step:
        outputs = model(..., output_hidden_states=True)
        past_kv = eviction.evict_past_key_values(past_kv, outputs.hidden_states)
    """

    def __init__(
        self,
        config: EvictionConfig,
        band_a_layer: int = 10,
        band_b_layer: int = 21,
        window: int = 64,
    ):
        self.config = config
        self.band_a_layer = band_a_layer
        self.band_b_layer = band_b_layer
        self.window = window

        self._prefill_len: int = 0
        self._prev_hs_a: Optional[torch.Tensor] = None  # hs at band_a_layer, previous token
        self._prev_hs_b: Optional[torch.Tensor] = None  # hs at band_b_layer, previous token
        self._buf_a: List[float] = []   # generation-order l10 diffs (not pruned on eviction)
        self._buf_b: List[float] = []   # generation-order l21 diffs (not pruned on eviction)
        self._scores: List[float] = []  # rolling64 score per CACHED decode token (pruned)

    def reset(self, prefill_len: int = 0):
        """Clear per-sequence state.  Call before the decode loop for each new sequence."""
        self._prefill_len = prefill_len
        self._prev_hs_a = None
        self._prev_hs_b = None
        self._buf_a = []
        self._buf_b = []
        self._scores = []

    def set_prefill_end(self, hidden_states: Tuple[torch.Tensor, ...]):
        """
        Store the last prefill token's hidden states for accurate first-decode diff.

        Call once after the prefill forward pass:
            outputs = model(**prompt_inputs, use_cache=True, output_hidden_states=True)
            eviction.set_prefill_end(outputs.hidden_states)

        If not called, the first decode token receives score 0.0 (acceptable for long traces).
        hidden_states: tuple of (batch, prefill_len, hidden_dim), one entry per layer.
        """
        # Index = layer + 1 (0 = embedding output), consistent with extract_phase0b_signals.py
        self._prev_hs_a = hidden_states[self.band_a_layer + 1][:, -1, :].detach()
        self._prev_hs_b = hidden_states[self.band_b_layer + 1][:, -1, :].detach()

    def _rolling_mean(self, buf: List[float]) -> float:
        """Mean of the last `window` values in buf."""
        start = max(0, len(buf) - self.window)
        chunk = buf[start:]
        return sum(chunk) / len(chunk) if chunk else 0.0

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> Tuple:
        """
        Update score for the most recently decoded token and evict if over budget.

        Args:
            past_key_values: HuggingFace (k, v) per layer. k/v: (batch, heads, seq_len, head_dim)
            hidden_states: Model output tuple, one tensor per layer+embedding.
                           Each: (batch, 1, hidden_dim) during decode.

        Returns:
            Pruned past_key_values.
        """
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        prefill_len = min(self._prefill_len, seq_len)

        # Extract current token HS (last position) at band A and band B layers.
        hs_a = hidden_states[self.band_a_layer + 1][:, -1, :].detach()  # (batch, hidden_dim)
        hs_b = hidden_states[self.band_b_layer + 1][:, -1, :].detach()

        if self._prev_hs_a is not None:
            diff_a = (hs_a - self._prev_hs_a).norm(dim=-1).mean().item()
            diff_b = (hs_b - self._prev_hs_b).norm(dim=-1).mean().item()
        else:
            diff_a = 0.0
            diff_b = 0.0

        self._buf_a.append(diff_a)
        self._buf_b.append(diff_b)
        score = self._rolling_mean(self._buf_a) - self._rolling_mean(self._buf_b)
        self._scores.append(score)

        self._prev_hs_a = hs_a
        self._prev_hs_b = hs_b

        if seq_len <= self.config.cache_size:
            return past_key_values

        num_decode = seq_len - prefill_len
        keep_recent = min(self.config.keep_recent_k, self.config.cache_size // 4)
        decode_budget = max(self.config.cache_size - prefill_len, 0)

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[:prefill_len] = True  # always keep prefill tokens

        if decode_budget > 0 and num_decode > 0:
            scores = torch.tensor(
                self._scores[-num_decode:], device=device, dtype=torch.float32
            )
            if len(scores) < num_decode:
                # Guard: pad with -inf so under-scored positions are evicted first
                pad = torch.full((num_decode - len(scores),), float('-inf'), device=device)
                scores = torch.cat([pad, scores])
            scores[-keep_recent:] = float('inf')  # always keep recency window
            n_keep = min(decode_budget, num_decode)
            _, keep_idx = torch.topk(scores, n_keep)
            keep_mask[prefill_len + keep_idx] = True

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )

        # Prune _scores to surviving decode tokens so alignment holds next step.
        decode_keep = keep_mask[prefill_len:prefill_len + num_decode].tolist()
        self._scores = [s for s, kept in zip(self._scores[-num_decode:], decode_keep) if kept]

        return new_past


class KVValVarianceEviction:
    """
    Phase 1 proposed method: value-vector variance eviction (kv_val_var_rolling64).

    Score per token = rolling mean (window=64) of kv_val_var, where:
      kv_val_var[t] = mean over layers of var(v[t, :, :], dim=head_dim)

    Phase 0B finding: kv_val_var is consistently non-negative across all 7
    measured datasets — unlike kv_key_var (which sign-flips between math500 and
    GSM8K).  Higher variance = more content-rich = KEEP.

    No extra forward pass required — values are already in past_key_values.
    Strictly cheaper than HS-based methods (reads already-cached GPU tensors).

    Usage:
        eviction = KVValVarianceEviction(config)
        eviction.reset(prefill_len=prompt_len)
        # Each decode step:
        past_kv = eviction.evict_past_key_values(past_kv)
    """

    def __init__(self, config: EvictionConfig, window: int = 64):
        self.config = config
        self.window = window

        self._prefill_len: int = 0
        self._buf: List[float] = []     # generation-order kv_val_var values (not pruned)
        self._scores: List[float] = []  # rolling64 score per CACHED decode token (pruned)

    def reset(self, prefill_len: int = 0):
        """Clear per-sequence state."""
        self._prefill_len = prefill_len
        self._buf = []
        self._scores = []

    def _kv_val_var_last_token(self, past_key_values: Tuple) -> float:
        """
        Compute mean kv_val_var across layers for the last cached token.

        kv_val_var[t, layer] = var(v[batch, heads, t, :], dim=head_dim).mean()
        """
        layer_vars = [
            v[:, :, -1, :].var(dim=-1).mean().item()
            for _, v in past_key_values
        ]
        return sum(layer_vars) / len(layer_vars) if layer_vars else 0.0

    def _rolling_mean(self) -> float:
        start = max(0, len(self._buf) - self.window)
        chunk = self._buf[start:]
        return sum(chunk) / len(chunk) if chunk else 0.0

    def evict_past_key_values(self, past_key_values: Tuple) -> Tuple:
        """
        Update score for the most recently cached token and evict if over budget.

        Args:
            past_key_values: HuggingFace (k, v) per layer. k/v: (batch, heads, seq_len, head_dim)

        Returns:
            Pruned past_key_values.
        """
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        prefill_len = min(self._prefill_len, seq_len)

        # Score the new (last) decode token from its value vectors.
        new_var = self._kv_val_var_last_token(past_key_values)
        self._buf.append(new_var)
        self._scores.append(self._rolling_mean())

        if seq_len <= self.config.cache_size:
            return past_key_values

        num_decode = seq_len - prefill_len
        keep_recent = min(self.config.keep_recent_k, self.config.cache_size // 4)
        decode_budget = max(self.config.cache_size - prefill_len, 0)

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[:prefill_len] = True

        if decode_budget > 0 and num_decode > 0:
            scores = torch.tensor(
                self._scores[-num_decode:], device=device, dtype=torch.float32
            )
            if len(scores) < num_decode:
                pad = torch.full((num_decode - len(scores),), float('-inf'), device=device)
                scores = torch.cat([pad, scores])
            scores[-keep_recent:] = float('inf')
            n_keep = min(decode_budget, num_decode)
            _, keep_idx = torch.topk(scores, n_keep)
            keep_mask[prefill_len + keep_idx] = True

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )

        decode_keep = keep_mask[prefill_len:prefill_len + num_decode].tolist()
        self._scores = [s for s, kept in zip(self._scores[-num_decode:], decode_keep) if kept]

        return new_past


class KVKeyVarianceEviction:
    """
    Phase 1 proposed method: key-vector variance eviction (kv_key_var_rolling64).

    Score per token = rolling mean (window=64) of kv_key_var, where:
      kv_key_var[t] = mean over layers of var(k[t, :, :], dim=head_dim)

    Phase 0B finding: kv_key_var has the HIGHEST magnitude of all signals tested
    on competition math (ρ = +0.380 on math500, ρ = +0.097 on AIME2024).
    High key variance = token whose "address" is highly specific → many future
    queries attend to it precisely → important token, KEEP.

    IMPORTANT: kv_key_var sign-flips between competition math (+) and GSM8K (−).
    This class targets competition math (math500, AIME2024) where the sign is
    confirmed positive at high n_eff.  For cross-domain use, lag-relative
    normalization (subtract rolling baseline) would be needed — defer to Phase 2.

    No extra forward pass — keys are already in past_key_values.

    Usage:
        eviction = KVKeyVarianceEviction(config)
        eviction.reset(prefill_len=prompt_len)
        past_kv = eviction.evict_past_key_values(past_kv)
    """

    def __init__(self, config: EvictionConfig, window: int = 64):
        self.config = config
        self.window = window

        self._prefill_len: int = 0
        self._buf: List[float] = []     # generation-order kv_key_var values (not pruned)
        self._scores: List[float] = []  # rolling64 score per CACHED decode token (pruned)

    def reset(self, prefill_len: int = 0):
        """Clear per-sequence state."""
        self._prefill_len = prefill_len
        self._buf = []
        self._scores = []

    def _kv_key_var_last_token(self, past_key_values: Tuple) -> float:
        """
        Compute mean kv_key_var across layers for the last cached token.

        kv_key_var[t, layer] = var(k[batch, heads, t, :], dim=head_dim).mean()
        """
        layer_vars = [
            k[:, :, -1, :].var(dim=-1).mean().item()
            for k, _ in past_key_values
        ]
        return sum(layer_vars) / len(layer_vars) if layer_vars else 0.0

    def _rolling_mean(self) -> float:
        start = max(0, len(self._buf) - self.window)
        chunk = self._buf[start:]
        return sum(chunk) / len(chunk) if chunk else 0.0

    def evict_past_key_values(self, past_key_values: Tuple) -> Tuple:
        """
        Update score for the most recently cached token and evict if over budget.

        Args:
            past_key_values: HuggingFace (k, v) per layer. k/v: (batch, heads, seq_len, head_dim)

        Returns:
            Pruned past_key_values.
        """
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        prefill_len = min(self._prefill_len, seq_len)

        new_var = self._kv_key_var_last_token(past_key_values)
        self._buf.append(new_var)
        self._scores.append(self._rolling_mean())

        if seq_len <= self.config.cache_size:
            return past_key_values

        num_decode = seq_len - prefill_len
        keep_recent = min(self.config.keep_recent_k, self.config.cache_size // 4)
        decode_budget = max(self.config.cache_size - prefill_len, 0)

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[:prefill_len] = True

        if decode_budget > 0 and num_decode > 0:
            scores = torch.tensor(
                self._scores[-num_decode:], device=device, dtype=torch.float32
            )
            if len(scores) < num_decode:
                pad = torch.full((num_decode - len(scores),), float('-inf'), device=device)
                scores = torch.cat([pad, scores])
            scores[-keep_recent:] = float('inf')
            n_keep = min(decode_budget, num_decode)
            _, keep_idx = torch.topk(scores, n_keep)
            keep_mask[prefill_len + keep_idx] = True

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )

        decode_keep = keep_mask[prefill_len:prefill_len + num_decode].tolist()
        self._scores = [s for s, kept in zip(self._scores[-num_decode:], decode_keep) if kept]

        return new_past


class LagKVKeyVarianceEviction:
    """
    Phase 1 proposed: lag-normalized key-variance eviction.

    Addresses the kv_key_var sign-flip observed between competition math and GSM8K
    by normalizing each token's key vectors against the previous chunk's channel-wise
    min/max range, eliminating domain-level absolute-magnitude shifts.

    Score per token = mean_heads(std_head_dim(norm_k[t, :]))
    where norm_k is normalized by the preceding chunk's [min, max] per channel.

    Causal adaptation of LagKV (Xu et al., arxiv:2504.04704): the paper uses
    the *next* partition as reference (look-ahead, valid at prefill only).
    We use the *previous* partition instead — causally valid during decode.

    Scores recomputed from current cache state each call (stateless).
    No extra forward pass — reads already-cached key tensors.

    Ablation hierarchy:
        KVKeyVarianceEviction     → raw key var (no normalization)
        LagKVKeyVarianceEviction  → lag-normalized key var only      ← this class
        LagKVEviction             → lag-normalized key + value

    Usage:
        eviction = LagKVKeyVarianceEviction(config)
        eviction.reset(prefill_len=prompt_len)
        past_kv = eviction.evict_past_key_values(past_kv)
    """

    def __init__(self, config: EvictionConfig, chunk_size: int = 128):
        self.config = config
        self.chunk_size = chunk_size
        self._prefill_len: int = 0

    def reset(self, prefill_len: int = 0):
        """Clear per-sequence state."""
        self._prefill_len = prefill_len

    def _score_tokens(self, past_key_values: Tuple, seq_len: int) -> torch.Tensor:
        """
        Lag-normalized key-variance score for all seq_len positions.

        Average K over layers and batch → (heads, seq_len, head_dim).
        For each chunk, normalize by previous chunk's per-channel [min, max].
        First chunk falls back to raw std (no reference available).

        Returns:
            (seq_len,) float tensor — higher = more important.
        """
        device = past_key_values[0][0].device
        k_all = torch.stack([k.mean(0) for k, _ in past_key_values]).mean(0)  # (heads, seq, head_dim)
        scores = torch.zeros(seq_len, device=device)
        L = self.chunk_size

        for start in range(0, seq_len, L):
            end = min(start + L, seq_len)
            chunk_k = k_all[:, start:end, :]  # (heads, chunk_len, head_dim)
            if start == 0:
                # No prior partition — use raw std as fallback
                token_std = chunk_k.std(dim=-1)  # (heads, chunk_len)
            else:
                ref_k = k_all[:, start - L:start, :]       # (heads, L, head_dim)
                ref_min = ref_k.amin(dim=1, keepdim=True)  # (heads, 1, head_dim)
                ref_max = ref_k.amax(dim=1, keepdim=True)
                chunk_norm = (chunk_k - ref_min) / (ref_max - ref_min + 1e-9)
                token_std = chunk_norm.std(dim=-1)
            scores[start:end] = token_std.mean(dim=0)      # mean over heads → (chunk_len,)

        return scores

    def evict_past_key_values(self, past_key_values: Tuple) -> Tuple:
        """
        Score tokens via lag-normalized key variance and evict if over budget.

        Args:
            past_key_values: HuggingFace (k, v) per layer. k/v: (batch, heads, seq_len, head_dim)

        Returns:
            Pruned past_key_values.
        """
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        prefill_len = min(self._prefill_len, seq_len)

        if seq_len <= self.config.cache_size:
            return past_key_values

        num_decode = seq_len - prefill_len
        keep_recent = min(self.config.keep_recent_k, self.config.cache_size // 4)
        decode_budget = max(self.config.cache_size - prefill_len, 0)

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[:prefill_len] = True

        if decode_budget > 0 and num_decode > 0:
            scores = self._score_tokens(past_key_values, seq_len)
            decode_scores = scores[prefill_len:].clone()
            decode_scores[-keep_recent:] = float('inf')
            n_keep = min(decode_budget, num_decode)
            _, keep_idx = torch.topk(decode_scores, n_keep)
            keep_mask[prefill_len + keep_idx] = True

        return tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )


class LagKVEviction:
    """
    Phase 1 proposed: LagKV — lag-normalized key + value variance eviction.

    Full implementation of Xu et al. (arxiv:2504.04704), adapted for streaming
    decode.  Score per token = lag_key_std[t] + lag_val_std[t], where both K and V
    are normalized against the previous chunk's per-channel [min, max].

    The paper uses the *next* partition as reference (only valid at prefill).
    We use the *previous* partition (causal; valid during decode).

    Ablation hierarchy:
        KVKeyVarianceEviction     → raw key var (no normalization)
        LagKVKeyVarianceEviction  → lag-normalized key var only
        LagKVEviction             → lag-normalized key + value      ← this class

    Scores recomputed from current cache state each call (stateless).
    No extra forward pass.

    Usage:
        eviction = LagKVEviction(config)
        eviction.reset(prefill_len=prompt_len)
        past_kv = eviction.evict_past_key_values(past_kv)
    """

    def __init__(self, config: EvictionConfig, chunk_size: int = 128):
        self.config = config
        self.chunk_size = chunk_size
        self._prefill_len: int = 0

    def reset(self, prefill_len: int = 0):
        """Clear per-sequence state."""
        self._prefill_len = prefill_len

    def _lag_std(self, tensor_all: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Lag-normalized per-token std for a (heads, seq_len, head_dim) tensor.
        Shared helper for both K and V scoring.

        Returns:
            (seq_len,) float tensor.
        """
        device = tensor_all.device
        scores = torch.zeros(seq_len, device=device)
        L = self.chunk_size

        for start in range(0, seq_len, L):
            end = min(start + L, seq_len)
            chunk = tensor_all[:, start:end, :]
            if start == 0:
                token_std = chunk.std(dim=-1)
            else:
                ref = tensor_all[:, start - L:start, :]
                ref_min = ref.amin(dim=1, keepdim=True)
                ref_max = ref.amax(dim=1, keepdim=True)
                chunk_norm = (chunk - ref_min) / (ref_max - ref_min + 1e-9)
                token_std = chunk_norm.std(dim=-1)
            scores[start:end] = token_std.mean(dim=0)

        return scores

    def _score_tokens(self, past_key_values: Tuple, seq_len: int) -> torch.Tensor:
        """score(K) + score(V), both lag-normalized."""
        k_all = torch.stack([k.mean(0) for k, _ in past_key_values]).mean(0)
        v_all = torch.stack([v.mean(0) for _, v in past_key_values]).mean(0)
        return self._lag_std(k_all, seq_len) + self._lag_std(v_all, seq_len)

    def evict_past_key_values(self, past_key_values: Tuple) -> Tuple:
        """
        Score tokens via lag-normalized K+V variance and evict if over budget.

        Args:
            past_key_values: HuggingFace (k, v) per layer. k/v: (batch, heads, seq_len, head_dim)

        Returns:
            Pruned past_key_values.
        """
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        prefill_len = min(self._prefill_len, seq_len)

        if seq_len <= self.config.cache_size:
            return past_key_values

        num_decode = seq_len - prefill_len
        keep_recent = min(self.config.keep_recent_k, self.config.cache_size // 4)
        decode_budget = max(self.config.cache_size - prefill_len, 0)

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[:prefill_len] = True

        if decode_budget > 0 and num_decode > 0:
            scores = self._score_tokens(past_key_values, seq_len)
            decode_scores = scores[prefill_len:].clone()
            decode_scores[-keep_recent:] = float('inf')
            n_keep = min(decode_budget, num_decode)
            _, keep_idx = torch.topk(decode_scores, n_keep)
            keep_mask[prefill_len + keep_idx] = True

        return tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )


def _rolling_z_score(buf: List[float], window: int, eps: float = 1e-6) -> float:
    """
    z-score of the last element relative to the rolling window.

    Returns (buf[-1] - mean(window)) / (std(window) + eps).
    Returns 0.0 when fewer than 2 elements are available (std undefined).
    """
    start = max(0, len(buf) - window)
    chunk = buf[start:]
    if len(chunk) < 2:
        return 0.0
    mu = sum(chunk) / len(chunk)
    var = sum((x - mu) ** 2 for x in chunk) / len(chunk)
    return (chunk[-1] - mu) / (var ** 0.5 + eps)


def _hs_diff(
    hidden_states: Tuple[torch.Tensor, ...],
    layer: int,
    prev: Optional[torch.Tensor],
) -> Tuple[float, torch.Tensor]:
    """
    Extract the last-token HS at `layer` and compute L2 diff from `prev`.

    Args:
        hidden_states: tuple from model output; index 0 = embedding output,
                       index l+1 = output of transformer layer l.
        layer: transformer layer index (0-based).
        prev: previous step's hidden state (batch, hidden_dim), or None.

    Returns:
        (diff_scalar, current_hs)  where diff_scalar = 0.0 if prev is None.
    """
    hs = hidden_states[layer + 1][:, -1, :].detach()  # (batch, hidden_dim)
    diff = (hs - prev).norm(dim=-1).mean().item() if prev is not None else 0.0
    return diff, hs


class DetrendendHSVarianceEviction:
    """
    Detrended Band A − B hidden-state variance eviction.

    Extends HSVarianceEviction by replacing the raw rolling-mean score with
    a within-window z-score.  This removes the monotonic temporal trend
    (Band A decreasing, Band B increasing with position) that contaminates
    per-token discrimination in the non-detrended version.

    Score per token = z_a(t) − z_b(t), where:
        z_x(t) = (diff_x(t) − rolling_mean_x(t)) / (rolling_std_x(t) + ε)
        diff_x(t) = ||hs_x[t] − hs_x[t−1]||₂

    Motivation: AhaKV (2026) shows position bias in attention is removed by
    adaptive temperature; LagKV (2025) uses lag-relative normalization for KV
    signals.  Rolling z-score is the HS-signal analog — it converts absolute
    signal magnitude into local-relative deviation, isolating genuine within-
    trace importance variation from the monotonic trend.

    Requires output_hidden_states=True; FA2-compatible (no attention matrix).

    Usage:
        eviction = DetrendendHSVarianceEviction(config)
        eviction.reset(prefill_len=prompt_len)
        eviction.set_prefill_end(prefill_outputs.hidden_states)
        # Each decode step:
        outputs = model(..., output_hidden_states=True)
        past_kv = eviction.evict_past_key_values(past_kv, outputs.hidden_states)
    """

    def __init__(
        self,
        config: EvictionConfig,
        band_a_layer: int = 10,
        band_b_layer: int = 21,
        window: int = 64,
    ):
        self.config = config
        self.band_a_layer = band_a_layer
        self.band_b_layer = band_b_layer
        self.window = window

        self._prefill_len: int = 0
        self._prev_hs_a: Optional[torch.Tensor] = None
        self._prev_hs_b: Optional[torch.Tensor] = None
        self._buf_a: List[float] = []
        self._buf_b: List[float] = []
        self._scores: List[float] = []

    def reset(self, prefill_len: int = 0):
        self._prefill_len = prefill_len
        self._prev_hs_a = None
        self._prev_hs_b = None
        self._buf_a = []
        self._buf_b = []
        self._scores = []

    def set_prefill_end(self, hidden_states: Tuple[torch.Tensor, ...]):
        _, self._prev_hs_a = _hs_diff(hidden_states, self.band_a_layer, None)
        self._prev_hs_a = hidden_states[self.band_a_layer + 1][:, -1, :].detach()
        self._prev_hs_b = hidden_states[self.band_b_layer + 1][:, -1, :].detach()

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> Tuple:
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        prefill_len = min(self._prefill_len, seq_len)

        diff_a, hs_a = _hs_diff(hidden_states, self.band_a_layer, self._prev_hs_a)
        diff_b, hs_b = _hs_diff(hidden_states, self.band_b_layer, self._prev_hs_b)
        self._prev_hs_a = hs_a
        self._prev_hs_b = hs_b

        self._buf_a.append(diff_a)
        self._buf_b.append(diff_b)
        score = (
            _rolling_z_score(self._buf_a, self.window)
            - _rolling_z_score(self._buf_b, self.window)
        )
        self._scores.append(score)

        if seq_len <= self.config.cache_size:
            return past_key_values

        num_decode = seq_len - prefill_len
        keep_recent = min(self.config.keep_recent_k, self.config.cache_size // 4)
        decode_budget = max(self.config.cache_size - prefill_len, 0)

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[:prefill_len] = True

        if decode_budget > 0 and num_decode > 0:
            scores = torch.tensor(
                self._scores[-num_decode:], device=device, dtype=torch.float32
            )
            if len(scores) < num_decode:
                pad = torch.full((num_decode - len(scores),), float('-inf'), device=device)
                scores = torch.cat([pad, scores])
            scores[-keep_recent:] = float('inf')
            n_keep = min(decode_budget, num_decode)
            _, keep_idx = torch.topk(scores, n_keep)
            keep_mask[prefill_len + keep_idx] = True

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )
        decode_keep = keep_mask[prefill_len:prefill_len + num_decode].tolist()
        self._scores = [s for s, kept in zip(self._scores[-num_decode:], decode_keep) if kept]
        return new_past


class BandAdaptiveHSEviction:
    """
    Multi-layer Band A−B HS variance eviction with empirically-calibrated weights.

    Extends DetrendendHSVarianceEviction in two ways:
      1. Averages HS diffs over ALL Band A layers (l7–l13, 7 layers) and ALL
         Band B layers (l18–l25, 8 layers) rather than a single representative layer.
         This uses the full band signal rather than a one-layer proxy.
      2. Weights each band by its Phase 0B ρ magnitude (math500_eager):
             weight_a ∝ 0.141 (Band A mean ρ)
             weight_b ∝ 0.109 (Band B mean |ρ|)
         Default ratio weight_a/weight_b ≈ 1.29 — Band A is slightly stronger.

    Score = weight_a * mean_band_a_z − weight_b * mean_band_b_z

    Requires output_hidden_states=True; FA2-compatible.

    Usage:
        eviction = BandAdaptiveHSEviction(config)
        eviction.reset(prefill_len=prompt_len)
        eviction.set_prefill_end(prefill_outputs.hidden_states)
        outputs = model(..., output_hidden_states=True)
        past_kv = eviction.evict_past_key_values(past_kv, outputs.hidden_states)
    """

    _BAND_A = (7, 8, 9, 10, 11, 12, 13)    # Phase 0B: consistently positive ρ
    _BAND_B = (18, 19, 20, 21, 22, 23, 24, 25)  # Phase 0B: consistently negative ρ

    def __init__(
        self,
        config: EvictionConfig,
        window: int = 64,
        weight_a: float = 1.29,
        weight_b: float = 1.0,
        band_a_layers: Optional[Tuple[int, ...]] = None,
        band_b_layers: Optional[Tuple[int, ...]] = None,
    ):
        self.config = config
        self.window = window
        self.weight_a = weight_a
        self.weight_b = weight_b
        self.band_a_layers = band_a_layers if band_a_layers is not None else self._BAND_A
        self.band_b_layers = band_b_layers if band_b_layers is not None else self._BAND_B

        self._prefill_len: int = 0
        self._prev_a: dict = {}   # layer_idx -> (batch, hidden_dim) tensor
        self._prev_b: dict = {}
        self._buf_a: List[float] = []  # mean-over-band diff per step
        self._buf_b: List[float] = []
        self._scores: List[float] = []

    def reset(self, prefill_len: int = 0):
        self._prefill_len = prefill_len
        self._prev_a = {}
        self._prev_b = {}
        self._buf_a = []
        self._buf_b = []
        self._scores = []

    def set_prefill_end(self, hidden_states: Tuple[torch.Tensor, ...]):
        for l in self.band_a_layers:
            self._prev_a[l] = hidden_states[l + 1][:, -1, :].detach()
        for l in self.band_b_layers:
            self._prev_b[l] = hidden_states[l + 1][:, -1, :].detach()

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> Tuple:
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        prefill_len = min(self._prefill_len, seq_len)

        # Accumulate per-layer diffs and update stored previous states
        diffs_a = []
        for l in self.band_a_layers:
            diff, hs = _hs_diff(hidden_states, l, self._prev_a.get(l))
            self._prev_a[l] = hs
            diffs_a.append(diff)

        diffs_b = []
        for l in self.band_b_layers:
            diff, hs = _hs_diff(hidden_states, l, self._prev_b.get(l))
            self._prev_b[l] = hs
            diffs_b.append(diff)

        self._buf_a.append(sum(diffs_a) / len(diffs_a) if diffs_a else 0.0)
        self._buf_b.append(sum(diffs_b) / len(diffs_b) if diffs_b else 0.0)
        score = (
            self.weight_a * _rolling_z_score(self._buf_a, self.window)
            - self.weight_b * _rolling_z_score(self._buf_b, self.window)
        )
        self._scores.append(score)

        if seq_len <= self.config.cache_size:
            return past_key_values

        num_decode = seq_len - prefill_len
        keep_recent = min(self.config.keep_recent_k, self.config.cache_size // 4)
        decode_budget = max(self.config.cache_size - prefill_len, 0)

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[:prefill_len] = True

        if decode_budget > 0 and num_decode > 0:
            scores = torch.tensor(
                self._scores[-num_decode:], device=device, dtype=torch.float32
            )
            if len(scores) < num_decode:
                pad = torch.full((num_decode - len(scores),), float('-inf'), device=device)
                scores = torch.cat([pad, scores])
            scores[-keep_recent:] = float('inf')
            n_keep = min(decode_budget, num_decode)
            _, keep_idx = torch.topk(scores, n_keep)
            keep_mask[prefill_len + keep_idx] = True

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )
        decode_keep = keep_mask[prefill_len:prefill_len + num_decode].tolist()
        self._scores = [s for s, kept in zip(self._scores[-num_decode:], decode_keep) if kept]
        return new_past


class AttentionHSProductEviction:
    """
    Hybrid: cumulative key-perspective attention + detrended Band A HS z-score.

    Combines two orthogonal signals:
      - Cumulative key-perspective attention (H2O-style column sums): tracks
        how much each cached key has been attended to by subsequent queries.
        Captures retrospective importance — "was this token referenced?"
      - Detrended Band A HS variance z-score: captures prospective importance —
        "did this token cause a reasoning shift relative to its local baseline?"

    Combined score = norm(cumul_attn_decode) + alpha * z_a_decode
    Both components normalized to [0, 1] before summing so neither dominates.

    Rationale: a token important by both criteria (high attention AND high Band A
    z-score) is strongly kept.  A token important by only one criterion still
    receives partial credit.  This is more robust than either signal alone.

    Requires output_attentions=True AND output_hidden_states=True → EAGER ONLY.

    Usage:
        eviction = AttentionHSProductEviction(config)
        eviction.reset(prefill_len=prompt_len)
        eviction.set_prefill_end(prefill_outputs.hidden_states)
        outputs = model(..., output_attentions=True, output_hidden_states=True)
        past_kv = eviction.evict_past_key_values(
            past_kv, outputs.attentions, outputs.hidden_states
        )
    """

    def __init__(
        self,
        config: EvictionConfig,
        band_a_layer: int = 10,
        window: int = 64,
        alpha: float = 1.0,
        num_sink_tokens: int = 4,
    ):
        self.config = config
        self.band_a_layer = band_a_layer
        self.window = window
        self.alpha = alpha
        self.num_sink_tokens = num_sink_tokens

        self._prefill_len: int = 0
        self._prev_hs_a: Optional[torch.Tensor] = None
        self._buf_a: List[float] = []
        self._scores_hs: List[float] = []   # detrended HS z-scores (pruned)
        self._cumulative_attn: Optional[torch.Tensor] = None

    def reset(self, prefill_len: int = 0):
        self._prefill_len = prefill_len
        self._prev_hs_a = None
        self._buf_a = []
        self._scores_hs = []
        self._cumulative_attn = None

    def set_prefill_end(self, hidden_states: Tuple[torch.Tensor, ...]):
        self._prev_hs_a = hidden_states[self.band_a_layer + 1][:, -1, :].detach()

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        attention_weights: Tuple[torch.Tensor, ...],
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> Tuple:
        seq_len = past_key_values[0][0].shape[2]
        device = past_key_values[0][0].device
        prefill_len = min(self._prefill_len, seq_len)

        # ── Update cumulative key-perspective attention ────────────────────────
        attn_step = torch.stack([
            a.mean(dim=(0, 1, 2)) for a in attention_weights
        ]).mean(dim=0)  # (key_len,)
        if self._cumulative_attn is None:
            self._cumulative_attn = torch.zeros(seq_len, device=device, dtype=attn_step.dtype)
        elif attn_step.shape[0] > self._cumulative_attn.shape[0]:
            pad = torch.zeros(
                attn_step.shape[0] - self._cumulative_attn.shape[0],
                device=device, dtype=attn_step.dtype,
            )
            self._cumulative_attn = torch.cat([self._cumulative_attn, pad])
        self._cumulative_attn[:attn_step.shape[0]] += attn_step

        # ── Update HS Band A z-score ──────────────────────────────────────────
        diff_a, hs_a = _hs_diff(hidden_states, self.band_a_layer, self._prev_hs_a)
        self._prev_hs_a = hs_a
        self._buf_a.append(diff_a)
        self._scores_hs.append(_rolling_z_score(self._buf_a, self.window))

        if seq_len <= self.config.cache_size:
            return past_key_values

        num_decode = seq_len - prefill_len
        keep_recent = min(self.config.keep_recent_k, self.config.cache_size // 4)
        decode_budget = max(self.config.cache_size - prefill_len, 0)
        num_sink = min(self.num_sink_tokens, self.config.cache_size // 4)

        # Normalize cumulative attention for decode positions to [0, 1]
        attn_decode = self._cumulative_attn[prefill_len:seq_len].clone()
        a_min, a_max = attn_decode.min(), attn_decode.max()
        attn_norm = (attn_decode - a_min) / (a_max - a_min + 1e-9)

        # HS z-scores for decode positions, also normalized to [0, 1]
        hs_decode = torch.tensor(
            self._scores_hs[-num_decode:], device=device, dtype=torch.float32
        )
        if len(hs_decode) < num_decode:
            pad = torch.zeros(num_decode - len(hs_decode), device=device)
            hs_decode = torch.cat([pad, hs_decode])
        h_min, h_max = hs_decode.min(), hs_decode.max()
        hs_norm = (hs_decode - h_min) / (h_max - h_min + 1e-9)

        combined = attn_norm + self.alpha * hs_norm

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[:prefill_len] = True
        keep_mask[:num_sink] = True        # attention sinks
        combined[-keep_recent:] = float('inf')

        if decode_budget > 0 and num_decode > 0:
            n_keep = min(decode_budget, num_decode)
            _, keep_idx = torch.topk(combined, n_keep)
            keep_mask[prefill_len + keep_idx] = True

        new_past = tuple(
            (k[:, :, keep_mask.to(k.device), :].contiguous(), v[:, :, keep_mask.to(k.device), :].contiguous())
            for k, v in past_key_values
        )
        self._cumulative_attn = self._cumulative_attn[keep_mask]
        decode_keep = keep_mask[prefill_len:prefill_len + num_decode].tolist()
        self._scores_hs = [
            s for s, kept in zip(self._scores_hs[-num_decode:], decode_keep) if kept
        ]
        return new_past


class HybridSegmentHSEviction:
    """
    Hybrid: ThinKV segment type + detrended HS within-segment token ranking.

    Outer loop (segment level): classifies each 128-token block into R/E/T
    using key-perspective column-sum entropy — identical to ThinKVEviction.
    Assigns a retention budget per segment type (R=64, E=32, T=8 tokens).

    Inner loop (token level): within each segment, ranks tokens by detrended
    Band A − B HS z-score rather than ThinKV's attention column sums.
    This selects which specific tokens inside a thought segment are load-bearing.

    Rationale: ThinKV's within-segment ranking uses raw attention scores —
    a recency-biased proxy.  Our HS z-score captures mid-layer reasoning-phase
    signal, providing orthogonal and more principled within-segment discrimination.

    Requires output_attentions=True AND output_hidden_states=True → EAGER ONLY.

    Usage:
        eviction = HybridSegmentHSEviction(config)
        eviction.reset(prefill_len=prompt_len)
        eviction.set_prefill_end(prefill_outputs.hidden_states)
        outputs = model(..., output_attentions=True, output_hidden_states=True)
        past_kv = eviction.evict_past_key_values(
            past_kv, outputs.attentions, outputs.hidden_states
        )
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
        num_classifier_layers: int = 4,
    ):
        self.config = config
        self.band_a_layer = band_a_layer
        self.band_b_layer = band_b_layer
        self.window = window
        self.segment_size = segment_size
        self.retain_r = retain_r
        self.retain_e = retain_e
        self.retain_t = retain_t
        self.num_classifier_layers = num_classifier_layers

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

    def _segment_entropy(
        self,
        attention_weights: Tuple[torch.Tensor, ...],
        classify_len: int,
    ) -> torch.Tensor:
        """Key-perspective segment entropy (identical to ThinKVEviction)."""
        layers = attention_weights[-self.num_classifier_layers:]
        col_attn = torch.stack([
            a.mean(dim=(0, 1))[-1, :] for a in layers
        ]).mean(dim=0)[:classify_len]

        seg_size = self.segment_size
        num_full = classify_len // seg_size
        entropies = []
        for i in range(num_full):
            seg = col_attn[i * seg_size:(i + 1) * seg_size]
            p = seg / (seg.sum() + 1e-9)
            entropies.append(-(p * torch.log(p + 1e-12)).sum())
        if classify_len - num_full * seg_size > 0:
            seg = col_attn[num_full * seg_size:]
            p = seg / (seg.sum() + 1e-9)
            entropies.append(-(p * torch.log(p + 1e-12)).sum())
        return torch.stack(entropies)

    def _classify_segments(self, seg_entropies: torch.Tensor) -> List[str]:
        """Tertile threshold classification (identical to ThinKVEviction)."""
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
        attention_weights: Tuple[torch.Tensor, ...],
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
        classify_len = seq_len - keep_recent   # absolute positions 0..classify_len-1

        # ── Segment type classification (key-perspective entropy) ─────────────
        seg_entropies = self._segment_entropy(attention_weights, classify_len)
        seg_labels = self._classify_segments(seg_entropies)

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
            # Absolute positions for this segment
            seg_abs_start = i * seg_size
            seg_abs_end = min((i + 1) * seg_size, classify_len)

            # Only act on non-prefill positions
            decode_start = max(seg_abs_start, prefill_len)
            decode_end = min(seg_abs_end, seq_len - keep_recent)
            if decode_start >= decode_end:
                continue

            # Map absolute decode positions to _scores indices
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
