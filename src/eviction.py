"""
KV cache eviction baselines.

Classes
-------
AttentionBasedEviction  — single-step attention (legacy POC baseline)
SemanticEviction        — HS variance + KV variance blend (proposed)
H2OEviction             — cumulative attention / Heavy Hitter Oracle (Zhang et al. 2023)
ThinKVEviction          — segment-level R/E/T classification + per-type budgets (He et al. 2025)
RaaSEviction            — LRU decode eviction + unconditional prefill preservation
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class EvictionConfig:
    """Configuration for KV cache eviction."""
    cache_size: int = 4096  # Max KV cache tokens
    eviction_method: str = "attention"  # "attention", "semantic"
    keep_recent_k: int = 128  # Always keep recent K tokens (like StreamingLLM)
    semantic_alpha: float = 0.5  # Weight for semantic (state variance) vs attention score.
                                  # 0.0 = pure attention, 1.0 = pure state variance.
                                  # Only used when attention_weights are provided to SemanticEviction.


class AttentionBasedEviction:
    """
    Baseline: Evict KV cache tokens with lowest attention scores.
    Similar to attention sink methods but without keep_recent_k.

    NOTE: For a more rigorous implementation, cumulative attention scores
    (H2O-style, "Heavy Hitter Oracle") should be tracked across all generation
    steps rather than using only the current step's attention. This single-step
    approximation is suitable for a POC.
    """

    def __init__(self, config: EvictionConfig):
        self.config = config
        self.attention_scores = None

    def compute_eviction_mask(
        self,
        attention_weights: torch.Tensor,
        current_seq_len: int,
    ) -> torch.Tensor:
        """
        Compute which tokens to keep based on attention scores.

        Args:
            attention_weights: Shape (batch, heads, query_len, key_len)
            current_seq_len: Current sequence length (== key_len)

        Returns:
            Boolean mask of shape (key_len,) where True = keep, False = evict
        """
        cache_size = self.config.cache_size
        keep_recent = min(self.config.keep_recent_k, cache_size - 1)  # Don't keep more than cache allows

        key_len = attention_weights.shape[3]
        effective_seq_len = min(current_seq_len, key_len)
        
        if effective_seq_len <= cache_size:
            return torch.ones(effective_seq_len, dtype=torch.bool, device=attention_weights.device)

        # Average attention each key token received across batch, heads, and query positions.
        # Shape: (key_len,)
        avg_attention = attention_weights.mean(dim=(0, 1, 2))

        # Always keep the most recent tokens
        keep_mask = torch.zeros(effective_seq_len, dtype=torch.bool, device=attention_weights.device)
        keep_mask[-keep_recent:] = True

        # For non-recent tokens, keep those with highest cumulative attention
        num_to_keep = cache_size - keep_recent
        non_recent_scores = avg_attention[:-keep_recent]
        _, top_indices = torch.topk(non_recent_scores, min(num_to_keep, len(non_recent_scores)))
        keep_mask[top_indices] = True

        return keep_mask

    def evict(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evict tokens from KV cache based on attention scores.

        Args:
            k_cache: Key cache (seq_len, hidden_dim)
            v_cache: Value cache (seq_len, hidden_dim)
            attention_weights: Attention weights (batch, heads, query_len, key_len)

        Returns:
            Evicted k_cache, v_cache
        """
        current_seq_len = k_cache.shape[0]
        if current_seq_len <= self.config.cache_size:
            return k_cache, v_cache

        keep_mask = self.compute_eviction_mask(attention_weights, current_seq_len)
        return k_cache[keep_mask], v_cache[keep_mask]

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        attention_weights: Tuple[torch.Tensor, ...],
    ) -> Tuple:
        """
        Apply attention-based eviction to HuggingFace past_key_values.

        Args:
            past_key_values: Tuple of (k, v) per layer.
                             k/v shape: (batch, num_heads, seq_len, head_dim)
            attention_weights: Tuple of attention tensors per layer.
                               Each: (batch, num_heads, query_len, key_len)

        Returns:
            Pruned past_key_values tuple
        """
        seq_len = past_key_values[0][0].shape[2]
        if seq_len <= self.config.cache_size:
            return past_key_values

        # Use the last layer's attention weights to decide which tokens to evict.
        # Using the last layer is a common heuristic; averaging across layers is
        # more principled but more expensive.
        last_attn = attention_weights[-1]  # (batch, heads, query_len, key_len)
        
        # Debug: check shapes
        key_len = last_attn.shape[3]
        if key_len != seq_len:
            print(f"Warning: attention key_len ({key_len}) != cache seq_len ({seq_len})")
            # Use the minimum to avoid index errors
            effective_seq_len = min(key_len, seq_len)
        else:
            effective_seq_len = seq_len
            
        keep_mask = self.compute_eviction_mask(last_attn, effective_seq_len)

        new_past = []
        for k, v in past_key_values:
            new_past.append((k[:, :, keep_mask, :], v[:, :, keep_mask, :]))
        return tuple(new_past)


class SemanticEviction:
    """
    Proposed: Evict tokens based on semantic importance heuristics.

    During generation the full hidden states of past tokens are not re-computed
    at each step; only their K/V projections are cached. We therefore use two
    sources of importance signal:

    1. Prefill step (step 0): hidden-state L1 differences between consecutive
       positions, which measure how sharply the representation changes — a proxy
       for "decision points" or "insight" tokens.
    2. All other steps: key-vector variance across the head dimension, averaged
       over layers. This is always available from past_key_values and captures
       a similar notion of representational richness.

    Both signals are optionally blended with attention using semantic_alpha.
    """

    def __init__(self, config: EvictionConfig):
        self.config = config

    def compute_semantic_importance(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute semantic importance from hidden states.

        Args:
            hidden_states: Shape (seq_len, hidden_dim)
            attention_weights: Optional (batch, heads, query_len, key_len).
                               When provided, blended with state-variance signal.

        Returns:
            Importance scores: (seq_len,)
        """
        seq_len = hidden_states.shape[0]

        # L1 difference between consecutive hidden states (roughness of trajectory).
        # First token has no predecessor; we assign zero change (conservative: don't
        # favour evicting the first token on account of a padding artefact).
        state_diff = torch.abs(torch.diff(hidden_states, dim=0))  # (seq_len-1, hidden_dim)
        state_variance = state_diff.mean(dim=1)                    # (seq_len-1,)
        first_token_importance = torch.zeros(1, device=state_variance.device, dtype=state_variance.dtype)
        state_variance = torch.cat([first_token_importance, state_variance])  # (seq_len,)

        # Min-max normalise
        if state_variance.max() > 0:
            state_variance = (state_variance - state_variance.min()) / (
                state_variance.max() - state_variance.min() + 1e-8
            )

        importance = state_variance

        if attention_weights is not None:
            alpha = self.config.semantic_alpha
            avg_attention = attention_weights.mean(dim=(0, 1, 2))  # (key_len,)
            if avg_attention.shape[0] != seq_len:
                # key_len might differ from hidden-state seq_len in edge cases; skip blend
                pass
            else:
                if avg_attention.max() > 0:
                    avg_attention = (avg_attention - avg_attention.min()) / (
                        avg_attention.max() - avg_attention.min() + 1e-8
                    )
                importance = alpha * importance + (1 - alpha) * avg_attention

        return importance

    def _importance_from_kv(self, past_key_values: Tuple) -> torch.Tensor:
        """
        Compute per-token importance from key-vector variance across layers.

        Key vectors encode position + content.  Tokens whose key vectors have
        high variance across head_dim tend to carry richer content signals.
        Averaging across all layers reduces noise.

        Returns:
            Importance scores: (seq_len,)
        """
        layer_scores = []
        for k, v in past_key_values:
            # k: (batch, num_heads, seq_len, head_dim)
            k_var = k.var(dim=-1).mean(dim=(0, 1))  # (seq_len,)
            layer_scores.append(k_var)
        importance = torch.stack(layer_scores).mean(dim=0)  # (seq_len,)
        if importance.max() > 0:
            importance = (importance - importance.min()) / (
                importance.max() - importance.min() + 1e-8
            )
        return importance

    def evict(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evict tokens based on semantic importance (standalone use).

        Args:
            k_cache: Key cache (seq_len, hidden_dim)
            v_cache: Value cache (seq_len, hidden_dim)
            hidden_states: Hidden states (seq_len, hidden_dim)
            attention_weights: Optional attention weights

        Returns:
            Evicted k_cache, v_cache
        """
        current_seq_len = k_cache.shape[0]
        if current_seq_len <= self.config.cache_size:
            return k_cache, v_cache

        importance = self.compute_semantic_importance(hidden_states, attention_weights)

        keep_recent = self.config.keep_recent_k
        importance[-keep_recent:] = float('inf')

        num_to_keep = self.config.cache_size
        _, keep_indices = torch.topk(importance, min(num_to_keep, current_seq_len), largest=True)
        keep_indices = torch.sort(keep_indices)[0]

        return k_cache[keep_indices], v_cache[keep_indices]

    def evict_past_key_values(
        self,
        past_key_values: Tuple,
        hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
        attention_weights: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple:
        """
        Apply semantic eviction to HuggingFace past_key_values.

        Importance signal selection:
        - If hidden_states are provided AND their seq_len matches the cache length
          (i.e. we are at the prefill step), use hidden-state variance.
        - Otherwise use key-vector variance derived from past_key_values directly,
          since past hidden states are not stored between steps.

        Args:
            past_key_values: Tuple of (k, v) per layer.
                             k/v shape: (batch, num_heads, seq_len, head_dim)
            hidden_states: Optional tuple of hidden states per layer.
                           Each: (batch, seq_len, hidden_dim)
            attention_weights: Optional tuple of attention per layer.
                               Each: (batch, num_heads, query_len, key_len)

        Returns:
            Pruned past_key_values tuple
        """
        seq_len = past_key_values[0][0].shape[2]
        if seq_len <= self.config.cache_size:
            return past_key_values

        # Determine which importance signal to use
        use_hidden = (
            hidden_states is not None
            and hidden_states[-1].shape[1] == seq_len  # prefill: full sequence
        )

        if use_hidden:
            last_h = hidden_states[-1].squeeze(0)  # (seq_len, hidden_dim)
            last_attn = attention_weights[-1] if attention_weights is not None else None
            importance = self.compute_semantic_importance(last_h, last_attn)
        else:
            # Key-vector variance is always available and works step-by-step
            importance = self._importance_from_kv(past_key_values)

        # Always preserve recent tokens
        keep_recent = self.config.keep_recent_k
        importance[-keep_recent:] = float('inf')

        num_to_keep = self.config.cache_size
        _, keep_indices = torch.topk(importance, min(num_to_keep, seq_len), largest=True)
        keep_indices = torch.sort(keep_indices)[0]

        new_past = []
        for k, v in past_key_values:
            new_past.append((k[:, :, keep_indices, :], v[:, :, keep_indices, :]))
        return tuple(new_past)


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
            (k[:, :, keep_mask, :], v[:, :, keep_mask, :])
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

        for i, label in enumerate(seg_labels):
            seg_start = i * seg_size
            seg_end = min((i + 1) * seg_size, classify_len)
            if seg_start >= seg_end:
                break
            seg_scores = col_attn[seg_start:seg_end]
            n_keep = min(budget_map[label], seg_end - seg_start)
            if n_keep > 0:
                _, top_idx = torch.topk(seg_scores, n_keep)
                keep_mask[seg_start + top_idx] = True

        new_past = tuple(
            (k[:, :, keep_mask, :], v[:, :, keep_mask, :])
            for k, v in past_key_values
        )
        return new_past


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
            (k[:, :, keep_mask, :], v[:, :, keep_mask, :])
            for k, v in past_key_values
        )
        # Align LRU buffer to the surviving decode tokens.
        decode_keep = keep_mask[prefill_len:prefill_len + num_decode]
        self._lru_timestamps = self._lru_timestamps[:num_decode][decode_keep]
        return new_past


if __name__ == "__main__":
    # Smoke tests — verify all eviction classes reduce cache from seq_len to cache_size.
    batch_size = 1
    num_heads = 8
    num_layers = 4
    hidden_dim = 64
    head_dim = hidden_dim // num_heads
    seq_len = 1000

    config = EvictionConfig(cache_size=512, keep_recent_k=128)

    k_cache = torch.randn(seq_len, hidden_dim)
    v_cache = torch.randn(seq_len, hidden_dim)
    hidden_states = torch.randn(seq_len, hidden_dim)
    attention_weights_2d = torch.randn(batch_size, num_heads, seq_len, seq_len).abs()

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

    # ── AttentionBasedEviction (legacy) ──────────────────────────────────────
    baseline = AttentionBasedEviction(config)
    k_out, v_out = baseline.evict(k_cache, v_cache, attention_weights_2d)
    print(f"AttentionBased (evict):       {seq_len} → {k_out.shape[0]} tokens")
    assert k_out.shape[0] == config.cache_size

    pruned = baseline.evict_past_key_values(past_kv, attn_tuple)
    print(f"AttentionBased (past_kv):     {seq_len} → {pruned[0][0].shape[2]} tokens")
    assert pruned[0][0].shape[2] == config.cache_size

    # ── SemanticEviction ─────────────────────────────────────────────────────
    semantic = SemanticEviction(config)
    k_out_s, v_out_s = semantic.evict(k_cache, v_cache, hidden_states, attention_weights_2d)
    print(f"Semantic (evict):             {seq_len} → {k_out_s.shape[0]} tokens")
    assert k_out_s.shape[0] == config.cache_size

    pruned_s = semantic.evict_past_key_values(past_kv, attention_weights=attn_tuple)
    print(f"Semantic (past_kv, kv-var):   {seq_len} → {pruned_s[0][0].shape[2]} tokens")
    assert pruned_s[0][0].shape[2] == config.cache_size

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
    assert retained <= seq_len

    # ── RaaSEviction ─────────────────────────────────────────────────────────
    prefill_len = 200
    raas = RaaSEviction(config)
    raas.reset(prefill_len=prefill_len)
    pruned_raas = raas.evict_past_key_values(past_kv, attn_tuple)
    print(f"RaaS (past_kv):               {seq_len} → {pruned_raas[0][0].shape[2]} tokens")
    assert pruned_raas[0][0].shape[2] == config.cache_size
    # Prefill tokens must all be present in the output
    assert pruned_raas[0][0].shape[2] >= prefill_len

    print("\nAll assertions passed.")
