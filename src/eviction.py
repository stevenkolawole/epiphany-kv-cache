"""
Baseline KV cache eviction using attention scores.
Implements standard attention-based eviction for comparison.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
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
        keep_recent = self.config.keep_recent_k

        if current_seq_len <= cache_size:
            return torch.ones(current_seq_len, dtype=torch.bool, device=attention_weights.device)

        # Average attention each key token received across batch, heads, and query positions.
        # Shape: (key_len,)
        avg_attention = attention_weights.mean(dim=(0, 1, 2))

        # Always keep the most recent tokens
        keep_mask = torch.zeros(current_seq_len, dtype=torch.bool, device=attention_weights.device)
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
        keep_mask = self.compute_eviction_mask(last_attn, seq_len)

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


if __name__ == "__main__":
    # Simple smoke test
    batch_size = 2
    num_heads = 8
    hidden_dim = 64
    seq_len = 1000

    config = EvictionConfig(cache_size=512, keep_recent_k=128)

    k_cache = torch.randn(seq_len, hidden_dim)
    v_cache = torch.randn(seq_len, hidden_dim)
    hidden_states = torch.randn(seq_len, hidden_dim)
    attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len).abs()

    baseline = AttentionBasedEviction(config)
    k_out, v_out = baseline.evict(k_cache, v_cache, attention_weights)
    print(f"Baseline: {seq_len} → {k_out.shape[0]} tokens (target: {config.cache_size})")
    assert k_out.shape[0] == config.cache_size, "Baseline eviction failed"

    semantic = SemanticEviction(config)
    k_out_s, v_out_s = semantic.evict(k_cache, v_cache, hidden_states, attention_weights)
    print(f"Semantic: {seq_len} → {k_out_s.shape[0]} tokens (target: {config.cache_size})")
    assert k_out_s.shape[0] == config.cache_size, "Semantic eviction failed"

    # Test past_key_values interface (HuggingFace format)
    num_layers = 4
    head_dim = hidden_dim // num_heads
    past_kv = tuple(
        (torch.randn(1, num_heads, seq_len, head_dim),
         torch.randn(1, num_heads, seq_len, head_dim))
        for _ in range(num_layers)
    )
    attn_tuple = tuple(
        torch.randn(1, num_heads, 1, seq_len).abs()
        for _ in range(num_layers)
    )
    pruned = baseline.evict_past_key_values(past_kv, attn_tuple)
    print(f"Baseline past_kv: seq_len {seq_len} → {pruned[0][0].shape[2]}")
    assert pruned[0][0].shape[2] == config.cache_size

    pruned_s = semantic.evict_past_key_values(past_kv, attention_weights=attn_tuple)
    print(f"Semantic past_kv (kv-variance): seq_len {seq_len} → {pruned_s[0][0].shape[2]}")
    assert pruned_s[0][0].shape[2] == config.cache_size

    print("\nAll assertions passed.")
