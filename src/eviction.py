"""
Baseline KV cache eviction using attention scores.
Implements standard attention-based eviction for comparison.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class EvictionConfig:
    """Configuration for KV cache eviction."""
    cache_size: int = 4096  # Max KV cache tokens
    eviction_method: str = "attention"  # "attention", "fifo", "lru"
    keep_recent_k: int = 128  # Always keep recent K tokens (like StreamingLLM)
    

class AttentionBasedEviction:
    """
    Baseline: Evict KV cache tokens with lowest attention scores.
    Similar to attention sink methods but without keep_recent_k.
    """
    
    def __init__(self, config: EvictionConfig):
        self.config = config
        self.attention_scores = None
        
    def compute_eviction_mask(
        self, 
        attention_weights: torch.Tensor,
        current_seq_len: int
    ) -> torch.Tensor:
        """
        Compute which tokens to keep based on attention scores.
        
        Args:
            attention_weights: Shape (batch, heads, query_len, key_len)
            current_seq_len: Current sequence length
            
        Returns:
            Boolean mask of shape (key_len,) where True = keep, False = evict
        """
        cache_size = self.config.cache_size
        keep_recent = self.config.keep_recent_k
        
        if current_seq_len <= cache_size:
            # No eviction needed yet
            return torch.ones(current_seq_len, dtype=torch.bool)
        
        # Compute average attention per token across batch and heads
        # attention_weights: (batch, heads, query_len, key_len)
        avg_attention = attention_weights.mean(dim=(0, 1, 2))  # (key_len,)
        
        # Keep recent tokens regardless of attention
        recent_mask = torch.zeros(current_seq_len, dtype=torch.bool)
        recent_mask[-keep_recent:] = True
        
        # For non-recent tokens, keep those with highest attention
        non_recent_scores = avg_attention[:-keep_recent].clone()
        num_to_keep = cache_size - keep_recent
        
        _, top_indices = torch.topk(non_recent_scores, min(num_to_keep, len(non_recent_scores)))
        
        non_recent_mask = torch.zeros(current_seq_len - keep_recent, dtype=torch.bool)
        non_recent_mask[top_indices] = True
        
        keep_mask = torch.zeros(current_seq_len, dtype=torch.bool)
        keep_mask[:-keep_recent] = non_recent_mask
        keep_mask[-keep_recent:] = True
        
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
            attention_weights: Attention weights for current step
            
        Returns:
            Evicted k_cache, v_cache
        """
        current_seq_len = k_cache.shape[0]
        
        if current_seq_len <= self.config.cache_size:
            return k_cache, v_cache
        
        # Compute eviction mask
        keep_mask = self.compute_eviction_mask(attention_weights, current_seq_len)
        
        # Apply mask
        k_evicted = k_cache[keep_mask]
        v_evicted = v_cache[keep_mask]
        
        return k_evicted, v_evicted


class SemanticEviction:
    """
    Proposed: Evict tokens based on semantic importance heuristics.
    Uses hidden state characteristics and token patterns.
    """
    
    def __init__(self, config: EvictionConfig):
        self.config = config
        
    def compute_semantic_importance(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute semantic importance scores for tokens.
        
        Heuristics:
        - High state variance = important (insight/decision point)
        - Attention patterns (similar to baseline)
        - Token types (conclusion words vs filler)
        
        Args:
            hidden_states: Shape (seq_len, hidden_dim)
            attention_weights: Optional attention weights
            
        Returns:
            Importance scores: (seq_len,)
        """
        seq_len, hidden_dim = hidden_states.shape
        
        # Compute state variance (roughness/change in representation)
        state_diff = torch.abs(torch.diff(hidden_states, dim=0))  # (seq_len-1, hidden_dim)
        state_variance = state_diff.mean(dim=1)  # (seq_len-1,)
        state_variance = torch.cat([state_variance[:1], state_variance])  # Pad for first token
        
        # Normalize
        if state_variance.max() > 0:
            state_variance = (state_variance - state_variance.min()) / (state_variance.max() - state_variance.min() + 1e-8)
        
        importance = state_variance
        
        # Optional: combine with attention
        if attention_weights is not None:
            avg_attention = attention_weights.mean(dim=(0, 1, 2))
            if avg_attention.max() > 0:
                avg_attention = (avg_attention - avg_attention.min()) / (avg_attention.max() - avg_attention.min() + 1e-8)
            importance = 0.5 * importance + 0.5 * avg_attention
        
        return importance
    
    def evict(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evict tokens based on semantic importance.
        
        Args:
            k_cache: Key cache
            v_cache: Value cache
            hidden_states: Hidden states for all tokens
            attention_weights: Optional attention weights
            
        Returns:
            Evicted k_cache, v_cache
        """
        current_seq_len = k_cache.shape[0]
        
        if current_seq_len <= self.config.cache_size:
            return k_cache, v_cache
        
        # Compute importance scores
        importance = self.compute_semantic_importance(hidden_states, attention_weights)
        
        # Keep recent tokens + high importance ones
        keep_recent = self.config.keep_recent_k
        cache_size = self.config.cache_size
        
        # Always keep recent tokens
        importance[-keep_recent:] = float('inf')
        
        # Keep top importance tokens
        num_to_keep = cache_size
        _, keep_indices = torch.topk(importance, min(num_to_keep, current_seq_len), largest=True)
        keep_indices = torch.sort(keep_indices)[0]  # Sort for contiguity
        
        k_evicted = k_cache[keep_indices]
        v_evicted = v_cache[keep_indices]
        
        return k_evicted, v_evicted


if __name__ == "__main__":
    # Simple test
    batch_size = 2
    num_heads = 8
    hidden_dim = 64
    seq_len = 1000
    
    config = EvictionConfig(cache_size=512, keep_recent_k=128)
    
    # Dummy data
    k_cache = torch.randn(seq_len, hidden_dim)
    v_cache = torch.randn(seq_len, hidden_dim)
    hidden_states = torch.randn(seq_len, hidden_dim)
    attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
    
    # Test baseline
    baseline = AttentionBasedEviction(config)
    k_evicted, v_evicted = baseline.evict(k_cache, v_cache, attention_weights)
    print(f"Baseline: {seq_len} → {k_evicted.shape[0]} tokens (cache size: {config.cache_size})")
    
    # Test semantic
    semantic = SemanticEviction(config)
    k_evicted_sem, v_evicted_sem = semantic.evict(k_cache, v_cache, hidden_states, attention_weights)
    print(f"Semantic: {seq_len} → {k_evicted_sem.shape[0]} tokens (cache size: {config.cache_size})")
