# Experiments and Progress

This document tracks the design, execution, and results of experiments for epiphany-aware KV cache management in reasoning models.

## Overview
- **Goal**: Reduce KV cache memory usage in reasoning models (10k-100k tokens) while maintaining accuracy, by evicting based on semantic importance rather than attention scores.
- **Models**: LLaMA-3, Mistral, Qwen, DeepSeek (OSS focus).
- **Tasks**: Math reasoning (GSM8K), long-form QA (HotpotQA).
- **Hardware**: Single GPU start, scale to cluster/multi-GPU.
- **Success Criteria**: Negligible accuracy loss, maximal memory reduction.
- **Automation Vision**: Rule-based eviction (e.g., inspired by StreamingLLM: keep first N tokens + sliding window + semantic heuristics) without classifiers/LLM-judge.

## Experiment Order
1. **Data Collection & Analysis** (Foundation)
2. **Baseline Implementation & Benchmarking**
3. **Semantic Importance Scoring Development**
4. **POC Testing & Validation**
5. **Scaling Experiments**
6. **Interpretability & Edge Cases**

## Progress Log

### [Date: March 4, 2026] - Project Setup ✓
- Scaffolded Python project with src/, tests/, notebooks/, data/, experiments/, scripts/
- Installed dependencies: PyTorch, Transformers, etc.
- Created this tracking document.

### [Date: March 4, 2026] - Data Collection & Analysis ✓
- Created `src/data_collection.py` with trace collection and analysis utilities
- Created `scripts/analyze_traces.py` for offline analysis of reasoning traces
- Implemented heuristic-based segment classification: rambling, exploration, insight, neutral
- Generated 3 synthetic math reasoning traces and saved to `data/synthetic_math_traces.jsonl`
- Patterns Identified:
  - "Rambling" segments: "let me", "hmm", "wait", "actually" (thinking process)
  - "Exploration" segments: Questions and hypothesis testing
  - "Insight" segments: Conclusions and assertions
- Next: Fetch real DeepSeek/Qwen traces from HuggingFace and run baseline implementation

### [Date: March 4, 2026] - Baseline Implementation ✓
- Implemented `AttentionBasedEviction` in `src/eviction.py`
- Baseline strategy: Keep recent K tokens + top attention scores (StreamingLLM-inspired)
- Also implemented `SemanticEviction` using hidden state variance + attention
- Both reduce cache from 1000 → 512 tokens as expected
- Tested and verified on dummy data (batch_size=2, seq_len=1000)
- Next: Create POC harness to test on actual LLaMA/Qwen models with math reasoning tasks.

### [Date: March 4, 2026] - Initial POC Run (GPT-2)
- Ran GPT-2 test with **10 examples** (not 50 — synthetic dataset has 10 traces)
- Cache size 512 was never actually exceeded by 256-token generations, so eviction
  was never triggered in practice. Both "baseline" and "semantic" runs used identical
  unmodified generation; any accuracy difference was random noise, not eviction effect.
- Real memory usage was identical between methods (confirmed by peak_memory_mb in results)
- Next: Integrate eviction into generation loop (see codebase-fix entry below)

### [Date: March 4, 2026] - Git & Visualizations
- Initialized git repository
- Created `scripts/visualize.py` with segment and length visualizations
- `viz_memory_reduction.png` used made-up reduction percentages (25–45%) — NOT measured data
- Created `MODEL_VARIANTS.md` documenting vanilla vs reasoning-enabled testing strategy

### [Date: March 4, 2026] - Codebase Audit & Fixes ✓
Issues identified and corrected:

**eviction.py**
- Fixed: first-token padding used `state_variance[:1]` (copied first diff) instead of zeros
- Added: `evict_past_key_values()` to both classes for use with HuggingFace `past_key_values`
- Added: `_importance_from_kv()` to `SemanticEviction` — key-vector variance proxy for steps
  where full hidden states are not available (all decoding steps after prefill)
- Added: `semantic_alpha` as a configurable `EvictionConfig` parameter (was hardcoded 0.5)

**poc_harness.py**
- Fixed: replaced `model.generate()` with a manual step-by-step loop that passes
  `past_key_values` back to the model and calls `evict_past_key_values()` when the
  cache exceeds `cache_size`. This is the first time eviction is actually integrated.
- Fixed: removed hardcoded mock results (accuracy=0.65/0.60, memory=1500/2048 MB) that
  were silently written to poc_results.jsonl when model loading failed. Harness now
  skips a variant and logs clearly if the model cannot be loaded.
- Fixed: answer matching changed from substring `in` (false positives: "2" inside "12")
  to word-boundary regex (`(?<!\w)answer(?!\w)`)
- Added: `eviction_method="none"` baseline that runs unmodified generation, providing
  a true control condition separate from eviction-based methods

**data_collection.py / analyze_traces.py / visualize.py**
- Fixed: segment classifier duplicated 3× with subtly different keyword lists; consolidated
  to a single set of compiled regex patterns shared across all files
- Fixed: word-boundary patterns replace plain substring matching — "how" no longer matches
  "however", "so" no longer matches "also", etc.

**visualize.py**
- Fixed: `plot_memory_reduction()` replaced with `plot_theoretical_eviction_savings()`,
  which plots the mathematically correct retention fraction (cache_size / max_seq_len)
  and is clearly labelled "THEORETICAL — not measured"

### [Date: TBD] - Scaling
- Status: Not started
- Plan: Larger models (LLaMA-7B, Qwen-7B), full datasets, longer sequences
- Output: Scalability analysis, ablation studies

### [Date: TBD] - Interpretability
- Status: Not started
- Plan: Analyze evicted tokens, edge cases, failure modes
- Output: Refinements and insights

## Next Steps
1. **Immediate**: Run POC with GPT2 quickly, verify pipeline works
2. **This week**: 
   - Load a small LLaMA or Qwen model (1.5-3B params)
   - Generate 100+ math problem reasoning traces
   - Compare baseline vs semantic eviction
   - Collect metrics: accuracy drop %, memory saved, latency impact
3. **Findings to track**:
   - Does semantic eviction beat attention-based? By how much?
   - Which semantic heuristic works best (state variance, attention, hybrid)?
   - Can we achieve automation without classifiers/LLM-judge?
   
## Key Insights So Far
- Attention-based eviction (e.g., attention sinks) keeps high-attention tokens
- But in reasoning: high attention ≠ important (e.g., "let me think" gets high attention)
- Hypothesis: Look at hidden state *changes* (jumps in representation space) instead
- Goal: Approximate these changes with fast on-the-fly heuristics (no training needed)
- StreamingLLM-style approach: Keep recent K + important historical tokens

## Notes
- Prioritize LLaMA-3 for popularity (also test Qwen, Mistral)
- Aim for automation: Compute semantic proxies efficiently (state variance + attention patterns)
- Update this file after each major milestone</content>
<parameter name="filePath">/home/skolawol/workspace/kvcache/experiments/experiments.md