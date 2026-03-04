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

### [Date: March 4, 2026] - POC Harness ✓
- Created `scripts/poc_harness.py` for evaluating models on reasoning tasks
- Supports loading any HuggingFace model (LLaMA, Qwen, DeepSeek, etc.)
- Measures: accuracy, tokens generated, peak memory, time/example
- Ready to test on small models; will scale to larger models
- Next: Run POC on GPT2 (quick test) then LLaMA-1.5B (real test)

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