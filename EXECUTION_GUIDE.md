# Execution Guide for KV Cache Experiments

Quick reference for running experiments. See `experiments/research_overview.md` for full context and `experiments/progress.md` for current status.

## Setup
```bash
pip install -r requirements.txt
# HF cache is set in .zshrc (on compute nodes: /data/hf_cache/skolawol)
```

---

## Phase 0: Signal Validation (CURRENT PRIORITY)

This phase answers: *does any hidden-state/KV-vector variance signal outperform cumulative attention (H2O) as a token importance proxy?*

**Step 1 — Generate real reasoning traces**
```bash
# MATH-500: competition-level math problems (matches ThinKV/RaaS)
python scripts/collect_traces.py \
  --model deepseek-ai/deepseek-r1-distill-llama-8b \
  --dataset hendrycks/competition_math \
  --n_samples 100 \
  --output data/math500_traces.jsonl

# AIME 2024 only (NOT pooled years — must match ThinKV exactly)
python scripts/collect_traces.py \
  --model deepseek-ai/deepseek-r1-distill-llama-8b \
  --dataset Maxwell-Jia/AIME_2024 \
  --n_samples 30 \
  --output data/aime2024_traces.jsonl

# Saves: generated token IDs, per-layer KV tensors, correctness label
```

**Step 2 — Build counterfactual importance labels**
```bash
# For each trace, mask sliding windows and check if answer flips
python scripts/label_importance.py \
  --traces data/math500_traces.jsonl \
  --window_size 16 \
  --output data/math500_importance_labels.jsonl
# Output: per-token binary importance labels (1 = masking this window flips the answer)
```

**Step 3 — Sweep signal type variants (Dimension 1 ablation)**
```bash
# Compute all 6 signal variants + H2O cumulative attention at each token position
# Report Spearman correlation with counterfactual labels
python scripts/signal_ablation.py \
  --traces data/math500_traces.jsonl \
  --labels data/math500_importance_labels.jsonl \
  --dimension signal_type \
  --output experiments/ablation_signal_type.json
```

**Step 4 — Sequential ablation across remaining dimensions**
```bash
# After identifying best signal type, sweep each remaining dimension
python scripts/signal_ablation.py --dimension rope_interaction --best_signal <from step 3>
python scripts/signal_ablation.py --dimension layer_aggregation --best_signal <from step 3>
python scripts/signal_ablation.py --dimension temporal_aggregation --best_signal <from step 3>
python scripts/signal_ablation.py --dimension head_aggregation --best_signal <from step 3>
```

See `experiments/research_overview.md` §3.1 for all variants and hypotheses.

---

## Phase 1: Eviction Logic (DONE — but needs upgrades)

```bash
# Verify current eviction implementations work
python src/eviction.py
# Expected: cache reduces from 1000 → 512 tokens for both methods
```

**Upgrades needed before benchmarking**:
- `AttentionBasedEviction`: upgrade to H2O (cumulative attention, not single-step)
- Add ThinKV thought classifier (R/E/T from KDE on 4 layers, refresh every 128 steps)
- Add RaaS eviction (LRU timestamp + prefill preservation)

---

## Phase 2: POC Harness (DONE — needs real models + data)

```bash
# Run with DeepSeek-R1-Distill (primary target — long reasoning traces)
python scripts/poc_harness.py \
  --model deepseek-ai/deepseek-r1-distill-llama-8b \
  --cache_size 128 \
  --eviction_method semantic

# Run baseline comparison (no eviction)
python scripts/poc_harness.py \
  --model deepseek-ai/deepseek-r1-distill-llama-8b \
  --cache_size 128 \
  --eviction_method none

# Results saved to: experiments/poc_results.jsonl
```

**Note**: GPT-2 is no longer the target. Use DeepSeek-R1-Distill variants — vanilla instruction
models (LLaMA-3, Qwen) produce traces too short to stress the KV cache meaningfully.

For RaaS comparability, also run with Qwen2.5-Math-7B-Instruct:
```bash
python scripts/poc_harness.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --dataset GSM8K MATH-500 AIME_2024 \
  --cache_size 128 \
  --eviction_method semantic
```

---

## Phase 3: Baseline Comparison Curves (TODO)

Once H2O, ThinKV, and RaaS are implemented:

```bash
# Primary comparison (ThinKV/RaaS benchmarks)
python scripts/benchmark.py \
  --model deepseek-ai/deepseek-r1-distill-llama-8b \
  --dataset MATH-500 AIME_2024 LiveCodeBench GSM8K \
  --cache_sizes 32 64 128 256 512 1024 \
  --methods none h2o thinKV raas semantic \
  --output experiments/benchmark_results.json

# RaaS-specific comparison (their model, their datasets)
python scripts/benchmark.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --dataset GSM8K MATH-500 AIME_2024 \
  --cache_sizes 64 128 256 512 1024 \
  --methods none h2o raas semantic \
  --output experiments/benchmark_results_raas_model.json
```

Target: beat ThinKV's accuracy vs. cache-size Pareto curve on MATH-500, AIME 2024, LiveCodeBench.

**Benchmark framing notes**:
- MATH-500 + AIME 2024 + LiveCodeBench: head-to-head with ThinKV
- MATH-500 + AIME 2024 + GSM8K on Qwen2.5-Math: head-to-head with RaaS
- GSM8K on DeepSeek-R1-Distill: low-pressure control only (traces too short for cache pressure; run to show no regression)
- HotpotQA: secondary evaluation for Gap F (non-monotonic recall claim); NOT a ThinKV/RaaS comparison point

---

## Monitoring & Debugging

```bash
# Verify imports
python -c "import torch, transformers; print('OK')"

# Check JSONL traces (note: use readlines(), not json.load, for JSONL)
python -c "
import json
with open('data/math500_traces.jsonl') as f:
    samples = [json.loads(l) for l in f]
print(f'{len(samples)} traces, first keys: {list(samples[0].keys())}')
"

# Quick model load test
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-r1-distill-llama-8b')
print('Tokenizer OK, vocab size:', tok.vocab_size)
"

# Memory check before running large model
python -c "import torch; print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

---

## Key Metrics to Track Per Experiment

| Metric | How to measure |
|---|---|
| Accuracy | Word-boundary regex match against known answer |
| Peak GPU memory | `torch.cuda.reset_peak_memory_stats()` before each example, read after |
| Tokens generated | `len(generated_ids)` (not word count) |
| Eviction recall rate | Fraction of "important" tokens (counterfactual labels) retained in cache |
| Signal correlation | Spearman(signal_score, importance_label) per trace, averaged |
| Cache retention % | `cache_size / max_seq_len` at end of generation |
