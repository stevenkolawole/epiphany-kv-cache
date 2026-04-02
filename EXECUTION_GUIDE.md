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

Signals collected per token (9 total):
- **Always**: `kv_key_var`, `kv_key_norm`, `kv_val_var`, `cross_head_var` (free from KV cache), `hs_l2_diff`, `hs_cos_dist`, `hs_norm` (post-hoc forward pass, seq ≤ `--max_new_tokens`)
- **With `--force_eager_attn` only**: `h2o_attn`, `attn_entropy` (require full attention matrix; FlashAttention incompatible)

**Step 1 — Generate reasoning traces**
```bash
# Pipeline test (fast — do this first)
python scripts/collect_traces.py --dataset math500 --n_samples 10 --max_new_tokens 4096

# Full collection with Phase 0B signals (--phase0b adds pre-RoPE key variance + layer-wise HS at 16/20/24)
python scripts/collect_traces.py --dataset math500   --n_samples 100 --max_new_tokens 32768 --phase0b
python scripts/collect_traces.py --dataset aime2024  --n_samples 30  --max_new_tokens 32768 --phase0b

# With H2O + attention entropy (requires eager attention — more VRAM at long contexts)
# AIME eager capped at 16384: at 32768 tokens the attention matrix is ~64GB (OOM risk)
python scripts/collect_traces.py --dataset math500   --n_samples 100 --max_new_tokens 16384 --force_eager_attn --phase0b
python scripts/collect_traces.py --dataset aime2024  --n_samples 30  --max_new_tokens 16384 --force_eager_attn --phase0b

# Output: data/<dataset>_traces.jsonl
# After collection, run posthoc extraction + cross-validate before labelling:
python scripts/extract_phase0b_signals.py --input data/<dataset>_traces.jsonl --output data/<dataset>_traces_posthoc.jsonl
python scripts/extract_phase0b_signals.py --input data/<dataset>_traces_posthoc.jsonl --compare data/<dataset>_traces.jsonl
```

**Step 2 — Label token importance (counterfactual occlusion)**
```bash
# Dry run first (3 traces, fast)
python scripts/label_importance.py --dataset math500 --dry_run

# Full labelling — only runs on correctly-answered traces
# ~5–15 min/trace depending on length; use --max_traces to run a subset first
python scripts/label_importance.py --dataset math500 --max_traces 10
python scripts/label_importance.py --dataset math500   # all correct traces

# Output: data/<dataset>_traces_labelled.jsonl
```

**Masking methodology (important)**: label_importance.py uses **occlusion**, not truncation.
For each window, it replaces those tokens with `pad_id`, feeds the full modified reasoning trace up to the `</think>` boundary as context, then asks the model to generate the final answer. Every window call feeds the same context length — only the window's content varies. This is a content test, not a position test. The old approach (truncating at `mask_start`) was a position proxy and has been fixed.

**Step 3 — Run signal ablation (Dimension 1)**
```bash
python scripts/signal_ablation.py --dataset math500
# Output: results/<dataset>_signal_ablation.csv + ranked console table

# Key result: does any residual-stream signal beat H2O (Spearman ρ)?
# If yes → proceed to Dimension 2–5 ablations (see research_overview.md §3.1)
# If no  → hypothesis needs revision; review which signals came closest
```

See `experiments/research_overview.md` §3.1 for all 8 signal variants and the full ablation plan.

---

---

## SLURM Batch Scripts (cluster)

All scripts live in `slurm/`. All use `--partition=general` (48h, non-preemptible).

**Submit only these 4 commands.** Label jobs are auto-submitted via `afterok` dependency once cross-validation passes:
```bash
sbatch slurm/run_math500_collect.sh          # math500, 32768 tok, FA2, Phase 0B — chains run_math500_label.sh
sbatch slurm/run_math500_eager_collect.sh    # math500, 16384 tok, eager, Phase 0B — chains run_math500_eager_label.sh
sbatch slurm/run_aime2024_collect.sh         # aime2024, 32768 tok, FA2, Phase 0B — chains run_aime2024_label.sh
sbatch slurm/run_aime2024_eager_collect.sh   # aime2024, 16384 tok, eager, Phase 0B — chains run_aime2024_eager_label.sh
```

Each collect script runs three steps internally:
1. Collect traces (`collect_traces.py --phase0b`)
2. Posthoc Phase 0B extraction (`extract_phase0b_signals.py --input → _posthoc.jsonl`)
3. Cross-validate posthoc vs. collected signals — must PASS before label job is submitted

If cross-validation fails, the label job is **not** submitted and the script exits non-zero. Inspect the posthoc vs. collected trace files before rerunning manually.

**Note**: `run_math500_eager.sh` (old combined collect+label) is deprecated and deleted. `run_aime2024_preempt.sh` and `run_aime2024_eager_preempt.sh` are old preempt-partition scripts — also deprecated.

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
