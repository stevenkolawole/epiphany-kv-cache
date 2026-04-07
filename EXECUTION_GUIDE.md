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

# Full collection with Phase 0B signals (--phase0b adds pre-RoPE key variance + per-layer HS at all 32 layers 0-31)
python scripts/collect_traces.py --dataset math500   --n_samples 100 --max_new_tokens 32768 --phase0b
python scripts/collect_traces.py --dataset aime2024  --n_samples 30  --max_new_tokens 32768 --phase0b
python scripts/collect_traces.py --dataset aime2025  --n_samples 30  --max_new_tokens 32768 --phase0b
python scripts/collect_traces.py --dataset aime2026  --n_samples 30  --max_new_tokens 32768 --phase0b

# With H2O + attention entropy (requires eager attention — more VRAM at long contexts)
# AIME eager capped at 16384: at 32768 tokens the attention matrix is ~64GB (OOM risk)
# GSM8K: use 500 samples (traces ~1-2k tokens, fast)
python scripts/collect_traces.py --dataset math500   --n_samples 100 --max_new_tokens 16384 --force_eager_attn --phase0b
python scripts/collect_traces.py --dataset aime2024  --n_samples 30  --max_new_tokens 16384 --force_eager_attn --phase0b
python scripts/collect_traces.py --dataset aime2025  --n_samples 30  --max_new_tokens 16384 --force_eager_attn --phase0b
python scripts/collect_traces.py --dataset aime2026  --n_samples 30  --max_new_tokens 16384 --force_eager_attn --phase0b
python scripts/collect_traces.py --dataset gsm8k     --n_samples 500 --max_new_tokens 16384 --force_eager_attn --phase0b

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

All scripts live in `slurm/`. Most use `--partition=general` (48h, non-preemptible).
GSM8K uses `--partition=preempt` with `--requeue --signal=B:USR1@60` (traces short enough that requeues are cheap; collect and label scripts both have resume logic).

**Submit only collect scripts — label jobs auto-chain via `afterok`:**
```bash
# math500 (complete)
sbatch slurm/run_math500_collect.sh          # math500, 32768 tok, FA2 — chains run_math500_label.sh
sbatch slurm/run_math500_eager_collect.sh    # math500, 16384 tok, eager — chains run_math500_eager_label.sh

# aime2024 (complete; eager rerun in progress with hook fix)
sbatch slurm/run_aime2024_collect.sh         # aime2024, 32768 tok, FA2 — chains run_aime2024_label.sh
sbatch slurm/run_aime2024_eager_collect.sh   # aime2024, 16384 tok, eager — chains run_aime2024_eager_label.sh

# aime2025 / aime2026 (MathArena — in progress)
sbatch slurm/run_aime2025_collect.sh         # aime2025, 32768 tok, FA2 — chains run_aime2025_label.sh
sbatch slurm/run_aime2025_eager_collect.sh   # aime2025, 16384 tok, eager — chains run_aime2025_eager_label.sh
sbatch slurm/run_aime2026_collect.sh         # aime2026, 32768 tok, FA2 — chains run_aime2026_label.sh
sbatch slurm/run_aime2026_eager_collect.sh   # aime2026, 16384 tok, eager — chains run_aime2026_eager_label.sh

# gsm8k — preempt partition, requeue-safe
sbatch slurm/run_gsm8k_eager_collect.sh      # gsm8k, 500 samples, 16384 tok, eager — chains run_gsm8k_eager_label.sh
```

Each collect script runs three steps internally:
1. Collect traces (`collect_traces.py --phase0b`)
2. Posthoc Phase 0B extraction (`extract_phase0b_signals.py --input → _posthoc.jsonl`)
3. Cross-validate posthoc vs. collected signals — must PASS before label job is submitted

If cross-validation fails, the label job is **not** submitted and the script exits non-zero. Inspect the posthoc vs. collected trace files before rerunning manually.

**Note**: `run_math500_eager.sh` (old combined collect+label) is deprecated and deleted. `run_aime2024_preempt.sh` and `run_aime2024_eager_preempt.sh` are old preempt-partition scripts — also deprecated.

---

## Phase 1: Eviction Logic

```bash
# Verify baseline eviction implementations work
python src/eviction.py
# Expected: all 5 classes reduce cache from 1000 → 512 tokens
```

**Current state of `src/eviction.py`:**
- `H2OEviction` — stateful cumulative attention, sink tokens, recency window. Usable as baseline.
- `ThinKVEviction` — segment entropy R/E/T classification. **Known bug**: per-segment fixed budgets don't enforce total cache_size; can over-retain. Fix before using as benchmark baseline.
- `RaaSEviction` — LRU timestamps + unconditional prefill preservation. Usable as baseline.
- `SemanticEviction` — **stale, replace for Phase 1.** Uses last-layer HS L1 diff + average post-RoPE KV variance. Phase 0B showed both choices are wrong (last layer is Band B; average over all layers washes out Band A). Replace with `HSVarianceEviction` implementing `l10_rolling64 − l21_rolling64` with incremental online computation.
- `AttentionBasedEviction` — legacy single-step POC baseline. Superseded by H2OEviction.

**Phase 1 deliverable**: `HSVarianceEviction` — incremental HS scoring at layers 10 and 21, causal rolling64 smoothing, stored per-token importance scores, no extra forward pass.

---

## Phase 2: Accuracy vs. Cache-Size Benchmarking (TODO — after Phase 1)

Once `HSVarianceEviction` is implemented, run end-to-end accuracy curves:

```bash
# Primary comparison (ThinKV/RaaS benchmarks)
python scripts/benchmark.py \
  --model deepseek-ai/deepseek-r1-distill-llama-8b \
  --dataset math500 aime2024 gsm8k \
  --cache_sizes 32 64 128 256 512 1024 \
  --methods none h2o thinKV raas hs_variance \
  --output results/benchmark_results.json

# RaaS-specific comparison (their model, their datasets)
python scripts/benchmark.py \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --dataset gsm8k math500 aime2024 \
  --cache_sizes 64 128 256 512 1024 \
  --methods none h2o raas hs_variance \
  --output results/benchmark_results_raas_model.json
```

**Note**: `scripts/poc_harness.py` is deprecated — known issues with hardcoded mock results and eviction never being called during generation. Do not use.

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
