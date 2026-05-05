# Execution Guide for KV Cache Experiments

Quick reference for running experiments. See `experiments/research_overview.md` for full context and `experiments/progress.md` for current status.

## Setup
```bash
pip install -r requirements.txt
# HF cache is set in .zshrc (on compute nodes: /data/hf_cache/skolawol)
```

---

## Phase 0: Signal Validation (COMPLETE — April 13, 2026)

This phase answered: *does any hidden-state/KV-vector variance signal outperform cumulative attention (H2O) as a token importance proxy?* Answer: yes — Band A (l7–l13) HS variance is positive across datasets, Band B (l18–l25) is negative, and h2o_attn is the weakest signal tested. Reproduction commands below; full results in `experiments/phase0b_ablation_results.md`.

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

Scripts are organized by phase: `slurm/phase0/` (trace collection + labelling), `slurm/phase1/` (accuracy/timing benchmarks), `slurm/setup/` (environment, e.g. `install_flash_attn.sh`). Most use `--partition=general` (48h, non-preemptible). GSM8K uses `--partition=preempt` with `--requeue --signal=B:USR1@60` (traces short enough that requeues are cheap; resume logic in both collect and label scripts).

**Phase 0 (collect → label auto-chains via `afterok`):**
```bash
# math500
sbatch slurm/phase0/run_math500_collect.sh          # 32768 tok, FA2
sbatch slurm/phase0/run_math500_eager_collect.sh    # 16384 tok, eager

# aime2024
sbatch slurm/phase0/run_aime2024_collect.sh         # 32768 tok, FA2
sbatch slurm/phase0/run_aime2024_eager_collect.sh   # 16384 tok, eager

# aime2025 / aime2026 (MathArena)
sbatch slurm/phase0/run_aime2025_collect.sh
sbatch slurm/phase0/run_aime2025_eager_collect.sh
sbatch slurm/phase0/run_aime2026_collect.sh
sbatch slurm/phase0/run_aime2026_eager_collect.sh

# gsm8k — preempt partition, requeue-safe
sbatch slurm/phase0/run_gsm8k_eager_collect.sh
```

Each collect script runs three steps internally:
1. Collect traces (`collect_traces.py --phase0b`)
2. Posthoc Phase 0B extraction (`extract_phase0b_signals.py --input → _posthoc.jsonl`)
3. Cross-validate posthoc vs. collected signals — must PASS before label job is submitted

If cross-validation fails, the label job is **not** submitted and the script exits non-zero. Inspect the posthoc vs. collected trace files before rerunning manually.

**Note**: Old combined collect+label scripts and preempt-partition variants have been removed. The split collect→label model is the only supported flow.

---

## Phase 1: Eviction Logic + Accuracy Benchmarks (COMPLETE — April 24, 2026)

```bash
# Smoke-test all eviction classes (CPU-only, no model needed)
python src/eviction.py
# Expected: all classes reduce cache from 1000 → ≤cache_size; no NaN; bookkeeping OK
```

**`src/eviction.py` classes:**
- Baselines: `H2OEviction`, `ThinKVEviction` (budget-capped), `RaaSEviction` — all eager (need attn matrix).
- HS family (FA2-compatible): `HSVarianceEviction` (l10_r64 − l21_r64), `DetrendendHSVarianceEviction` (rolling z-score detrending), `BandAdaptiveHSEviction` (Band A/B aggregated).
- KV family (FA2-compatible): `KVValVarianceEviction`, `KVKeyVarianceEviction`, `LagKVKeyVarianceEviction`, `LagKVEviction`.
- Hybrid (eager only): `AttentionHSProductEviction`, `HybridSegmentHSEviction`.

### Running benchmarks (`scripts/benchmark.py`)

Phase 1 SLURM scripts are in `slurm/phase1/`. Results write to `/data/user_data/skolawol/kvcache/results/phase1/` (NVMe; /home is too small for full result JSONs). Logs stay on /home in `slurm_logs/phase1/`.

```bash
# Local smoke-test (single problem, single budget — verify pipeline works)
python scripts/benchmark.py \
  --dataset math500 --n_samples 2 --max_new_tokens 1024 \
  --cache_sizes 512 --methods none hs_variance \
  --output /tmp/smoke.json

# Cluster: full eager run (attention-required methods)
sbatch slurm/phase1/run_benchmark_math500_eager.sh   # 100 problems, 4 cache sizes
sbatch slurm/phase1/run_benchmark_aime2024_eager.sh  # 30 problems, 5 cache sizes

# Cluster: full flash run (FA2-compatible methods)
sbatch slurm/phase1/run_benchmark_math500_flash.sh
sbatch slurm/phase1/run_benchmark_aime2024_flash.sh
```

The eager scripts request 2 GPUs; the flash scripts request 1 GPU (single-GPU avoids a flash_attn multi-GPU kernel coordination crash). Both use `--resume` so partial results survive job preemption.

### Analyzing results (`scripts/analyze_phase1.py`)

```bash
python scripts/analyze_phase1.py
# Prints accuracy / wall_time / peak_gpu_mb tables for each dataset
# Writes accuracy-vs-cache-size PDFs to reports/phase1_plots/
```

Outputs:
- Console: 6 tables (3 metrics × 2 datasets), with FA2 ✓/✗ column.
- `reports/phase1_plots/accuracy_math500.pdf` and `accuracy_aime2024.pdf` — solid lines for FA2-compatible methods, dashed for attention-required, dotted for the `none` ceiling.

### Phase 1 results summary

See `experiments/progress.md` (April 24 entry) and `experiments/paper_strategy.md` (Phase 1 Results section) for the full table. Headlines: `hs_variance_detrend` reaches 72% on MATH-500 @ 4096 (FA2-compatible, beats ThinKV 71%); `lag_kv` reaches 37% on AIME-2024 @ 8192 (FA2-compatible, beats every attention method by 3 points); `lag_kv` is 2.8× faster than `raas` at the same cache budget; H2O collapses to empty generations on 93/100 MATH-500 problems at cache=1024.

---

## Phase 2: Robustness, Ablations, FA2-compatible hybrid (TODO)

Plan from `experiments/progress.md`:
1. **Robustness**: combine AIME 2024+2025+2026 to n=90; add GSM8K; extend cache budgets to 256 (find breakdown) and 12288 (interpolate AIME curve).
2. **Ablations**: per-layer ablation (l10 vs l21 vs l10−l21); detrending at lower budgets; token-retention case study figure.
3. **FA2-compatible hybrid**: `kv_seg_hs` — KV statistics for segment classification (replacing ThinKV's attention-entropy classifier) + HS for within-segment ranking. Closes the gap at tight budgets where `hybrid_seg_hs` (eager) currently wins.
4. **Engineering validation**: optional vLLM integration or long-prefill experiments to surface FA2's prefill memory advantage.

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
