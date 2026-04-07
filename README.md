# Epiphany-Aware KV Cache Management

KV cache eviction for long-reasoning LLMs (DeepSeek-R1 class) using hidden-state variance signals instead of attention weights.

## Problem

Reasoning models generate 10k–100k token traces. KV caches at this length exceed GPU memory. Existing eviction methods (H2O, ThinKV, SnapKV) use attention weights as importance proxies — but attention weight ≠ semantic importance in long reasoning traces, and materializing the attention matrix is architecturally incompatible with FlashAttention 2.

## Core Hypothesis

Hidden-state variance at specific mid-layers (and KV-key variance) predicts token importance better than attention weight, and can be computed entirely within a FA2-compatible forward pass with no extra memory overhead.

## Key Findings (Phase 0B)

- **Band A (layers 7–13):** consistently positive Spearman ρ with importance labels across all datasets — high HS variance at these layers = important token
- **Band B (layers 18–25):** consistently negative ρ — high variance here = dispensable token
- **Combined score** `l10_rolling64 − l21_rolling64` is the Phase 1 eviction signal
- **Rolling64 smoothing** outperforms raw and EMA signals by 30–57% universally; importance is a sustained contextual property, not an instantaneous spike
- **h2o_attn is the weakest signal tested** — 3–12× weaker than HS signals, confirming the core hypothesis
- **FA2 engineering advantage:** the method requires only `output_hidden_states=True` (standard, FA2-compatible) and two stored vectors (~20KB) for incremental score computation. No attention matrix materialization. No extra forward pass.

## Project Structure

```
src/
  eviction.py          — H2OEviction, ThinKVEviction, RaaSEviction baselines;
                         SemanticEviction (stale — Phase 1 replacement pending)
scripts/
  collect_traces.py    — trace collection with Phase 0B signals
  extract_phase0b_signals.py — posthoc HS extraction + cross-validation
  label_importance.py  — counterfactual occlusion labelling
  signal_ablation.py   — Spearman ρ ablation across all signal variants
slurm/                 — SLURM batch scripts for all datasets
experiments/
  progress.md          — experiment log and current status
  phase0b_ablation_results.md — full Phase 0B signal ablation results
  signals_reference.md — technical reference for all signals
  paper_strategy.md    — NeurIPS writing and presentation strategy
  research_overview.md — literature context and research design
data/                  — collected traces (gitignored)
results/               — signal ablation CSVs
```

## Current Status

- **Phase 0B complete:** signal ablation across math500, math500_eager, aime2024, aime2024_eager
- **Running now:** aime2024_eager rerun (hook fix), aime2025, aime2026, gsm8k (preempt)
- **Phase 1 next:** implement `HSVarianceEviction` using `l10_rolling64 − l21_rolling64`; run accuracy vs. cache-size curves against H2O/ThinKV/RaaS baselines

## Setup

```bash
pip install -r requirements.txt
# HF cache: /data/hf_cache/skolawol (set in .zshrc on cluster)
```

See `EXECUTION_GUIDE.md` for running experiments and `experiments/progress.md` for full status.
