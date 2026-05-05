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
  eviction.py          — All baselines (H2OEviction, ThinKVEviction, RaaSEviction) and
                         five Phase 1 HS/KV eviction methods
scripts/
  collect_traces.py    — trace collection with Phase 0B signals
  extract_phase0b_signals.py — posthoc HS extraction + cross-validation
  label_importance.py  — counterfactual occlusion labelling
  signal_ablation.py   — Spearman ρ ablation across all signal variants
  inspect_traces.py    — manual trace inspection (labels vs signal values)
  benchmark.py         — Phase 1: accuracy/timing/memory vs cache-size curves
  analyze_phase1.py    — Phase 1 result tables + accuracy-vs-budget plots
slurm/                 — SLURM batch scripts (phase0/, phase1/, setup/)
experiments/
  progress.md          — experiment log and current status
  phase0b_ablation_results.md — full Phase 0B signal ablation results
  signals_reference.md — technical reference for all signals
  paper_strategy.md    — NeurIPS writing and presentation strategy (incl. Phase 1 results)
  research_overview.md — literature context and research design
reports/phase1_plots/  — Phase 1 accuracy curves (PDF)
data/                  — collected traces (gitignored)
results/               — Phase 0B signal ablation CSVs (Phase 1 results live on /data — see EXECUTION_GUIDE.md)
```

## Current Status

- **Phase 0B complete (April 13, 2026):** signal ablation across 9 datasets; Band A/B layer anatomy confirmed; ThinKV budget bug fixed; codebase audit done.
- **Phase 1 complete (April 24, 2026):** all eviction methods implemented; eager + flash benchmarks run on MATH-500 + AIME-2024; analysis tables + plots produced.

### Phase 1 headline numbers (DeepSeek-R1-Distill-LLaMA-8B)

- **MATH-500 @ 4096-token cache**: `hs_variance_detrend` (FA2-compatible) reaches 72% — beating ThinKV (71%) and H2O (67%); ceiling is 75%.
- **AIME-2024 @ 8192-token cache**: `lag_kv` (FA2-compatible) reaches 37% — outperforming every attention-required eviction by 3 absolute points; ceiling is 43%.
- **Speed**: `lag_kv` is 2.8× faster than `raas` at the same cache budget on AIME (441s vs 1239s per problem). Several FA2 methods are *faster than no-eviction* at large cache budgets.
- **H2O collapse**: H2O produces empty generations on 93/100 MATH-500 problems at cache=1024 — matching the attention-map failure mode RaaS documented.

### Phase 2 next

Robustness (combine AIME 2024+2025+2026 to n=90; add GSM8K), per-layer / per-band ablations, and an FA2-compatible analog of `hybrid_seg_hs` for tight-budget regimes. See `experiments/progress.md` for the detailed plan.

## Setup

```bash
pip install -r requirements.txt
# HF cache: /data/hf_cache/skolawol (set in .zshrc on cluster)
```

See `EXECUTION_GUIDE.md` for running experiments and `experiments/progress.md` for full status.
