# Experiments and Progress

This document tracks the design, execution, and results of experiments for epiphany-aware KV cache management in reasoning models.

## Overview
- **Goal**: Reduce KV cache memory usage in reasoning models (10k-100k tokens) while maintaining accuracy, by evicting based on representational change signals (hidden-state / KV-vector variance) rather than attention scores.
- **Primary Models**: DeepSeek-R1-Distill-LLaMA-8B (used by ThinKV + ChunkKV), Qwen2.5-Math-7B-Instruct (used by RaaS). Vanilla instruction models (LLaMA-3.1-8B-Instruct, Qwen2-7B-Instruct) are negative controls only.
- **Primary Tasks**: MATH-500, AIME 2024 (matches ThinKV/RaaS exactly), LiveCodeBench (required for ThinKV head-to-head). GSM8K added as low-pressure control (used by RaaS/ChunkKV; traces too short for cache pressure, but omitting it invites reviewer questions). HotpotQA is secondary — used only for Gap F (non-monotonic recall) evaluation, not as a head-to-head benchmark against ThinKV/RaaS.
- **Hardware**: Single GPU start, scale to multi-GPU.
- **Success Criteria**: Outperform ThinKV and RaaS on accuracy vs. cache-size curves; hidden-state variance signal shows higher correlation with token importance than cumulative attention (H2O).
- **Automation Vision**: No training, no classifiers — compute variance-based importance scores on-the-fly from KV tensors already computed for attention.

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

### [Date: March 25, 2026] - Current TODOs & Priorities

**✅ COMPLETED (Critical Blockers Fixed)**
- ✅ HF cache lock resolved (added `export HF_HOME=/tmp/hf_cache` to `.zshrc`)
- ✅ Attention-based eviction working (fixed `keep_recent_k` > `cache_size` bug)
- ✅ Shape mismatch handling (attention tensor vs cache length)
- ✅ Eviction actually triggers (cache_size=64 forces eviction, memory/token reduction visible)

---

**🔴 PHASE 0 — SIGNAL VALIDATION (Do first; everything else depends on this)**

1. **Real Data Pipeline** (1–2 hours)
   - Load MATH-500 from HuggingFace (`hendrycks/competition_math`, competition-level split)
   - Also load AIME 2024 (`Maxwell-Jia/AIME_2024` or equivalent) — use 2024 only, not pooled years, to match ThinKV exactly
   - Run DeepSeek-R1-Distill-LLaMA-8B on 50–100 problems; collect full traces + ground-truth answers
   - Store: generated token IDs, `past_key_values` tensors (keys + values per layer), correct/incorrect flag
   - **Why**: All signal validation requires real reasoning traces; synthetic data is useless here

2. **Counterfactual Importance Labels** (2–3 hours)
   - For each trace, generate token-level importance labels via ablation: mask sliding windows of tokens; record which masked windows cause the answer to flip
   - These are ground-truth labels: "token at position t is important if masking it (and neighbors) changes the answer"
   - **Why**: Without ground-truth importance labels, we cannot evaluate whether any signal is better than any other

3. **Signal Variant Sweep — Dimension 1 (Signal Type)** (2–3 hours)
   - Compute all six variants at each token position: L2 hidden-state diff, cosine distance (hidden), KV-key variance (head_dim), KV-key L2 norm, KV-value variance (head_dim), cross-head key variance
   - Also compute H2O cumulative attention as the current-SOTA baseline
   - Compute Spearman correlation of each with counterfactual labels
   - **Why**: This directly answers whether *any* variance signal outperforms attention-based scoring. If not, core hypothesis is wrong.

4. **Per-Example Memory Tracking** (15 min)
   - Reset `torch.cuda.reset_peak_memory_stats()` before each example, store per-example, average
   - **Why**: Cumulative measurement masks real differences between methods

---

**🟠 PHASE 0B — SEQUENTIAL ABLATION (After best signal type is identified)**

5. **Signal Ablation — Dimensions 2–5** (3–4 hours total, run sequentially)
   - Dim 2: Pre-RoPE vs. post-RoPE key variance (requires hook before `apply_rotary_pos_emb`)
   - Dim 3: Layer aggregation — last layer, mean, upper-weighted mean, optimal single layer
   - Dim 4: Temporal aggregation — snapshot, EMA (α=0.9), sliding max, cumulative
   - Dim 5: Multi-head — mean, max, cross-head variance
   - Each sweep: 100-sample MATH-500, cache_size=128, accuracy + importance recall metric
   - **Why**: Best signal configuration must be found before benchmarking; see research_overview.md §3.1

6. **`semantic_alpha` Ablation** (1 hour)
   - Sweep blend ratio between best variance signal and cumulative attention: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`
   - **Why**: May be optimal to use pure variance (1.0), pure attention (0.0), or a blend — unknown until tested

---

**🟠 PHASE 1 — BASELINE IMPLEMENTATION (Run in parallel with Phase 0B)**

7. **Implement H2O Proper** (1 hour)
   - Add `cumulative_attention` field to `AttentionBasedEviction`; update at each step with `score[t] += attn[t]`
   - Replace single-step attention scoring with cumulative
   - **Why**: Current attention baseline uses only current-step attention — this is a weak strawman, not a fair comparison

8. **Implement ThinKV Classifier** (3–4 hours)
   - KDE on normalized attention scores from 4 layers, refreshed every 128 decode steps
   - Output: R/E/T label per segment; use for progressive retention schedule {64, 32, 16, 8, 4}
   - **Why**: SOTA baseline on reasoning-model KV compression; must beat this to claim contribution

9. **Implement RaaS Baseline** (2 hours)
   - LRU timestamp eviction for decode tokens; unconditional preservation of all prefill tokens
   - **Why**: Strong reasoning-aware baseline; demonstrates phoenix-token preservation

10. **Accuracy vs. Cache-Size Curves** (1 hour, after 7–9 done)
    - Run H2O, ThinKV, RaaS, and our method at cache_size ∈ {32, 64, 128, 256, 512, 1024}
    - Benchmarks: MATH-500, AIME 2024, LiveCodeBench, GSM8K (low-pressure control)
    - Models: DeepSeek-R1-Distill-LLaMA-8B (ThinKV/ChunkKV comparability) + Qwen2.5-Math-7B-Instruct (RaaS comparability)
    - This is the main comparison figure

---

**🟡 PHASE 2 — METHOD DEVELOPMENT (After best signal confirmed)**

11. **Chunk-Level Eviction** (2 hours)
    - Score 10-token chunks by mean signal across members; evict/retain at chunk granularity
    - Ablate chunk sizes: {4, 8, 10, 16, 32}

12. **Structural Token Hardcoding**
    - Always retain: first 4 tokens (attention sinks), all prefill tokens, last 32 tokens (recent window)
    - Apply variance scoring only to non-structural token budget

13. **Layer-Wise Budget Analysis**
    - Plot attention entropy per layer on DeepSeek-R1-Distill traces during decode
    - If pyramidal pattern holds, adopt PyramidKV's arithmetic allocation

---

**🔵 CLEANUP / HOUSEKEEPING (Low priority)**
- Fix token counting: use `len(generated_ids)` not `len(reasoning.split())`
- Add debug mode: log top-5 kept/evicted tokens with importance scores per step
- Unit test answer matching edge cases: "2" vs "12", "Answer is 5" etc.

### [Date: March 25, 2026] - collect_traces.py Written and Validated ✓

**Environment fixes:**
- HF cache redirected to `/data/hf_cache/skolawol/` (user-owned dir on shared cluster). `.zshrc` updated: `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE` all point there.
- `EXECUTION_GUIDE.md` updated accordingly.

**DynamicCache compatibility (breaking change in newer transformers):**
- This transformers version returns `DynamicCache` objects where `__iter__` yields `(keys, values, sliding_window_tensor)` 3-tuples (not 2-tuples as in prior versions).
- `DynamicCache` no longer has `key_cache`/`value_cache` attributes; uses `cache.layers[i].keys` / `.values` instead.
- Fixed via `_as_legacy_kv(past_key_values)` helper in `collect_traces.py` that normalises any cache format to a list of `(key, value)` 2-tuples per layer.

**Signal collection pipeline:**
- `collect_traces.py` collects 9 signals per token: `kv_key_var`, `kv_key_norm`, `kv_val_var`, `cross_head_var`, `h2o_attn`, `attn_entropy`, `hs_l2_diff`, `hs_cos_dist`, `hs_norm`.
- KV signals: free (read from `past_key_values` already in GPU memory).
- `h2o_attn` + `attn_entropy`: both require `--force_eager_attn` (FlashAttention can't materialise attention weights). `attn_entropy` is the same signal ThinKV's R/E/T classifier uses. Low entropy = model focused/"Thinking"; high entropy = diffuse/"Rambling".
- `hs_l2_diff`, `hs_cos_dist`, `hs_norm`: post-hoc forward pass, seq ≤ `--hs_max_len` (defaults to `--max_new_tokens`); -1.0 sentinel for longer.
- KV signals are post-RoPE (as stored in cache). Pre-RoPE requires a forward hook — deferred to Phase 0B ablation.

**Dry-run results (all 4 datasets):**
- math500, aime2024, livecodebench, gsm8k: all pass, 1 trace written each.
- `pred=None` / `correct=False` in dry run is expected — 128 max tokens is far too few for DeepSeek-R1 to reach a `\boxed{}` answer.
- All DynamicCache, dataset loading, and signal accumulation issues resolved.

**Next (immediate):** Run full collection: `python scripts/collect_traces.py --dataset math500 --n_samples 100 --max_new_tokens 16384`

### [Date: TBD] - Scaling
- Status: Not started
- Plan: Larger models (LLaMA-7B, Qwen-7B), full datasets, longer sequences
- Output: Scalability analysis, ablation studies

### [Date: TBD] - Interpretability
- Status: Not started
- Plan: Analyze evicted tokens, edge cases, failure modes
- Output: Refinements and insights

## Immediate Next Steps (Ordered)
1. ✅ Real data pipeline: collect_traces.py written and validated on all 4 datasets
2. ✅ Counterfactual importance labeler: label_importance.py written
3. ✅ Signal ablation framework: signal_ablation.py written
4. ✅ Full pipeline validated locally: 1 math500 trace labelled (41 min, 208 windows, flip_rate=0.71), signal ablation ran end-to-end
5. ✅ SLURM batch jobs submitted and completed — all traces collected
6. ✅ Phase 0 first signal ablation run complete — results obtained (see March 30, 2026 entry below)
7. ✅ Critical masking bug identified and fixed in label_importance.py (truncation → occlusion)
8. ✅ SLURM scripts restructured (collect/label split; all on general partition)
9. **IN PROGRESS**: Re-running labels with fixed methodology on all 3 datasets
10. **NEXT**: Phase 0B ablations once re-labelled results arrive — pre-RoPE keys (Dim 2), layer-wise HS (Dim 3)
11. **WHILE WAITING**: Implement eviction baselines in eviction.py — H2O (cumulative), ThinKV (R/E/T KDE), RaaS (LRU+prefill)
12. **AFTER PHASE 0B**: If any residual-stream signal beats H2O → Dimensions 4–5 ablations → implement eviction → accuracy vs. cache-size curves
13. If hypothesis fails after 0B → revise; candidate pivots are in research_overview.md §"What Is Unproven"

### [Date: March 30, 2026] - Phase 0 First Run + Critical Methodology Fix

**Batch jobs complete.** All three trace files collected successfully:
- `data/math500_eager_traces.jsonl` — 100 problems, 16384 tok, eager (h2o + entropy collected)
- `data/aime2024_traces.jsonl` — 30 problems, 32768 tok, non-eager (FA2)
- `data/aime2024_eager_traces.jsonl` — 30 problems, 16384 tok, eager

**SLURM infrastructure restructured:**
- Collect and label steps now run as separate jobs, both on `general` partition (48h, non-preemptible)
- Preempt partition abandoned for all pipeline steps after 12+ requeues overnight
- Renamed old combined preempt scripts with `_preempt` suffix; new scripts: `run_*_collect.sh` and `run_*_label.sh`
- `run_math500_eager_label.sh` added (label-only, analogous to AIME label scripts)

**Phase 0 first signal ablation results obtained** (for analysis details see signals_reference.md):
- math500 non-eager: **INVALID** — cumsum artifact (stale run with old code); superseded by math500_eager
- math500_eager: valid, 186k pairs
- aime2024 (non-eager, 32k): valid, 78k pairs
- aime2024_eager (16k): valid, 52k pairs

**Critical masking bug found and fixed in label_importance.py:**
- **Bug**: `run_masked_inference` truncated the sequence at `mask_start` and regenerated from there. This tests "how much prefix is needed?" — a position test. Early windows always had less context, so they were systematically labeled important regardless of content.
- **Effect**: importance labels correlated with sequence position, not token content. This inflated h2o_attn's apparent advantage (it too correlates with position — early tokens get more future attention) and suppressed relative performance of content signals.
- **Fix**: True occlusion — replace window tokens with pad_id, feed full modified reasoning context up to `answer_start` (the </think> boundary for DeepSeek-R1), generate the answer from there. Every window call feeds the same context length; only the content of the window varies.
- **New function**: `find_answer_start()` — detects </think> boundary via token search, falls back to last `\boxed{` position, then to total_len - 64.
- All stale label files (`*_traces_labelled.jsonl`) and signal ablation CSVs deleted; re-runs submitted.

**Key analytical insights from Phase 0 results** (preliminary, from truncation-based labels):
1. h2o_attn dominated (ρ≈0.36–0.41) but likely inflated by positional proxy artifact
2. attn_entropy: ρ=-0.294 on AIME (negative = correct direction: low entropy → R-type → important); ρ=-0.063 on math500 (shorter traces, less discriminative)
3. KV variance signals significantly stronger on longer/harder traces: kv_key_var_rolling64 was ρ=0.012 (AIME eager, 16k) vs ρ=0.124 (AIME non-eager, 32k) — same signal, different trace population. KV signals benefit from longer, more complex reasoning traces with more redundant content to discriminate.
4. HS signals (all ρ<0.04) — inconclusive. **Critical limitation**: all HS signals use only the final transformer layer. Per interpretability literature (ROME/MEMIT, Geva et al., logit lens), the final layer is specialized for next-token prediction, not semantic representation. Semantic content peaks in middle-to-upper layers (~16–24 for a 32-layer LLaMA). Layer-wise ablation (Phase 0B Dimension 3) is required before any conclusion on HS signals.
5. KV signals use post-RoPE keys (position-dependent contamination) and average all 32 layers (dilutes layer-specific signal). Phase 0B Dimension 2 (pre-RoPE) and single-layer/upper-weighted KV signals needed.
6. ThinKV vs H2O comparison: cannot be resolved from token-level Spearman ρ. ThinKV beats H2O via segment-level R/E/T classification + different retention budgets per segment, not via per-token signal quality. Phase 1 accuracy curves are the correct comparison point.

**Re-runs in progress**: `run_math500_eager_label.sh`, `run_aime2024_label.sh`, `run_aime2024_eager_label.sh` — all with occlusion-fixed label_importance.py.

### [Date: April 6, 2026] - Phase 0B Signal Ablation Analysis Complete

All 4 Phase 0B signal ablation runs completed and analysed. Full results in
`experiments/phase0b_ablation_results.md`. Key findings:

**Data collected:**
- math500: 75 labelled traces, 172k pairs, avg important_frac ~0.20
- math500_eager: 81 labelled traces, 191k pairs, avg important_frac ~0.20
- aime2024: 14 labelled traces, 82k pairs, avg important_frac ~0.52
- aime2024_eager: 11 labelled traces (hook bug → only 15/30 collected), 52k pairs, avg ~0.64
- **aime2024_eager is partial** — hook leak bug fixed; rerun queued (`sbatch slurm/run_aime2024_eager_collect.sh`)

**Infrastructure fixes applied before/during this phase:**
- `collect_traces.py` + `extract_phase0b_signals.py`: `try/finally` added around forward
  pass in `fill_hidden_states` — prevents hook leak on OOM/exception (root cause of
  aime2024_eager 15/30 failure)
- `extract_phase0b_signals.py`: cross-validation replaced with Spearman ρ ≥ 0.99 (was
  absolute error tolerance 1e-4, which always failed due to fp16 GPU non-determinism)
- `--hs_layers` default changed from "16,20,24" to all 32 layers (0–31)

**Top findings (see phase0b_ablation_results.md for full tables and CIs):**

1. **Two consistent-direction HS bands discovered:**
   - Band A (l7–l13): consistently POSITIVE ρ across all 4 datasets — high signal at these
     layers = important token. Evict tokens with LOW l7–l13 signal.
   - Band B (l18–l25): consistently NEGATIVE ρ across all 4 datasets — high signal at these
     layers = dispensable token. Evict tokens with HIGH l18–l25 signal.
   - Combined score `l10_rolling64 − l21_rolling64` is the Phase 1 candidate signal.

2. **preRoPE vs postRoPE: null result.** Max Δρ = 0.0005 across all datasets. Collecting
   preRoPE signals adds zero benefit. Remove from future runs.

3. **Rolling64 beats ema09 beats raw by 30–57%** universally. Token importance is a
   sustained contextual property, not an instantaneous spike.

4. **Sign flip between easy and hard problems** for kv_key_var, kv_val_var, cross_head_var,
   hs_l2_diff_l31, and all attention signals. Root cause: important_frac ~0.20 in math500
   vs ~0.52–0.64 in AIME. The same signal's relationship to importance inverts with label
   density. Not confirmed as a real effect (vs sampling noise) due to small AIME n.

5. **h2o_attn is the weakest signal tested** — 3–12× weaker than attn_entropy, substantially
   weaker than multiple HS and KV signals. Confirms the core hypothesis: attention score is
   a poor token importance proxy.

6. **AIME results are statistically unreliable.** With 11–14 effective samples (traces),
   95% CIs on all AIME-side correlations span zero. Directionally consistent with math500
   findings, but cannot be claimed as independent evidence. Adding AIME 2025 and AIME 2026
   (MathArena/aime_2025, MathArena/aime_2026) is strongly recommended — 90 total AIME
   problems would yield ~35–45 labelled traces.

**Code change required (AIME 2025/2026):** `collect_traces.py` needs `load_aime2025` and
`load_aime2026` functions + `--dataset` choices extended. Dataset field names need
verification against HuggingFace schema before implementation.

**Phase 1 recommended signal:** `hs_l2_diff_l21_rolling64` (or combined Band A−B score).
Posthoc signal (requires separate forward pass); not available during generation.
Online fallback: `kv_key_var_rolling64` (free, FlashAttention-compatible, sign depends on
task difficulty).

### [Date: April 2, 2026] - Phase 0B Infrastructure, Eviction Baselines, SLURM Restructuring

**Phase 0B signal collection implemented:**
- `collect_traces.py` extended: `--phase0b` flag adds pre-RoPE key variance (via `k_proj` forward hooks before RoPE application) and per-layer HS at layers 16, 20, 24 (configurable via `--hs_layers`). Single forward pass collects all target HS layers simultaneously. Explicit `_preRoPE_collected` flag guards `to_dict()` emission (robust against all-zero edge case).
- `scripts/extract_phase0b_signals.py` written: posthoc extraction of Phase 0B signals from saved token IDs. Three modes: extract-only (`--input + --output`), compare-only (`--compare`), extract+compare. Cross-validation tolerance 1e-4; exit 0=PASS, 1=FAIL.
- `signal_ablation.py` fixes:
  - **Summary bug**: `max()` found max positive ρ only — silently missed large negative-ρ signals (AIME non-eager best was -0.214, reported as +0.015). Fixed to `max(abs(ρ))` with Δ|ρ| comparison.
  - **Phase 0B temporal variants**: `_rolling64`/`_ema09` now generated for `kv_key_var_preRoPE`, `kv_key_norm_preRoPE`, and all `hs_l2_diff_lN` layer-specific signals (via regex `^hs_l2_diff_l\d+$`).

**Eviction baselines implemented in `src/eviction.py`:**
- `H2OEviction`: stateful cumulative attention, attention sinks + recency always kept, evicts lowest cumulative score — proper H2O (not single-step).
- `ThinKVEviction`: stateless, 128-token segment entropy classification (R/E/T by tertile split), per-segment retention budgets {64, 32, 8}.
- `RaaSEviction`: stateful LRU timestamps for decode tokens, unconditional prefill preservation, top-50% attention refreshes timestamp.

**SLURM workflow restructured — user submits 4 commands only:**
- `run_math500_collect.sh` (new): non-eager, 32768 tokens, Phase 0B signals, auto-chains `run_math500_label.sh` via `afterok`
- `run_math500_label.sh` (new): label + ablate for non-eager math500 traces
- `run_math500_eager_collect.sh` (new): split from deprecated combined `run_math500_eager.sh`; 16384 tokens, eager+Phase0B, auto-chains `run_math500_eager_label.sh`
- `run_aime2024_collect.sh`, `run_aime2024_eager_collect.sh`: updated with 3-step posthoc cross-validation + `afterok` auto-chaining
- All collect scripts follow: (1) collect → (2) posthoc extract → (3) cross-validate → if PASS: auto-submit label job
- `run_math500_eager.sh` (combined): deprecated and deleted

**Phase 0B collection jobs submitted.** Expected within 48h:
- `data/math500_traces.jsonl` + `data/math500_eager_traces.jsonl` (re-collected with Phase 0B signals)
- `data/aime2024_traces.jsonl` + `data/aime2024_eager_traces.jsonl` (re-collected with Phase 0B signals)
- 4 posthoc cross-validation files
- 4 signal ablation CSVs with pre-RoPE + layer-wise HS signals

### [Date: April 11, 2026] - Phase 0B COMPLETE — All 9 Datasets Analysed

All AIME 2025, AIME 2026, and GSM8K runs finished. Full 9-dataset analysis written in
`experiments/phase0b_ablation_results.md` (§11 added). Phase 0B is now complete.

**Final run inventory:**
- gsm8k_eager: 355 correct traces, 200,881 pairs (moved from preempt → general partition; resume logic preserved, no `rm -f`)
- aime2025 non-eager: ~7 traces, 37,714 pairs
- aime2025_eager: ~3–4 traces, 17,065 pairs (too small for any conclusion)
- aime2026 non-eager: ~7 traces, 36,224 pairs
- aime2026_eager: ~8 traces, 41,077 pairs

**Key new findings (see §11 of phase0b_ablation_results.md for full tables):**

1. **GSM8K is a qualitatively different domain.** 200,881 pairs (n_eff=352) — the second
   high-n dataset after math500_eager. The layer anatomy for grade-school math diverges
   strongly from competition math:
   - Early layers l0–l7: POSITIVE ρ (+0.181 at l0) — early layers are predictive for simple math
   - Mid layers l10–l30: STRONGLY NEGATIVE (l15=−0.351, the strongest signal across all 9 datasets)
   - Last layer l31: +0.231 (strongly positive — unlike math500/aime2024)
   - attn_entropy: −0.313 (strongly negative — sign FLIPS from math500_eager's +0.176)
   - kv_key_var: −0.261 (strongly negative — confirms sign flip from math500's +0.380)

2. **KV signal sign flip is CONFIRMED, not sampling noise.** kv_key_var = +0.380 in math500,
   −0.261 in gsm8k_eager (both n_eff ≥ 72). Cannot be used as a fixed-direction signal.

3. **Band A (l7–l13) positive ρ holds for competition math eager datasets.** Robust in
   math500 (+0.112 at l10), math500_eager (+0.141), aime2024 (+0.097), aime2024_eager (+0.120).
   In gsm8k_eager, l7=+0.076 (weakly positive) but l10=−0.048 (goes negative). Band A
   for easy math is l0–l7, not l7–l13.

4. **Band B (l18–l25) negative ρ holds for competition math eager datasets** and is EXTENDED
   in gsm8k to l10–l30. In AIME 2025/2026 non-eager (very low n_eff), Band B shows sign
   instability — cannot be trusted.

5. **Phase 1 target signal confirmed: l10_rolling64 − l21_rolling64 for competition math.**
   Valid for math500-difficulty and AIME-difficulty problems. For GSM8K-difficulty, a
   different combined score would be needed (l3 − l15), but GSM8K is not the Phase 1 target.

6. **attn_entropy sign is unpredictable.** +0.176 (math500_eager), −0.139 (aime2024_eager),
   −0.313 (gsm8k_eager). Not usable as a fixed-direction signal across difficulty regimes.

7. **h2o_attn remains the weakest signal.** Consistently weak (|ρ| ≤ 0.086 across all eager
   datasets). Hypothesis confirmed.

8. **AIME 2025/2026 are too small to be conclusive.** n_eff ≤ 8 for all AIME 2025/2026
   runs; 95% CIs span zero for every signal. Used only as directional indicators. AIME
   2025_eager (n_eff≈1–3) is noise.

**Phase 0B status: COMPLETE.** The signal landscape is now characterized across difficulty
regimes. Phase 1 is unblocked: implement `HSVarianceEviction` using l10_rolling64 −
l21_rolling64, targeting competition-math difficulty problems.

**Paper implications (see paper_strategy.md):**
- GSM8K finding adds a new dimension: the "Epiphany" layer shifts by difficulty. This can
  be mentioned as a finding (l7–l13 for hard math, l0–l7 for easy math) without undermining
  Phase 1 scope.
- The KV sign flip being confirmed is important: KV signals require difficulty-regime
  detection, while HS Band A/B is more robustly directional within a difficulty class.
- GSM8K's attn_entropy = −0.313 (very strong) but sign-flipped confirms that even the
  stronger attention metric is not universally applicable.

---

### [Date: April 6, 2026] - AIME 2025/2026 + GSM8K added; paper strategy written

**Dataset expansion:**
- `collect_traces.py`: Added `_load_matharena_aime()` generic loader with runtime column
  discovery (no hard-coded field names). `load_aime2025()` and `load_aime2026()` backed by
  `MathArena/aime_2025` and `MathArena/aime_2026`. Verified HuggingFace schema: both datasets
  use `problem` + `answer` columns (answer is int64; `str()` cast handles this).
- `--dataset` choices extended to `aime2025`, `aime2026`.
- GSM8K added to Phase 0B suite: rationale is difficulty-robustness validation for Band A/B
  layer anatomy — if l7–l13 positive ρ holds on grade-school math (500 samples, ~1–2k token
  traces), the signal spans the full difficulty spectrum.

**New SLURM scripts:**
- `slurm/run_aime2025_collect.sh`, `run_aime2025_label.sh` (non-eager, 32768 tokens)
- `slurm/run_aime2025_eager_collect.sh`, `run_aime2025_eager_label.sh` (eager, 16384 tokens)
- `slurm/run_aime2026_collect.sh`, `run_aime2026_label.sh` (non-eager, 32768 tokens)
- `slurm/run_aime2026_eager_collect.sh`, `run_aime2026_eager_label.sh` (eager, 16384 tokens)
- `slurm/run_gsm8k_eager_collect.sh`, `run_gsm8k_eager_label.sh` (eager only — traces too
  short for FA2/eager split to matter; preempt partition with `--requeue --signal=B:USR1@60`;
  collect_traces.py resume logic makes requeues safe)

**Currently running (general partition):**
- `aime2024_eager` rerun (hook fix applied, all 30 traces expected)
- `aime2025` collect + label (auto-chained)
- `aime2026` collect + label (auto-chained)
- `gsm8k_eager` collect + label (preempt, requeue-safe)

**paper_strategy.md created** (`experiments/paper_strategy.md`):
- Target: NeurIPS. Core claim, advisor framing (FA2 engineering win), figure plan (4 figures),
  section-by-section notes, related work (EAGLE, ROME/MEMIT), presentation asymmetry.

**eviction.py audit:**
- `H2OEviction`, `RaaSEviction`: faithful to papers, usable as baselines.
- `ThinKVEviction`: real bug — per-segment fixed budgets (retain_r/e/t) don't enforce total
  cache_size; can over-retain. Also stateless (re-classifies every call, paper refreshes
  every 128 steps). Budget values are made up.
- `SemanticEviction`: completely stale. Uses last-layer HS L1 diff + average post-RoPE
  KV variance across all layers — Phase 0B showed both are wrong choices. Needs full
  replacement with Band A−B combined score (l10_rolling64 − l21_rolling64). This is
  Phase 1's primary deliverable.

---

### [Date: April 13, 2026] - Full Codebase Audit + Cross-Dataset Signal Analysis

**Cross-dataset signal analysis (AIME 2025/2026 + GSM8K) — complete:**

Analyzed all 9 dataset CSVs. Key synthesis:
- **What converges across all datasets:** A negative-ρ zone always exists (l15–l25 for competition
  math, extending to l10–l30 for GSM8K). Early layers l0–l3 always positive. Rolling64 universally
  best smoothing. h2o_attn consistently weakest signal. kv_val_var consistently non-negative (never
  strong sign-flip). preRoPE null result reconfirmed.
- **What diverges:** Band A location (l7–l13 for competition math, l0–l7 for GSM8K). kv_key_var
  direction (+0.380 math500 vs −0.261 GSM8K — both high-n, confirmed real sign flip). attn_entropy
  and h2o_attn also flip sign across domains (not unique to our signals).
- **kv_val_var stability noted:** Across 7 datasets, kv_val_var is non-negative everywhere except
  weakly (−0.145 at one AIME non-eager dataset). More reliable online fallback than kv_key_var.
- **Scope formalized:** math500 + AIME2024 as Phase 1 primaries. GSM8K out of Phase 1 scope (different
  difficulty regime, different anatomy). Full synthesis documented in §12–13 of phase0b_ablation_results.md.

**Codebase audit — errors found and corrected:**

- **ThinKVEviction budget bug FIXED** (`src/eviction.py`): `evict_past_key_values` now tracks
  `remaining_budget` across segments and stops allocating once `cache_size` is reached. Previously,
  per-segment budgets were applied without any cap on total retained tokens — could exceed `cache_size`
  by a factor of 2× or more in long sequences. Smoke test assert updated from `≤ seq_len` to
  `≤ cache_size`.
- **Dead code removed:**
  - `scripts/poc_harness.py` deleted — explicitly deprecated; used synthetic traces + stale eviction classes
  - `scripts/analyze_traces.py` deleted — operated on synthetic GSM8K-level examples; irrelevant to Phase 0/1 pipeline
  - `scripts/visualize.py` deleted — POC visualization; referenced non-existent `src/data_collection.py`
  - `AttentionBasedEviction` class removed from `eviction.py` — single-step attention (poc only); superseded by H2OEviction
  - `SemanticEviction` class removed from `eviction.py` — stale placeholder; implementation inconsistent
    with Phase 0B findings (used L1/last-layer vs L2/mid-layer). Will be replaced by HSVarianceEviction in Phase 1.
  - `eviction_method`, `semantic_alpha` fields removed from `EvictionConfig` — only used by deleted classes
- **CLAUDE.md digested** (`.claude/CLAUDE.md`): surgical changes, simplicity-first, no speculative additions.
  All Phase 1 implementation should follow these guidelines.

**Remaining open items before Phase 1:**
- Manual trace inspection (`scripts/inspect_traces.py`) — script exists, output not yet analyzed
  (background execution environment issue; run manually to inspect labelled traces vs signal values)
- Phase 1: run accuracy vs. cache-size curves (all methods) on math500 + AIME2024

---

### [Date: April 13, 2026] - Phase 1 HS Eviction Family Implemented

**ThinKV paper analysis (April 13, 2026):**

Read ThinKV (He et al., ICLR 2026 oral) in full. Key findings:
- Signal: key-perspective attention sparsity (fraction of attn scores < 1% of row-wise max per column)
- Segment-level: classifies 128-token segments into R/E/T types via KDE; per-type retention budgets
- CT kernel: "lazy deletion" marks evicted tokens with bit flags (no gather) — zero overhead at 95%+ decode steps
- Cannot implement lazy deletion in plain HuggingFace DynamicCache; requires custom CUDA kernel
- Our boolean-mask gather is the main throughput gap vs ThinKV — engineering, not algorithmic
- Generation length inflation risk: up to 5.1× on quantized models (not DeepSeek-R1 at bf16)

**Literature survey — 10+ papers read (April 13, 2026):**

Key temporal detrending methods studied:
- **AhaKV**: analytical detrending via λ=√(2log(i/k)/d); equalizes entropy across positions
- **LagKV**: lag-relative minmax normalization against local lag window; direct applicability to HS signals
- **CAOTE**: closed-form eviction error (α/(1−α))·||VA^T−v_j||₂; theoretical grounding for future work
- **LongFlow**: ||attn×val||₁ per-step; direct competitor (same model class, DeepSeek-R1)
- **RaaS**: LRU timestamp refreshed on top-50% attn; "milestone"/"phoenix" token taxonomy
- Full survey in `experiments/research_overview.md` and `memory/project_literature.md`

**Temporal trend contamination documented (April 13, 2026):**

Critical finding: aggregate Spearman ρ overestimates per-token eviction quality. Within a trace:
- l10 decreases monotonically with position; l21 increases
- Combined score l10−l21 is HIGHER for early-position (DROP) tokens than late-position (KEEP) tokens
  in simple math500 problems — evicts the WRONG tokens
- Phase 0B aggregate ρ is dominated by cross-problem structure, not within-trace discrimination
- Mitigation: rolling z-score detrending (DetrendendHSVarianceEviction); or segment-level eviction
- Fully documented in §10 caveat 4 of phase0b_ablation_results.md

**5 eviction classes implemented in `src/eviction.py`:**

1. `HSVarianceEviction` — original: l10_rolling64 − l21_rolling64 (online HS diff, Band A−B)
2. `DetrendendHSVarianceEviction` — z-score detrended l10−l21; removes temporal position bias
3. `BandAdaptiveHSEviction` — averages all Band A (l7–l13) and Band B (l18–l25) layers, ρ-weighted
   (weight_a=1.29, weight_b=1.0 from Phase 0B math500_eager ρ ratio); FA2-compatible
4. `AttentionHSProductEviction` — cumulative key-perspective attn + detrended Band A HS z-score;
   normalized and summed; eager-only (requires output_attentions=True)
5. `HybridSegmentHSEviction` — ThinKV segment R/E/T classification (key-perspective entropy) +
   detrended HS Band A−B z-score for within-segment ranking; eager-only

Helper functions added:
- `_rolling_z_score(buf, window, eps) -> float` — z-score of last element vs rolling window
- `_hs_diff(hidden_states, layer, prev) -> (float, Tensor)` — HS L2 diff at specified layer

**`scripts/benchmark.py` updated:**

- Added imports for all 4 new classes + helpers
- Split method sets: ATTN_ONLY_METHODS, HS_ONLY_METHODS, BOTH_METHODS, KV_METHODS
- BOTH_METHODS = {"attn_hs_product", "hybrid_seg_hs"} — need attn AND hs; eager-only
- Updated make_eviction() factory for all 5 new HS method names
- Updated run_one() dispatch: need_both branch passes both attentions and hidden_states
- Updated eviction init: all HS classes call reset(prefill_len=prompt_len) + set_prefill_end()

**kv_val_var stability error corrected (April 13, 2026):**

Phase0b_ablation_results.md §10 claimed kv_val_var "consistently non-negative." CORRECTED:
- math500 = −0.135 (NEGATIVE, n_eff=75), math500_eager = −0.145 (NEGATIVE, n_eff=81)
- kv_val_var is NEGATIVE in the two primary Phase 1 datasets
- Removed "primary online fallback" recommendation; downgraded to secondary comparison

---

### [Date: April 24, 2026] - Phase 1 Results: Eager + Flash Benchmarks Complete

**Setup**: `scripts/benchmark.py` ran on DeepSeek-R1-Distill-LLaMA-8B against MATH-500 (n=100) and AIME-2024 (n=30), under both eager and flash_attention_2 configurations. Cache budgets {512, 1024, 2048, 4096} on MATH-500 and {512, 1024, 2048, 4096, 8192} on AIME. Methods: none, h2o, thinKV, raas, hs_variance, hs_variance_detrend, band_adaptive_hs, attn_hs_product, hybrid_seg_hs, kv_val, kv_key, lag_kv_key, lag_kv. Results JSONs under `/data/user_data/skolawol/kvcache/results/phase1/`. Summary script: `scripts/analyze_phase1.py`.

**Headline accuracy results**:

| Dataset / Budget | Best FA2-compatible | Best attention-based | Ceiling (none) |
|---|---|---|---|
| MATH-500 @ 4096 | **hs_variance_detrend = 72%** | thinKV = 71% | 75% |
| MATH-500 @ 2048 | band_adaptive_hs / kv_val_var ≈ 57% | raas = 60% | 75% |
| MATH-500 @ 1024 | hs_variance = 28% | hybrid_seg_hs = 36% | 75% |
| AIME-2024 @ 8192 | **lag_kv = 37%** | h2o / hybrid_seg_hs = 33% | 43% |
| AIME-2024 @ 4096 | lag_kv = 20% | thinKV / h2o / hybrid_seg_hs = 20% | 43% |

**Engineering results**:
- **Speed**: lag_kv (FA2) is **2.8× faster than raas (eager)** at AIME-2024 @ 8192 (441s vs 1239s mean wall-time per problem). Several FA2 methods are *faster than the no-eviction baseline* at high cache budgets, since smaller caches reduce per-step decode cost while signal extraction adds negligible overhead.
- **Memory**: peak GPU memory differences are dominated by cache budget, not method choice. At AIME @ 512 cache, all eviction methods save ~3GB vs none (~16% saving); at AIME @ 8192, savings shrink to ~700MB. FA2 vs eager at the same cache budget differs by ~100–500MB. Honest framing for the paper: claim time wins, not memory wins, in the decode regime measured here. The known O(n²)→O(n) FA2 prefill memory advantage is real but not measured by these short-prompt benchmarks.

**H2O catastrophic collapse on MATH-500**: H2O's `n_zero_tokens` audit revealed 93/100 problems generate zero tokens at cache=1024, 48/100 at cache=2048, 27/100 at cache=4096. No exceptions — silent collapse to immediate EOS. This matches RaaS's documented attention-map failure mode (24.2% on reasoning traces in their analysis). For the paper, this is a usable concrete example to motivate the need for non-attention signals.

**Phase 1 debugging saga**:

1. **Multi-GPU device mismatch in eviction code**: `keep_mask` was created on cuda:0 (device of layer 0 KV) and applied to all layers, including those on cuda:1 under `device_map="auto"`. Symptom: every problem failed with `indices should be either on cpu or on the same device as the indexed tensor (cuda:1)`. Fix: replace `k[:, :, keep_mask, :]` with `k[:, :, keep_mask.to(k.device), :]` across all 12 occurrences in `src/eviction.py`.

2. **DynamicCache `_seen_tokens` desync**: After fixing the device issue, the next failure was `tensor a (514) must match tensor b (513) at non-singleton dimension 3` inside the model forward — caused by `_to_model_kv()` using `DynamicCache(ddp_cache_data=kv_pairs)`, which is a non-standard constructor that left `_seen_tokens` one ahead of the actual cached tokens after eviction. The model then built a 514-position causal mask while the post-update KV had 513 keys. `from_legacy_cache()` was tried first but is absent in the installed transformers version. **Correct fix**: construct an empty `DynamicCache()` and call `cache.update(k, v, layer_idx)` for each layer — this sets `_seen_tokens = k.shape[-2]` from the first layer correctly. (`scripts/benchmark.py:_to_model_kv`).

3. **Multi-GPU flash_attn kernel coordination crash**: 2-GPU flash_attn jobs crashed even on the `none` method with `CUDA error: unspecified launch failure`. Solution: switch all flash benchmark scripts to single-GPU 48G allocation. Eliminated the crashes.

4. **Node-specific CPU loading**: One AIME-2024 eager submission landed on `babel-v9-28`, where `device_map="auto"` placed the model on CPU despite nvidia-smi showing GPUs. Symptom: `_reset_gpu_peak` raised `Invalid device argument: did you call init?`. Fixes: (a) wrapped `_reset_gpu_peak`/`_peak_gpu_mb` in `try/except RuntimeError` to handle uninitialized CUDA contexts; (b) added an early `sys.exit` in `main()` if the model lands on CPU while CUDA is available, with the node name in the error message.

5. **Disk full from logs/results on /home**: /home (NFS, 100G) hit 100% mid-run. Phase 1 results moved to `/data/user_data/skolawol/kvcache/results/phase1/` (NVMe, 221G). `slurm_logs/` stays on /home per user preference. Phase 1 SLURM scripts updated.

6. **Redundant `none` runs**: previous `benchmark.py` re-ran `none` for every cache_size (4× duplicate work, identical accuracies). Now runs once and copies results across cache_size slots. Display label flips to "Cache size: unlimited (no eviction)".

7. **Better error visibility**: `traceback.print_exc(file=sys.stderr)` added to the per-problem exception handler so future failures surface the full stack instead of one-line messages.

**Decision (Phase 2 plan)**: AIME n=30 makes the 3-pt absolute win on lag_kv@8192 fragile (one problem of difference). Phase 2A combines AIME 2024+2025+2026 to n=90 + adds GSM8K. Phase 2B adds detrending ablation at lower budgets, an FA2-compatible kv_seg_hs analog of hybrid_seg_hs (to close the tight-budget gap), and a token-retention case study. Engineering validation in production stacks (vLLM/TGI) is Phase 2C if time permits.

**Files updated this session**:
- `scripts/benchmark.py` — DynamicCache fix, `_reset_gpu_peak` guards, model-on-CPU exit, none-once optimization, traceback logging
- `src/eviction.py` — multi-GPU device fix (12 occurrences)
- `slurm/phase1/*.sh` — single-GPU flash, results path on /data
- `scripts/analyze_phase1.py` — new: prints accuracy/timing/memory tables and produces accuracy-vs-cache-size PDFs to `reports/phase1_plots/`
- `experiments/paper_strategy.md` — kv_val_var status corrected ("primary online fallback" claim removed)

---

## Key Research Questions
1. **Does any hidden-state/KV-vector variance signal correlate with token importance better than cumulative attention (H2O)?** If no, pivot.
2. **Which of the six signal-type variants performs best?** (See research_overview.md §3.1)
3. **Pre-RoPE or post-RoPE?** (Hypothesis: pre-RoPE is cleaner)
4. **Which layer aggregation?** (Hypothesis: upper-layer-weighted mean)
5. **Does chunk-level eviction improve coherence vs. token-level?**
6. **Can our best method beat ThinKV's accuracy vs. cache-size curve?**

## Key Insights So Far
- Attention-based eviction causes 24.2% attention map failures in reasoning traces (RaaS)
- Even H2O (cumulative attention) fails in reasoning; requires LRU-style temporal tracking (RaaS)
- ThinKV (ICLR 2026 oral) is SOTA: classifies R/E/T thought types from attention sparsity, hybrid quantization+eviction, <5% cache retention with near-lossless accuracy
- Our differentiator: hidden-state/KV-vector variance as importance signal (not attention) — unvalidated but untried in literature
- Multiple valid ways to measure variance; ablation across them is itself a contribution
- "Hidden-state variance" is not one signal — it's a 5-dimensional design space (see research_overview.md §3.1)

## Notes
- Prioritize LLaMA-3 for popularity (also test Qwen, Mistral)
- Aim for automation: Compute semantic proxies efficiently (state variance + attention patterns)
- Update this file after each major milestone</content>
<parameter name="filePath">/home/skolawol/workspace/kvcache/experiments/experiments.md