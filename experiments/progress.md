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