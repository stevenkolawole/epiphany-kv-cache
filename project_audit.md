# Epiphany-KV-Cache: Four-Stage Audit Prompt Set

**Purpose.** Verify every load-bearing claim in the epiphany-kv-cache project against
the code and the data. Decide what NeurIPS submission to make based on what survives.

**Hand-off.** Run each stage as a separate prompt with a strong code model that has
direct repo access (Claude Code, Cursor with full repo, or an agent with shell + git).
The output of each stage is an input to the next. Do not collapse stages into one
prompt — staging is what prevents the model from reasoning backward from "this should
be a NeurIPS paper."

**Calibration constants** (so the model knows what's at stake without me pre-judging):

- Repo: <https://github.com/stevenkolawole/epiphany-kv-cache>
- Model under study: DeepSeek-R1-Distill-LLaMA-8B (32-layer LLaMA-class).
- Primary datasets: MATH-500 (n=100), AIME-2024 (n=30). Secondary: AIME-2025/2026,
  GSM8K_eager. Eager and FA2 variants exist for most.
- Phase 0B label methodology: counterfactual occlusion. 32-token sliding window with
  stride 16; mask → regenerate → check answer flip vs. original. OR semantics across
  overlapping windows. Window=32 was the granularity of "important."
- Phase 1 implemented eviction methods (in `src/eviction.py`):
  - Baselines: H2OEviction, ThinKVEviction, RaaSEviction
  - HS family: HSVarianceEviction, DetrendendHSVarianceEviction, BandAdaptiveHSEviction,
    AttentionHSProductEviction, HybridSegmentHSEviction
  - KV family: KVValVariance, KVKeyVariance, LagKVKey, LagKV
- Phase 0B "primary signal": `hs_l2_diff_l10_rolling64 − hs_l2_diff_l21_rolling64`.
- Phase 1 headline numbers (paper_strategy.md, April 24 2026):
  - MATH-500 @ 4096: hs_variance_detrend 72%, ThinKV 71%, ceiling 75%
  - AIME-2024 @ 8192: lag_kv 37%, h2o/hybrid_seg_hs 33%, ceiling 43%
  - lag_kv 2.8× faster than RaaS at AIME @ 8192 (441s vs 1239s wall-clock)
  - H2O collapse: 93/100 empty generations on MATH-500 at cache=1024

**Hard rules that apply to every stage.** Repeat these to the model in each prompt:

1. If you cannot verify a claim from the code or the data, SAY SO. Do not pattern-match
   to what you'd expect the code to do. Do not infer correctness from the surrounding
   prose. Verify or label UNVERIFIED.
2. The internal docs (`experiments/*.md`) are NOT a source of truth. They are a set of
   claims to be cross-checked against the code and the result files. Where docs and
   code disagree, code wins; where the code does what the docs say but does it
   incorrectly, both are wrong.
3. Cite specific file:line for every finding. Quote the relevant snippet. Vagueness is
   not acceptable.
4. "Looks reasonable" is not a verdict. Either the code does what's claimed, or it
   doesn't, or you don't have enough information to tell.
5. Do not soften findings. Do not hedge. If something is broken, say it's broken. If
   something is correct, say it's correct.

---

## Stage 1 — Implementation Verification

```
You are auditing the codebase at https://github.com/stevenkolawole/epiphany-kv-cache.

Your job is to verify, claim by claim, that the implementation matches what the
documentation says it does. Be thorough and neutral. Do not pre-judge whether the
project is good or bad — your output is evidence the next stages will use.

REQUIRED READING (in this order):
1. README.md
2. experiments/paper_strategy.md
3. experiments/research_overview.md
4. experiments/phase0b_ablation_results.md
5. experiments/signals_reference.md
6. experiments/progress.md
7. EXECUTION_GUIDE.md, MODEL_VARIANTS.md
8. src/eviction.py (every line)
9. scripts/collect_traces.py
10. scripts/label_importance.py
11. scripts/extract_phase0b_signals.py
12. scripts/signal_ablation.py
13. scripts/benchmark.py
14. scripts/analyze_phase1.py
15. scripts/inspect_traces.py
16. tests/

Then, for each subsection below, verify the claims listed and produce a
finding row in the format:

  CLAIM_ID | CLAIM_TEXT | FILE:LINES | VERDICT | EVIDENCE | NOTES

Verdict ∈ {VERIFIED, PARTIAL, UNVERIFIED, CONTRADICTED}.
- VERIFIED = the code does exactly what the claim says, no caveats found.
- PARTIAL  = the code does most of what the claim says, but with a meaningful
  divergence you can characterize precisely.
- UNVERIFIED = you cannot determine from the code alone whether the claim holds.
  Specify what artifact would resolve it (a log file, a result CSV, a re-run).
- CONTRADICTED = the code does something different from what the claim says.
  Quote the divergence.

============================================================
1.A — Signal computation (verify against signals_reference.md and phase0b_ablation_results.md)
============================================================

For each of the following signals as implemented in scripts/collect_traces.py and
scripts/extract_phase0b_signals.py, verify:

  Signal: kv_key_var, kv_key_norm, kv_val_var, cross_head_var,
          h2o_attn, attn_entropy, hs_l2_diff (and all hs_l2_diff_l{0..31}),
          hs_cos_dist, hs_norm, kv_key_var_preRoPE, kv_key_norm_preRoPE

For each signal:
- Find the exact code that computes it. Quote it.
- Confirm the formula matches signals_reference.md verbatim. If it diverges, name
  the divergence.
- Verify multi-head aggregation matches what the docs claim (mean across heads
  is the documented choice).
- Verify layer aggregation matches what the docs claim.
- For attn_entropy specifically: is it query-perspective or key-perspective?
  signals_reference.md §8 says it's query-perspective; ThinKV uses key-perspective.
  Confirm the implementation matches the documented (query-perspective) version.
- For h2o_attn: is it cumulative attention received by each token across all
  decode steps, or is it single-step attention? README and progress.md (March 25)
  say cumulative is the requirement. Verify.
- For hs_l2_diff_lN: confirm the L2 norm is taken over the hidden-dim axis only,
  not flattened across batch/seq.
- For pre-RoPE keys: verify the forward hook on `layer.self_attn.k_proj` captures
  the projection output BEFORE `apply_rotary_pos_emb` is called, not after.

============================================================
1.B — Counterfactual labelling pipeline (label_importance.py)
============================================================

The whole project rests on the importance labels. Verify:

- Window size = 32, stride = 16 (creates overlapping windows; each interior
  position is covered by 2 windows).
- Overlapping-window resolution: progress.md says it was changed from "overwrite
  semantics" (last window wins) to "OR semantics" (any flip → label=1). Verify the
  current code uses OR semantics. Quote the exact lines.
- Truncation→occlusion fix (March 30): the buggy version truncated the sequence
  at mask_start and regenerated, which made early tokens systematically labeled
  important. The fix uses `find_answer_start()` to locate the answer boundary,
  replaces the window with pad_id, feeds the FULL modified context up to
  answer_start, then regenerates the answer. Verify:
    a) `find_answer_start()` exists and works as described (token search →
       fallback to last `\boxed{` → fallback to total_len - 64).
    b) Every masked-inference call feeds the same context length per trace
       (only window content varies).
    c) No remaining call site uses the old truncation path.
- max_new_tokens for re-generation: progress.md / research_overview.md §1 limitation
  6 says it's 512. Verify. If it's a different value, flag the divergence.
- `answers_match()` normalisation: research_overview.md §1 says it handles
  \dfrac/\frac, \left(/(, \text{}, set reordering. Does NOT handle symbolic
  equivalence, approximate decimal, multi-line answers. Verify.
- Determinism: are the regeneration calls seeded? If different runs of the
  labelling pipeline produce different labels, the signal-vs-label correlations
  are unstable. Confirm seed handling.

============================================================
1.C — Signal ablation pipeline (signal_ablation.py)
============================================================

- Verify the Spearman ρ calculation: per-token aggregation vs per-trace? The
  phase0b_ablation_results.md "n_eff is the trace count, not n_pairs" caveat
  matters here. What does `signal_ablation.py` actually compute, and how does
  it report it?
- Verify rolling64 implementation: is it CAUSAL (uses only positions 0..t)? Or
  does it use a centered window that peeks at future tokens? An off-by-one here
  would invalidate every "rolling64" result.
- Verify EMA implementation: α=0.9 weighting, causal accumulation.
- Verify the cumsum exclusion: progress.md says cumsum was removed because it's
  monotone-positional. Confirm cumsum is NOT among the variants currently
  computed. (If it still is, the docs are stale.)
- Verify the "summary bug" fix (April 2): `max(abs(ρ))` rather than `max(ρ)`.
  Quote current code.
- Look for data leakage: is the "best layer" or "best signal" ever selected on
  the same data the final ρ is reported on? If the same dataset answers both
  "which layer is best" and "what is the ρ at the best layer," that's a free
  parameter selected on test data.

============================================================
1.D — Eviction methods (src/eviction.py)
============================================================

For each of the 11 eviction classes, verify:

- H2OEviction:
    - Is it cumulative attention (running sum across decode steps), not
      single-step?
    - Are attention sinks always preserved (configurable, but should default to
      keeping at least the first ~4)?
    - Is recency window preserved?
    - Eviction triggers when len(cache) > cache_size? Or some threshold like
      1.5 × cache_size?
    - Verify the device fix from April 24: `keep_mask.to(k.device)` is applied
      everywhere across all 12 occurrences. (If even one is missed, multi-GPU
      runs will silently fail.)
    - Verify the DynamicCache `_seen_tokens` desync fix in benchmark.py
      `_to_model_kv`: an empty `DynamicCache()` is constructed and then
      `cache.update(k, v, layer_idx)` is called per layer. Confirm.

- ThinKVEviction:
    - Verify the "budget bug fix" (April 13): `evict_past_key_values` tracks
      `remaining_budget` across segments and stops once cache_size is reached.
      Quote the exact lines.
    - Segment classification: KDE on key-perspective attention sparsity from 4
      layers, refreshed every 128 decode steps. Is the KDE actually computed?
      Or is it stubbed with a tertile split? (progress.md April 2 says
      "tertile split" — was this kept or replaced?)
    - Is the segmentation refresh actually every 128 steps, or every call?
    - Per-segment budgets (R/E/T): {64, 32, 8} per progress.md. Are these
      hardcoded, configurable, or derived from cache_size?
    - Verify there is NO fork to a custom CUDA kernel — it's pure Python /
      HuggingFace DynamicCache. If true, this matters for the speed claim
      (the speed comparison reflects engineering, not algorithm).

- RaaSEviction:
    - LRU timestamps: refreshed when token attention falls in top-50%? Verify.
    - Prefill tokens always preserved? Verify.

- HSVarianceEviction:
    - Score = `hs_l2_diff_l10_rolling64(t) − hs_l2_diff_l21_rolling64(t)`?
    - Online computation: at each decode step, two stored vectors per layer
      (~20KB for d=5120) and a rolling buffer of last 64 values. Quote the
      data structures used.
    - Is the rolling64 buffer causal? An off-by-one means the score for token t
      includes information from t+1.
    - On prefill: how is the buffer initialized for tokens 0..N where the
      rolling window isn't full? Default values may favor or disfavor early
      eviction.
    - FA2 compatibility: check that the forward call uses
      `output_hidden_states=True` but does NOT set `output_attentions=True`.
      The whole FA2 claim depends on this.

- DetrendendHSVarianceEviction:
    - z-score: `z(t) = (signal(t) − rolling_mean[t]) / (rolling_std[t] + ε)`.
      Confirm the rolling mean and rolling std are CAUSAL (window over past
      only).
    - Combined score: `z_l10 − z_l21`? Or some other combination?
    - paper_strategy.md notes: "hs_variance_detrend (72%) only beats raw
      hs_variance (71%) by 1 point at 4096. Worth re-running at lower budgets
      (256, 512) before claiming detrending is load-bearing." Has this been
      re-run? Where would the result CSV live?

- BandAdaptiveHSEviction:
    - All Band A layers (l7..l13) and Band B layers (l18..l25), z-scored,
      weighted (weight_a=1.29, weight_b=1.0). Verify weights match the Phase 0B
      ρ ratio claimed (math500_eager: l10=0.141, l21=0.109; ratio = 1.29).
    - The "weights from math500_eager" choice is itself a free parameter
      selected on the data the method is later evaluated on. Note this in
      EVIDENCE; do not call it a bug, just flag.

- AttentionHSProductEviction (eager-only):
    - Cumulative key-perspective attn + detrended Band A HS z-score.
    - Each component normalized to [0,1] separately, then summed (or weighted
      sum) — verify against research_overview.md.

- HybridSegmentHSEviction (eager-only):
    - Outer loop: ThinKV segment classification by key-perspective entropy.
    - Inner loop: detrended HS Band A−B z-score for within-segment ranking.
    - Decode-relative indexing: prefill tokens always preserved; only decode
      tokens ranked. Verify.

- KVValVariance / KVKeyVariance:
    - Variance of value/key vectors over head_dim, averaged across heads and
      layers, rolling64. Verify.

- LagKVKey / LagKV:
    - "Lag-relative normalization" — what is the exact formula? What is the lag
      window? signals_reference.md and research_overview.md cite LagKV as
      "lag-relative minmax normalization" — is that what's implemented, or is
      it a different lag-relative scheme? Quote it.

============================================================
1.E — Benchmark harness (scripts/benchmark.py + analyze_phase1.py)
============================================================

- Decoding determinism: are generations deterministic (greedy / temperature=0)
  or sampled? Single-seed sampled runs at n=30 (AIME) make 1–3pt differences
  uninterpretable.
- Cache budget definition: when method M says "cache_size=4096", does that mean
  4096 tokens *retained* AFTER eviction, or 4096 is the trigger threshold
  before eviction kicks in? Is the definition consistent across all 11
  methods? In particular for ThinKV, whose per-segment budgets sum to a value
  that may or may not equal cache_size — verify the cap.
- Sink/recency tokens: do they count against cache_size, or do they sit on top
  of it? If method A counts them in the budget and method B doesn't, the
  comparison is unfair.
- Prefill preservation: H2O, ThinKV, RaaS, the HS family — do all of them apply
  eviction to prefill tokens, or only to decode tokens? RaaS explicitly
  preserves prefill; ThinKV may or may not; H2O typically applies to all
  tokens. The HS family — what does it do? If they differ, that's a confound.
- Timing measurement:
    - CUDA sync before/after timing region?
    - Warmup runs?
    - Per-problem wall-clock or aggregate?
    - Same kernel path (eager vs flash) at the same cache_size?
  paper_strategy.md says "lag_kv (FA2) is 2.8× faster than raas (eager) at AIME
  @ 8192 (441s vs 1239s)". The FA2 vs eager comparison is *known* to be a path
  difference — verify the wall-clock methodology is sound, since that's the
  load-bearing claim.
- Memory measurement:
    - Per-example reset of `torch.cuda.reset_peak_memory_stats()` (March 25
      TODO 4 says this was the fix).
    - Verify it actually happens per-example, not per-batch or per-run.
- H2O collapse claim (93/100 empty generations at cache=1024): verify the
  detection logic. "Empty generation" — is this `len(generated) == 0`, or
  `generated[0] == eos_token_id`, or "no `\boxed{}` in output"? The
  interpretation matters.
- The "none" baseline: progress.md April 24 says it now runs once and is copied
  across cache_size slots. Verify the copy is done correctly (same accuracy,
  same wall-time, but cache_size label differs).

============================================================
1.F — Tests
============================================================

- Inventory tests/. What is actually tested? List each test and what it
  verifies.
- What invariants are tested for the eviction methods? E.g., is it tested that
  after eviction, len(cache) ≤ cache_size? That position IDs are still
  contiguous? That sink tokens are preserved? That the cache device matches the
  model device?
- What is conspicuously NOT tested? Specifically:
    - Determinism of label generation under fixed seed
    - Causality of rolling64 (no peeking at future tokens)
    - DynamicCache _seen_tokens consistency after eviction
    - Multi-GPU device-map correctness
    - The specific bugs flagged in the April 24 progress entry — are there
      regression tests for them now?

============================================================
OUTPUT FORMAT
============================================================

Three artifacts:

A. DEFECT TABLE: every finding from sections 1.A–1.F as a single table, sorted
   by (verdict, severity), severity ∈ {blocker, major, minor, informational}.
   blocker = invalidates a headline result. major = could change a headline
   number by >2pt or change a directional claim. minor = correctness/hygiene
   issue with no result impact. informational = no defect, just a fact worth
   noting.

B. UNVERIFIABLE LIST: every claim where verdict = UNVERIFIED, with the specific
   artifact (log, CSV, re-run, file) that would resolve it.

C. METHODOLOGICAL CONCERNS: things that are "working as documented" but where
   the methodology has a vulnerability. Examples to scan for: data leakage in
   parameter selection, decoding seed control, single-seed variance, cache-budget
   definition mismatches, kernel-path mismatches in timing, label generation
   determinism. Do not call these bugs; call them concerns.

HARD RULES (repeat):
- No verdict without quoted code or quoted result file.
- No strategy, no recommendations, no "this could be reframed as." Just the
  audit.
- If a claim is fine, mark it VERIFIED and move on. Don't pad findings.
```

---

## Stage 2 — Comparison Setup & Threat Model

```
You previously produced an implementation audit of epiphany-kv-cache. Now
do a second pass on the comparison setup and the literature positioning.
You may treat the Stage 1 defect table as established context — do not re-verify
those, but DO incorporate them when making judgments here.

CONTEXT — adjacent literature this work positions against:
  Direct baselines (implemented):
    - H2O (Zhang et al., 2023): cumulative attention heavy hitters
    - ThinKV (He et al., ICLR 2026 oral): segment R/E/T classification, hybrid
      quantization+eviction
    - RaaS (Hu et al., Feb 2025): LRU timestamp, milestone/phoenix taxonomy

  Adjacent / cited but not implemented:
    - StreamingLLM (Xiao et al., 2023): sinks + recent
    - SnapKV (Li et al., 2024): prefill-time observation-window
    - PyramidKV (Cai et al., 2024): layer-wise pyramid budget
    - ChunkKV (NeurIPS 2025): chunk-level eviction
    - LongFlow (2025): ||attn × val||₁; same model class (DeepSeek-R1)
    - FreeKV (ICLR 2026): retrieval, not eviction
    - SideQuest (2026): model-driven cache management for tool responses
    - EpiCache (2025): episodic compression for long conversations
    - KIVI / KVQuant / MiniKV: quantization
    - LagKV: lag-relative normalization (cited as inspiration for LagKVKey/LagKV)
    - AhaKV: analytical detrending (cited as inspiration for DetrendendHS)
    - CAOTE: closed-form eviction error

  Open problem: search arxiv/OpenReview for any 2025-2026 reasoning-CoT KV
  compression work post-RaaS / post-ThinKV. The audit's job here includes
  finding work the project may have missed.

============================================================
2.A — Baseline fidelity
============================================================

For H2O, ThinKV, RaaS as implemented in src/eviction.py, compare against the
original papers (you may need to read or skim them):

For each baseline:
- Hyperparameters: are they set to the values the original paper reports for
  the same model/dataset/budget? List any divergences. Specifically:
    - H2O: heavy-hitter ratio, sink count, recency window
    - ThinKV: segment length (128), refresh interval (128 steps), R/E/T budget
      values (paper has specific values; the code uses {64, 32, 8} per
      progress.md April 2)
    - RaaS: top-percentile threshold for LRU refresh (50%? configurable?),
      prefill preservation (all? first-K?)

- Algorithmic faithfulness:
    - H2O: is "cumulative attention" computed exactly as the original paper
      defines it? (Sum of received attention from all subsequent decode steps,
      not single-step.)
    - ThinKV: KDE on key-perspective sparsity vs. tertile split — which is
      implemented? Does it match the paper? If a simplification was made for
      implementation tractability, that is itself a finding (the comparison
      then tests "ThinKV with KDE replaced by tertile" not "ThinKV").
    - RaaS: milestone/phoenix taxonomy is paper-level conceptual; the algorithm
      is LRU+prefill. Is the implementation faithful to Algorithm 1 of the
      paper?

- Engineering parity:
    - All three baselines and all HS family methods run on the same kernel path
      at the same cache_size? (Eager vs FA2.)
    - Same DynamicCache implementation across methods? (Some methods might use
      a faster code path.)
    - The lazy-deletion / CT-kernel speed advantage of ThinKV is NOT
      implemented here (per progress.md April 13: cannot implement in
      HuggingFace DynamicCache). This means the ThinKV speed numbers in this
      project are a lower bound on ThinKV's true speed. Note this — it does not
      affect the algorithmic comparison, but it affects how the speed claim
      against ThinKV should be framed.

============================================================
2.B — Comparison-framework validity
============================================================

Two specific framework concerns to address:

(1) The "h2o_attn is the weakest signal — 3–12× weaker than HS signals" claim.
    h2o_attn is being scored within the Spearman-ρ-against-occlusion-labels
    framework alongside HS signals. But H2O the eviction policy is not designed
    to produce per-token importance scores that predict occlusion impact —
    it's a cumulative-attention eviction rule. Is comparing them in this
    framework apples-to-apples?

    Two sub-questions:
      (a) For the SIGNAL comparison (Phase 0B Spearman ρ), is h2o_attn well-
          defined as a per-token signal? The signals_reference.md description
          ("cumulative attention received by each token") IS a per-token
          quantity, so the comparison is well-defined for Phase 0B. Confirm.
      (b) For the END-TO-END comparison (Phase 1 accuracy curves), the H2O
          eviction policy is being compared, not the h2o_attn signal in
          isolation. So the Phase 1 comparison is policy-vs-policy and is fair.
          Confirm.

    Conclusion target: separate the two claims. The ρ claim is "h2o_attn as a
    per-token importance score is weak"; the Phase 1 claim is "the H2O
    eviction policy collapses on reasoning workloads." Both can be true; they
    are different claims. Verify both are framed correctly in the docs (or
    flag where the framing slips between them).

(2) The "rolling64 outperforms raw and EMA by 30–57% universally" claim.
    The phase0b_ablation_results.md §2 table shows this for kv_key_var across
    4 datasets. The README says "universal." Is the universal claim supported
    by the data? Specifically:
      - Across the other 5 datasets (gsm8k, AIME 2025/2026 each in two
        configs)?
      - Across the other signal families (kv_val_var, hs_l2_diff_lN for various
        N, attn_entropy, h2o_attn)?
    Find the actual evidence in the result CSVs or tables. If the rolling64
    advantage is documented for one signal family on 4 datasets and called
    "universal," that's a framing concern.

============================================================
2.C — Threat model
============================================================

For each headline claim in paper_strategy.md, write the strongest plausible
counter-explanation a hostile reviewer would give. List, ranked by force:

  C1. MATH-500 @ 4096 hs_variance_detrend = 72% vs ThinKV = 71% (1pt over n=100).
  C2. AIME-2024 @ 8192 lag_kv = 37% vs h2o = 33% (3pt over n=30).
  C3. lag_kv 2.8× faster than RaaS (FA2 vs eager).
  C4. H2O collapse on 93/100 MATH-500 problems at cache=1024.
  C5. HS Band A (l7–l13) consistently positive ρ; Band B (l18–l25)
      consistently negative.
  C6. h2o_attn is the weakest importance signal tested.
  C7. Rolling64 outperforms raw and EMA universally by 30–57%.
  C8. FA2 compatibility for HS methods.

For each, give:
  - The strongest counter-explanation.
  - The minimum experiment that would refute the counter-explanation.
  - An assessment of whether that experiment exists in the repo's results
    already, or would need to be run.

Examples of strong counter-explanations to consider for each claim type:
  - For accuracy claims: single-seed variance, cherry-picked budget, ceiling
    proximity, baseline-implementation weakness, label leakage, data leakage
    in hyperparameter selection.
  - For speed claims: kernel path mismatch, batch size mismatch, model not
    fully on GPU, eviction overhead amortized differently.
  - For collapse claims: implementation bug masquerading as real behavior.
  - For Phase 0B band claims: data used to select the band is the same data
    used to evaluate the band's correlation.
  - For "universal" claims: the universal claim covers more than the data
    table actually shows.

============================================================
2.D — Distinctness from prior work
============================================================

How is the proposed method mechanically distinct from each of:
  - H2O / ThinKV / RaaS (the implemented baselines)
  - LongFlow (||attn × val||₁; same model class, attention-based)
  - EpiCache (episodic compression — note name collision with "epiphany")
  - LagKV (the LagKV variant of the project's own method has its own LagKV
    inspiration; clarify the relationship)
  - AhaKV (the detrending variant of the project's own method has AhaKV
    inspiration; clarify)

Specifically:
- For each pair (proposed method, prior method), list the mechanical
  difference. Not "we use HS instead of attention" — that's the framing. The
  mechanical question is: at a given decode step, what specific computation
  does our method do that the prior method does not, and vice versa?
- For LagKV vs LagKVKey/LagKV-the-method: is the project's LagKV literally
  applying LagKV's normalization to KV variance? If so, the contribution is
  applying LagKV's idea to a slightly different signal — useful, but not a new
  method. Frame this honestly.
- For AhaKV vs DetrendendHS: same question. Is detrending a known technique
  applied to a new signal, or is it novel?

============================================================
2.E — Coverage gaps
============================================================

What baselines are MISSING that a NeurIPS reviewer would expect to see for
this exact setting (long-CoT-trace KV eviction on DeepSeek-R1-class models)?
Search arxiv.org and OpenReview for 2025-2026 work on:
  - reasoning-trace KV compression
  - long-CoT inference efficiency
  - DeepSeek-R1 / o1-class inference

For each missing baseline, give: what it is, why a reviewer would ask for it,
how hard it would be to add (a separate eval against published numbers vs. a
full re-implementation).

============================================================
OUTPUT
============================================================

A. BASELINE FIDELITY TABLE: (baseline, status, divergences, severity).
B. THREAT MODEL: ranked list per 2.C, with refutation experiments.
C. NOVELTY ASSESSMENT: 2-3 paragraphs per 2.D.
D. COVERAGE GAPS: list per 2.E.

HARD RULES:
- No recommendations yet. Just diagnosis.
- If a baseline diverges from the original paper and you don't know whether the
  divergence is intentional or a bug, mark UNVERIFIED — do not pick a side.
- If you are unsure whether a claim is "novel" vs "applied," err on the side of
  describing the mechanical operation precisely and let the next stage judge.
```

---

## Stage 3 — Results Sanity Pass

```
You have produced (1) an implementation audit, (2) a baseline fidelity report,
threat model, novelty assessment, and coverage gaps.

Now read the actual experimental result files. Locations to check:

- Result CSVs from Phase 0B signal ablation (results/ directory)
- Result JSONs from Phase 1 benchmarks: paper_strategy.md says they live at
  /data/user_data/skolawol/kvcache/results/phase1/ — these may not be in the
  public repo. If they aren't, mark every Phase 1 numeric claim UNVERIFIED-AT-
  RESULT-LEVEL and base your analysis only on the in-repo evidence
  (analyze_phase1.py output, reports/phase1_plots/ PDFs if present).
- reports/phase1_plots/ for the accuracy curves (PDF format).
- Any *.log files in slurm_logs/ that survive in the repo.

For each headline number from Stage 2 section C (C1–C8), classify it as:

  ROBUST: implementation verified in Stage 1, baseline verified in Stage 2,
          no identified threat plausibly explains it away, and the result file
          confirms the number.
  SUSPICIOUS: at least one Stage 1 finding, Stage 2 baseline issue, or Stage 2C
          threat could plausibly explain part or all of the claim. Specify
          which.
  ARTIFACT: there is concrete evidence (a Stage 1 bug, a Stage 2 strawman, an
          evaluation issue, or a result-file-vs-paper-claim mismatch) that the
          number does not measure what it claims to.
  UNVERIFIABLE: the underlying result file is not accessible from the repo;
          you cannot confirm or deny the headline number.

For each SUSPICIOUS or ARTIFACT result, specify the smallest experiment that
would resolve which it is — be specific about: which method(s), on which
dataset(s), at which budget(s), with what seed control.

============================================================
3.A — Internal consistency check
============================================================

The internal docs make many claims. Check whether they are internally consistent.
Specifically, look for:

- paper_strategy.md says "hs_variance_detrend (72%) only beats raw hs_variance
  (71%) by 1 point at 4096. Detrending is a marginal helper at this budget,
  not a key fix." The README and the headline claim, however, lead with
  hs_variance_detrend. Are the README headline and the paper_strategy caveat
  consistent? If a reviewer reads only the README, do they get a misleading
  picture?

- phase0b_ablation_results.md §10 caveat 4 says: "Aggregate Spearman ρ
  overestimates real-world per-token eviction quality. ... The combined score
  l10−l21 is HIGHER for early-position (DROP) tokens than late-position (KEEP)
  tokens within a trace." This is a major within-trace ranking failure. Is the
  Phase 1 success of `hs_variance_detrend` (72%) a real refutation of this
  failure, or does the temporal trend still poison per-token rankings even
  after z-scoring? The detrending was supposed to fix this. Verify whether
  Phase 1 results actually demonstrate the fix.

- The "kv_val_var consistently non-negative" claim was retracted (paper_strategy
  April 13 correction). Are there any other claims with similar fragility? Look
  for "consistently," "universally," "across all datasets" — apply the same
  scrutiny to each.

- The H2O-as-importance-signal weakness claim and the H2O-policy-collapse claim
  are sometimes conflated in the docs. Check whether this conflation matters
  for any specific load-bearing argument.

- The README says "h2o_attn is 3–12× weaker than HS signals." The
  phase0b_ablation_results.md table actually shows ratios like:
    - math500_eager: best HS = +0.144 (l11), h2o_attn = +0.050; ratio ~2.9×
    - aime2024_eager: best HS = −0.227 (l21), h2o_attn = −0.011; ratio ~21×
    - gsm8k_eager: best HS = −0.351 (l15), h2o_attn = −0.086; ratio ~4×
  Where does "3–12×" come from? Is it a specific subset of datasets/layers? Is
  it accurate?

============================================================
3.B — Statistical reality check
============================================================

For each headline, what is the actual statistical power?

- MATH-500: n=100. A 1pt difference (72% vs 71%) is ~1 problem of 100. What's
  the binomial 95% CI on a single proportion at n=100? What's the CI on a
  paired difference? Is the 1pt margin meaningful?
- AIME-2024: n=30. A 3pt difference (37% vs 33%) is ~1 problem of 30. Is this
  within sampling noise? phase0b_ablation_results.md is candid about AIME's
  small n_eff for ρ; the same caveat applies to accuracy.
- ρ values from Phase 0B: phase0b_ablation_results.md table (math500: ρ=−0.254
  at l23; max ρ=0.380 for kv_key_var). With n_eff=75 (math500), ρ=0.25 has 95%
  CI roughly (0.02, 0.46). What does this say about the strength of the Phase
  0B claims?

============================================================
3.C — Keep / Grey / Kill list
============================================================

Based on Stages 1, 2, and 3.A/3.B, classify every headline framing claim into:

KEEP — survives all scrutiny, would survive reviewer pressure, can be claimed
       as-is in the paper. Be specific about exactly what is being claimed
       (the precise wording matters; e.g., "h2o_attn as a per-token importance
       signal is weaker than HS Band A signals on competition math" is keepable
       even if "H2O is 3–12× weaker than HS" is not).

GREY — could go either way. Specify the experiment that would move it to
       KEEP or KILL. The 3-week extension plan in Stage 4 will pull from this
       list.

KILL — should be dropped from the submission entirely because it cannot be
       defended even with reasonable additional work, OR because the cost of
       defending it (more experiments, reframing) exceeds the value it adds
       to the paper.

For each KILL, specify what part of the project's narrative depends on it. If
the project's main framing is on the kill list, that's a pivot signal — say
so plainly. If the keep list is short, the keep list is short. Don't pad.

============================================================
OUTPUT
============================================================

A. RESULT CLASSIFICATION TABLE: rows = headline numbers (C1–C8), cols =
   {robust/suspicious/artifact/unverifiable, evidence, refuting experiment}.
B. INTERNAL CONSISTENCY FINDINGS: from 3.A.
C. STATISTICAL POWER NOTES: from 3.B.
D. KEEP LIST: with precise wording for each kept claim.
E. GREY LIST: with the specific experiment for each.
F. KILL LIST: with what each kill removes from the narrative.

HARD RULES:
- A claim's vibe-strength does not matter; only what survives the audit, the
  threat model, and the result-level check matters.
- If KEEP is short, KEEP is short. Do not invent claims to keep.
- If the project's main framing is in KILL, name the pivot explicitly.
```

---

## Stage 4 — Two-Plan Submission Strategy

```
You have produced: (1) implementation audit, (2) baseline & threat report,
(3) keep/grey/kill lists. Now build two NeurIPS submission plans.

Reference: paper_strategy.md already sketches a Phase 2 plan (AIME pooling
2024+2025+2026 to n=90, GSM8K addition was already debated and excluded,
LagKV-style normalization, kv_seg_hs FA2 analog of HybridSegmentHS for tight
budgets, vLLM/TGI engineering validation as Phase 2C). Your Plan B should
ENGAGE WITH that scoped plan — endorsing parts, pushing back on others, and
re-prioritizing based on what survived Stage 3 — rather than inventing new
work the team hasn't considered.

============================================================
PLAN A — SUBMIT AS-IS
============================================================

Constraints: only the KEEP list from Stage 3 may be used as a contribution
claim. No new experiments. Writing and minor fixes only.

Produce:
  - Title (specific, not generic).
  - One-sentence pitch.
  - Contribution claim: be explicit about whether this is a method, an
    empirical study, an architecture-compatibility result, or a combination.
  - Section-by-section reorganization vs. paper_strategy.md's current sketch:
    what gets cut, what gets foregrounded, what gets moved to appendix.
  - Realistic acceptance probability with reasoning. Be specific:
    "borderline reject expected; rebuttal-dependent" is more useful than
    "decent chance." Note what the typical NeurIPS reviewer in this subfield
    will fixate on.
  - The SINGLE strongest weakness every reviewer will flag, and the response.
  - The two strongest reviews this paper would receive (one positive, one
    negative).

============================================================
PLAN B — 3-WEEK EXTENSION
============================================================

Budget: 3 weeks (≤ 90 engineer-hours). Phase 1 is already done; the team is
working on Phase 2.

Given the GREY list from Stage 3 and the Phase 2 plan in paper_strategy.md,
produce a ranked list of additions, ranked by ROI = (impact-on-acceptance /
engineer-hours).

For each addition, specify:
  - The experiment: methods, datasets, budgets, baselines, seeds.
  - The threat from Stage 2C it neutralizes (or grey claim from Stage 3 it
    moves to keep).
  - Engineer-hour estimate.
  - Expected outcome and the magnitude that would matter (e.g., "if
    detrending lifts hs_variance from 28% to >36% at cache=1024, the tight-
    budget regime claim becomes defensible").

Be ruthless about cuts. If experiment X requires also running fairness check Y,
and Y blows the budget, cut X. If the existing Phase 2 plan items have low ROI
relative to scrutiny in the audit, recommend dropping them and re-allocating.

Distinguish two kinds of additions:
  - Strengthen existing claim (e.g., AIME pooling to n=90 firms up the lag_kv
    headline).
  - Pivot the claim to something more defensible (e.g., if the within-trace
    temporal trend cannot be cleanly resolved, reframe the contribution as
    "characterization of when HS signals beat attention signals" rather than
    "method for KV eviction").

============================================================
RECOMMENDATION
============================================================

Pick: A, B, or NEITHER (workshop / ICLR with full extension).

Justify in 5-8 sentences referencing specific findings from prior stages.

If recommending NEITHER, specify the realistic resubmission path:
  - Workshop name(s) with deadlines that fit.
  - ICLR 2027 timeline and what would need to be true by then.
  - Whether to make the data/results public now (signal generosity, gather
    feedback) or hold for resubmission.

============================================================
HARD RULES
============================================================

- Do not recommend Plan B if Plan A's keep list cannot support a coherent
  paper even with extensions. If KEEP is too thin, the recommendation is
  NEITHER.
- "Add more ablations" is not a plan. Specify which ablations, on what
  hypothesis, with what magnitude would matter.
- If the right answer is "this isn't a NeurIPS paper, it's an ICLR or
  workshop paper," say so plainly with reasoning.
- The advisor's framing ("FA2 compatibility is an unexpected engineering
  win — make it impossible for reviewers to miss") is one input among many.
  Do not let it override what the audit found. If FA2 compatibility didn't
  survive Stage 1 (e.g., a method silently disables FA2 somewhere), say so.
- Single strongest weakness EVERY paper has one. Name it explicitly.
```

---

## Notes on running this

- **Stage cadence.** Each stage takes a strong code model 30–90 min of focused
  work given full repo access. Don't try to merge stages — the staging is what
  prevents the model from rationalizing toward "this should be a NeurIPS paper."

- **Hand-off.** After Stage 1, eyeball the defect table yourself before
  proceeding. If the model missed something obvious to you, that's a calibration
  signal — re-prompt with "you missed X, redo with that lens." Stages 2–4 are
  only as good as Stage 1.

- **Phase 1 result files.** paper_strategy.md says they live at
  `/data/user_data/skolawol/kvcache/results/phase1/` — outside the repo. If you
  want Stage 3 to verify the Phase 1 numbers at the result level rather than
  trust paper_strategy.md, either copy those JSONs into the repo before running
  Stage 3, or commit a sanitized summary CSV. Otherwise Stage 3 will mark Phase
  1 claims UNVERIFIABLE-AT-RESULT-LEVEL and lean on analyze_phase1.py output.

- **Optional calibration prompt.** Before Stage 1, you can feed the model a
  recent harsh OpenReview thread on a KV cache paper (or a ThinKV / RaaS
  rebuttal) so its sense of "what scrutinizing looks like" is anchored to
  the subfield's actual review style, not generic NeurIPS reviewing. Optional
  but tightens output quality.

- **A note on bias.** I was sliding into "judge what's robust" mode in earlier
  drafts. These prompts are designed so the audit *forms* its own verdict from
  evidence, not from my pre-judgment. The "Calibration constants" section at
  the top is factual setup, not opinion; if you spot an opinion smuggled in,
  flag it and I'll revise.