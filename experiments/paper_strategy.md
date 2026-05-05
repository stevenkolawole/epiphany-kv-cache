# Paper Strategy — NeurIPS Submission

**Venue:** NeurIPS (9-page limit, ~4 figures typical, appendix unlimited)
**Status:** Pre-writing; strategy scratchpad. Update as Phase 1+ results arrive.

## Title Candidates

Decision deferred until Phase 1 results confirm end-to-end eviction quality.

**Leading candidate:**
> *The Epiphany Hypothesis: Hidden-State Variance Predicts Token Importance in Long Reasoning Traces*

Precedent: "The Lottery Ticket Hypothesis" (Frankle & Carlin, ICLR 2019) used "hypothesis" to name a concept that the paper then validates — not as an admission of uncertainty. Same logic applies. FA2 engineering does not need to be in the title; it carries the abstract, intro, and conclusion.

Other candidates considered and set aside:
- *Epiphany-Aware KV Cache Eviction: Why Hidden States Outperform Attention* — project brand but long and "epiphany" needs context
- *Beyond Attention: Hidden-State Variance as a FlashAttention-Compatible Token Importance Signal* — descriptive but no hook
- *Hidden States Know Best: FA2-Compatible KV Cache Eviction for Reasoning Models* — too informal

Decision: finalize after Phase 1 results confirm end-to-end eviction quality.

---

## Core Claim

Representational change signals derived from a model's internal state — hidden-state
variance at mid-layers and KV-key variance — are stronger and more deployment-friendly
predictors of token importance than attention weight, enabling KV cache eviction that is
(a) empirically well-calibrated and (b) architecturally compatible with FlashAttention 2
out of the box — with no attention matrix materialization required.

The claim has three legs that must all stand:
- **HS signal leg:** HS variance at l10/l13 (rolling64) outperforms attention-based
  signals in Spearman ρ on held-out data; Band A (l7–l13) is consistently positive
  across datasets, Band B (l18–l25) consistently negative.
- **KV signal leg:** KV-key variance (especially pre-RoPE) supports the core hypothesis
  that representational change predicts importance. Initial results show more
  dataset-dependency than HS signals — this is itself a finding, not a failure.
  HS and KV signals together frame a richer picture of where importance lives.
- **Engineering leg:** Both HS and KV signals are available within a standard FA2
  forward pass (`output_hidden_states=True`; KV keys accessible via hooks or
  `past_key_values`). No attention matrix materialization needed. Attention-based
  eviction requires either a separate eager-mode pass or custom FA2 kernel modifications.
  Online HS score computation is incremental — two stored vectors (~20KB for a 5120-dim
  model) plus the current step's hidden states, which are already computed. No refresh
  cycle needed (unlike ThinKV's 128-step refresh) because hs_l2_diff is a fixed local
  property of each token, not context-dependent. Precise claim: "no extra forward passes;
  negligible additional memory" — not "zero overhead" (KV signals are strictly cheaper
  since they read already-cached tensors with no extra storage).

**Positioning note:** The method is not "hidden states only." It is a family of
representational-change signals. HS signals emerge as the more dataset-stable variant;
KV signals are more context-sensitive. Both are contributions — one for deployment,
one for understanding signal behavior across problem difficulty.

---

## Advisor Framing (April 2026 meeting)

> "The FA2 compatibility is an *unexpected* engineering win — if the signal stands after
> full ablation, this is extremely strong. Make it impossible for reviewers to miss."

This should not be a footnote or a "nice property" paragraph. It needs its own figure
and should appear in the abstract, intro, and conclusion. The framing is:
- Other methods sacrifice architectural compatibility for signal quality.
- We don't. We get better signal *and* full FA2 compatibility.
- At 128k+ context lengths (now routine), attention-based scoring is not just inefficient
  — it's architecturally infeasible in production stacks. We are the only approach that
  doesn't require stepping outside FA2.

---

## Figure Plan

Figures are the most reviewer-facing part of the paper after the abstract. Each figure
must prove something, not just show something. Planned figures:

### Figure 1 — Engineering compatibility (the "unexpected win" figure)
Two panels:

**Panel A — Memory scaling:**
- X: sequence length (4k → 128k, log scale)
- Y: GB of additional memory required for importance signal extraction
- Lines: attention-based (O(n² × h), quadratic) vs HS-based (O(n × d), linear)
- Annotate with actual model numbers (Qwen2.5-32B, 8 KV heads)
- Goal: make the quadratic wall viscerally clear; label where 32k, 64k, 128k fall

**Panel B — Inference pipeline (swimlane diagram):**
- Attention-based: FA2 pass → *extra eager pass for attn matrix* → score → evict
- Ours: FA2 pass (HS collected inline) → score → evict
- Caption must note: the extra pass roughly doubles eviction-decision cost

### Figure 2 — Layer anatomy
- X: layer index 0–31
- Y: Spearman ρ with importance labels
- One line per dataset (math500, math500_eager, aime2024, aime2024_eager + 2025/2026 once collected)
- Shade Band A (l7–l13, consistently positive) and Band B (l18–l25, consistently negative)
- This is the core empirical finding; must be clean and readable
- Confidence intervals essential — show AIME uncertainty explicitly (wide bands)

### Figure 3 — Temporal smoothing comparison
- Bar chart: raw vs ema09 vs rolling64 for top 3 signals on math500
- +30–57% improvement from rolling64 is a methodological contribution
- Framing: importance is a sustained contextual property, not a pointwise signal

### Figure 4 — Phase 1: Eviction quality vs memory budget (TBD — post Phase 1)
- X: KV cache budget (% of full cache retained)
- Y: task accuracy (math500 pass@1)
- Lines: random eviction, attention-based baseline (H2O/ThinKV), ours
- This is the result that closes the paper; everything else is setup for this

---

## Section-by-Section Notes

### Abstract (≤200 words, 4 sentences approximately)
Must contain:
1. Problem: KV cache in long reasoning traces; existing attention-based eviction breaks FA2
2. Method: hidden-state variance at specific mid-layers as importance signal
3. Result: outperforms attention-based signals; FA2-compatible with no extra forward pass
4. Implication: slots into production inference stacks without modification

Do NOT bury the engineering contribution. It belongs in sentence 1 or 2, not sentence 4.

### Introduction
Must establish before the reader hits the method:
1. Long-context reasoning models (DeepSeek-R1 class) produce very long traces; KV cache
   is the memory bottleneck
2. Existing eviction methods assume attention weight = importance — empirically weak and
   architecturally costly (forces out of FA2)
3. Our observation: hidden-state variance at mid-layers predicts importance better
4. Engineering payoff: this choice is not just accurate, it's free in FA2 inference stacks

Avoid spending too much introduction real estate on background. Reviewers know what KV
cache is. Get to the tension (attention = importance assumption is wrong AND costly) fast.

### Method
Keep tight. The key choices to justify:
- Why hidden-state variance (not norm, not final-layer activations)
- Why mid-layers specifically (Band A)
- Why rolling64 smoothing (sustained contextual property argument)
- Why the combined score l10_rolling64 − l21_rolling64 (Band A − Band B cancellation)
- Why this is FA2-compatible (no attention matrix needed, HS available via standard API)

### Experiments / Results
Structure around the two legs:
- Signal quality: Spearman ρ ablation across datasets (Figure 2 + 3)
- Engineering: Figure 1 (memory + pipeline)
- End-to-end: Figure 4 (eviction quality vs budget) — the proof it actually works

**What goes to appendix:**
- AIME results until n_eff ≥ 30 labelled traces per year (current CIs too wide)
- Full 32-layer ρ table (Figure 2 is the figure; table goes to appendix)
- preRoPE null result (important for rigor, not for narrative)
- Cross-validation methodology details

### Conclusion
Leave the reader believing:
1. HS variance is a better importance signal than attention weight — empirically
2. The FA2 compatibility is not a coincidence; it follows from not needing the attention
   matrix, which is the same design choice that makes FA2 fast
3. This method is ready to drop into production inference stacks today

---

## Related Work Notes

**Connections to flag (strengthen our framing):**

- **EAGLE / EAGLE-2 (speculative decoding):** EAGLE's draft model operates on hidden
  states from the base model, not on token embeddings, because HS carry richer
  information about what comes next. This independently validates the intuition that HS
  are information-rich beyond attention weights. EAGLE uses HS for forward prediction;
  we use HS variance for importance assessment — complementary applications of the same
  core observation. Also: EAGLE is FA2-compatible for the same reason we are. Worth
  noting as convergent evidence in related work. Cite in one clause, not a paragraph.

- **ROME / MEMIT (mechanistic interpretability):** Already cited in signals_reference.md
  as the grounding for Band A (l7–l13). ROME/MEMIT identifies mid-network layers as the
  site of factual retrieval and feature routing in transformer LLMs. Our Band A finding
  is consistent with this: the layers that mediate factual/semantic processing are also
  the layers where representational change best predicts token importance. This is a
  stronger citation than generic probing literature for this specific claim — it connects
  to a well-known mechanistic result rather than a statistical correlation.

- **H2O, ThinKV, SnapKV, RaaS:** attention-weight-based baselines. Frame as the class
  of methods we outperform and replace architecturally.

- **StreamingLLM:** attention sink phenomenon — explains why raw attention is a poor
  importance signal (sinks absorb mass regardless of content). Motivates our departure
  from attention-based scoring.

- **FlashAttention / FlashAttention-2:** the architectural context that makes our
  engineering contribution non-trivial.

---

## Presentation Asymmetry

Results to foreground in main paper:
- math500 / math500_eager (large n, statistically solid)
- The engineering figure (Figure 1) — independent of ablation completeness
- Phase 1 eviction quality (once available)

Results to relegate to appendix until data matures:
- AIME 2024 / 2025 / 2026 individual breakdowns (n_eff too small)
- kv_key_var results (inconsistent across datasets; interesting but not a main claim)
- attn_entropy (implementation mismatch with ThinKV paper; needs clarification)

---

## Phase 0B Complete — Updated Findings (April 11, 2026)

### GSM8K anatomy (new finding)

With 355 correct traces (n_eff=352), GSM8K is a high-confidence dataset. Its layer anatomy
differs sharply from competition math:
- Band A shifts to **l0–l7** (early layers) for grade-school math
- Band B extends to **l10–l30** (almost the entire mid-to-late network)
- l31 (last layer hs_l2_diff) is **+0.231** — strongly positive unlike in math500
- attn_entropy **flips sign**: −0.313 in gsm8k vs +0.176 in math500_eager
- kv_key_var **flips sign**: −0.261 vs +0.380 — KV sign flip is now CONFIRMED, not noise

**Paper decision**: This is reportable as a finding — "the Epiphany layer is difficulty-
dependent." For competition math (Phase 1 target), l7–l13 is the predictive band.
For grade-school math, l0–l7 is predictive. This enriches the story: the same HS
mechanism operates at different network depths depending on task complexity.

**What does NOT change for Phase 1**: The target signal (l10_rolling64 − l21_rolling64)
is calibrated for competition math and is the correct choice for Phase 1 benchmarks.

### KV signal status (updated)

KV sign flip is now confirmed across two high-n datasets (math500 n_eff=75, gsm8k n_eff=352).
This is a definitive result: kv_key_var cannot be used as a fixed-direction signal without
difficulty-regime detection. It should remain as an online-fallback option for math500-class
problems only, with an explicit caveat about domain-dependence.

### Revised presentation asymmetry

Main paper (math500/math500_eager + Phase 1 competition math accuracy curves):
- GSM8K ablation results go to appendix as supplemental (different anatomy, different scope)
- AIME 2025/2026 go to appendix (small n_eff, wide CIs)
- GSM8K anatomy plot could be Figure 2 panel B (showing difficulty-dependence) if reviewer
  appetite for it; otherwise appendix

## Open Questions (revisit as results come in)

- Does the Phase 1 combined score (l10_rolling64 − l21_rolling64) hold up on actual
  eviction, or does Band B subtraction hurt in practice?
- Can we quantify the latency cost of attention-based scoring vs ours on real hardware?
  (Would make Figure 1 Panel B quantitative rather than qualitative.)
- Is there a mechanistic explanation for why Band A shifts from l7–l13 (hard math) to
  l0–l7 (easy math)? Interpretability work (logit lens, probing classifiers) could address
  this. Not required for NeurIPS but would strengthen future work section.
- Why does l31 (last layer hs_l2_diff) flip from near-zero (math500) to strongly positive
  (gsm8k, aime2026_eager)? Last-layer behavior is the least understood.

---

---

## Phase 0 Complete — Audit and Scope Finalization (April 13, 2026)

**Phase 0 / Phase 0B is COMPLETE.** All signal ablations done; codebase audit done; Phase 1 is unblocked.

### Primary experimental scope for Phase 1
- **Primary:** math500, AIME2024 (high n_eff; matches ThinKV/RaaS paper benchmarks exactly)
- **Secondary:** AIME2025, AIME2026 (low n_eff, directional only — appendix material)
- **Out of scope for Phase 1:** GSM8K (different difficulty regime; different layer anatomy; not a ThinKV/RaaS benchmark)

### Updated KV signal status
- **kv_key_var:** sign flip confirmed at high n_eff. Cannot be used as fixed-direction signal.
  Worth monitoring; LagKV-style normalization (lag-relative normalization against neighboring
  token statistics) is a candidate fix — literature-grounded (analogous to LagKV's approach).
- **kv_val_var:** NEGATIVE on math500 (−0.135) and math500_eager (−0.145) — not a reliable
  signal. Downgraded from primary online fallback. Both attn_entropy and h2o_attn also sign-flip
  across domains — our signals are not uniquely unstable, but kv_val_var cannot be claimed as
  consistently non-negative.
- **Positioning:** KV signal instability is itself a finding: HS Band A/B is directionally robust
  within a difficulty class; KV signals are context-sensitive. Report both, position KV as a
  secondary contribution.

### Phase 1 deliverables (COMPLETE — April 24, 2026)
1. ✅ HS eviction family in `src/eviction.py`: HSVarianceEviction, DetrendendHSVarianceEviction, BandAdaptiveHSEviction, AttentionHSProductEviction, HybridSegmentHSEviction.
2. ✅ Accuracy vs. cache-size curves: H2O, ThinKV, RaaS, all 5 HS variants, all 4 KV variants on MATH-500 + AIME-2024, eager and flash. Results JSONs in `/data/user_data/skolawol/kvcache/results/phase1/`. Headline numbers below.
3. ✅ Figure 4 plotting script (`scripts/analyze_phase1.py`); accuracy-vs-cache-size PDFs in `reports/phase1_plots/`.

*Last updated: April 24, 2026 — Phase 1 complete; Phase 2 plan in progress.md.*

---

## Phase 1 Results — Headline Numbers (April 24, 2026)

### Accuracy

| Setting | Best FA2-compatible | Best attention-required | Ceiling (none) |
|---|---|---|---|
| MATH-500 @ 4096 | **hs_variance_detrend = 72%** | thinKV = 71% | 75% |
| MATH-500 @ 2048 | band_adaptive_hs / kv_val ≈ 57% | raas = 60% | 75% |
| MATH-500 @ 1024 | hs_variance = 28% | hybrid_seg_hs = 36% | 75% |
| AIME-2024 @ 8192 | **lag_kv = 37%** | h2o / hybrid_seg_hs = 33% | 43% |
| AIME-2024 @ 4096 | lag_kv / hs_variance = 17–20% | thinKV / h2o = 20% | 43% |

The two assertive claims for the paper:
- On MATH-500 at 4096 cache, an FA2-compatible method (`hs_variance_detrend`) is the best non-baseline result, narrowly beating ThinKV (the published SOTA on this problem class).
- On AIME-2024 at 8192 cache, `lag_kv` (FA2-compatible KV-variance with lag normalization) outperforms every attention-based baseline by 3 absolute points (37% vs 33%). With n=30 this is fragile — Phase 2A expands to n=90 across AIME 2024+2025+2026.

The two honest caveats:
- At tight budgets on MATH-500 (512 / 1024), the eager method `hybrid_seg_hs` (which uses both attention sparsity for segment classification and HS for within-segment ranking) is best (7%, 36%). No FA2-compatible method matches it in this regime. **Phase 2 fills this gap with a kv_seg_hs analog** (KV statistics for segment classification + HS for within-segment ranking).
- `hs_variance_detrend` (72%) only beats raw `hs_variance` (71%) by 1 point at 4096. Detrending is a marginal helper at this budget, not a key fix. Worth re-running at lower budgets (256, 512) before claiming detrending is load-bearing.

### Engineering: speed wins, memory doesn't (in the decode regime measured)

- **Speed**: `lag_kv` (FA2) is **2.8× faster than `raas` (eager)** at AIME-2024 @ 8192 (441s vs 1239s mean wall-time). Several FA2 methods are *faster than the no-eviction baseline* at large cache budgets — eviction reduces per-step decode cost while signal extraction adds negligible overhead.
- **Memory**: dominated by cache budget, not method choice. At AIME @ 512, all eviction methods save ~3GB vs `none` (16%). At AIME @ 8192, savings shrink to ~700MB (4%). FA2 vs eager at the same cache budget differs by only ~100–500MB. **Do not claim FA2 memory wins from these decode-only benchmarks.** The known O(n²) → O(n) prefill memory advantage of FA2 is real but lives in long-prompt workloads we don't measure.

### Concrete failure mode for the motivation: H2O collapse

`h2o` on MATH-500 produces **empty generations on 93/100 problems at cache=1024** (no exception, just immediate EOS), 48/100 at 2048, 27/100 at 4096. This matches the attention-map failure mode RaaS documented (24.2% in their analysis). Use this as a concrete example in the paper's motivation: attention-based eviction isn't just suboptimal at tight budgets — it can produce no output at all.

### Updated KV signal status

Phase 1 results refine the §"Updated KV signal status" claims above:
- **kv_key_var**: stable across budgets on MATH-500 (24%/57%/70% at 1024/2048/4096); on AIME, weaker than kv_val (13–23% at high budgets vs 17–33% for kv_val).
- **kv_val_var**: competitive with kv_key on MATH-500; on AIME, weaker at 8192 (33% vs lag_kv 37%).
- **lag_kv (lag-normalized key + value variance)**: the AIME standout — 37% at 8192, the best FA2 result in our suite. Lag normalization is the difference. Promote from "literature-grounded candidate" to "primary KV signal" in the paper. The strategy doc's earlier framing of KV signals as "secondary contribution" is too modest; KV-with-lag-normalization is one of the two headline FA2 methods.

---

## Phase 1 Design Notes — ThinKV Analysis and Literature Survey (April 13, 2026)

### Why lazy deletion matters (and why we can't implement it now)

ThinKV's CT kernel marks evicted tokens with a bit flag instead of performing a gather
(contiguous copy). New decode tokens overwrite evicted slots in-place. This gives ThinKV
zero gather overhead at 95%+ of decode steps. Our `boolean-mask gather` creates a full
contiguous copy each eviction call — this is the main throughput bottleneck.

**Cannot implement lazy deletion with plain HuggingFace DynamicCache** — DynamicCache has
no in-place slot reuse API. ThinKV forked a custom inference backend (or implemented a
custom CUDA kernel). This is an engineering gap, not an algorithmic one. Our wall_time_s
metric in benchmark.py will expose the difference; note it explicitly in the paper.

### New eviction methods — design rationale

**DetrendendHSVarianceEviction** (HS-only, FA2-compatible):
- Motivation: temporal trend contamination (l10 decreases, l21 increases monotonically with
  position). Raw Band A−B score evicts wrong tokens within simple traces.
- Solution: rolling z-score detrending `z(t) = (x(t) − μ_window) / (σ_window + ε)`.
- Principle borrowed from AhaKV (analytical softmax temperature detrending) and LagKV
  (lag-relative minmax normalization). Our adaptation: causal rolling window (past only).
- This is a direct response to the temporal trend caveat documented in §10.

**BandAdaptiveHSEviction** (HS-only, FA2-compatible):
- Motivation: single-layer choice (l10, l21) is fragile; better to aggregate over the full
  Band A (l7–l13) and Band B (l18–l25) structure confirmed in Phase 0B.
- Weight calibration: weight_a / weight_b = 1.29 from Phase 0B ρ ratio (math500_eager:
  l10=0.141, l21=0.109). Both z-scored before weighting, so units are comparable.

**AttentionHSProductEviction** (eager only — needs attn + HS):
- Combines H2O-style cumulative key-perspective attention with detrended Band A HS.
- Hypothesis: tokens important by BOTH signals are more robustly important than by either alone.
- Normalization: each component scored to [0,1] separately; combined = attn_norm + α * hs_norm.
- Eager-only because it requires the full attention matrix (like H2O, ThinKV).

**HybridSegmentHSEviction** (eager only — needs attn + HS):
- Outer loop: ThinKV segment classification by key-perspective entropy (R/E/T type assignment).
- Inner loop: within-segment token ranking by detrended HS Band A−B z-score (replaces
  ThinKV's within-segment attention column-sum ranking).
- Rationale: segment-level R/E/T classification sidesteps the temporal trend problem
  (segments are classified holistically, not per-token). HS replaces attention for
  within-segment ranking — this is the novel hybrid combination.
- Decode-relative indexing: prefill tokens always preserved; only decode tokens ranked by HS.

### Literature survey findings — key papers to cite/beat/borrow

**Direct competitors:**
- **ThinKV (He et al., ICLR 2026 oral)**: segment R/E/T classification via key-perspective
  KDE; lazy deletion CT kernel; NOT validated on standard LLMs; 5.1× generation inflation
  risk from quantization. Our HybridSegmentHS is a direct extension (better within-segment ranker).
- **RaaS (Xu et al., Feb 2025)**: LRU timestamp refreshed on top-50% attention; "milestone"
  and "phoenix" token taxonomy; O(1) per step. Baseline in benchmark.
- **LongFlow (2025)**: ||attn × value||₁ current-step norm; "zero-history estimation";
  DeepSeek-R1 class (same model). DIRECT competitor to HSVarianceEviction.

**Methods to cite for technical principles:**
- **AhaKV**: analytical detrending via adaptive softmax temperature λ=√(2log(i/k)/d). Our
  rolling z-score detrending follows the same principle — normalize for position bias.
- **LagKV**: lag-relative minmax normalization. Our causal rolling window is an adaptation.
- **CAOTE**: closed-form eviction error (α/(1−α))·||VA^T−v_j||₂. Theoretical grounding
  for error-based eviction; could inspire HS-space generalization in future work.
- **LAVa**: per-layer dynamic budget via entropy of importance distribution. Connects to
  Band A/B band-level budget allocation concept.
- **PyramidKV**: layer-adaptive budgets (lower layers need more tokens). Consistent with
  Band A (l7–l13) behavior.

**Our differentiation from all prior work:**
1. First use of layer-indexed HS L2 diff (Band A−B) as KV eviction signal — all prior
   methods use attention-based signals or KV statistics.
2. Explicit temporal detrending for HS signals — AhaKV detrends attention scores, we detrend
   HS diffs. Prior work doesn't apply detrending to HS signals.
3. Bi-directional Band polarity (Band A positive, Band B negative) — theoretically motivated
   by transformer depth anatomy; no prior method has this two-band structure.
4. Phase 0B counterfactual importance labeling — more rigorous validation than proxy
   evaluations used by ThinKV, RaaS, LongFlow.
5. FA2 compatibility for pure HS methods — attention-based methods (ThinKV, RaaS, H2O,
   LongFlow, AttentionHSProduct, HybridSegHS) require eager attn mode.

### Updated method taxonomy for benchmark

| Method | Signal | Attn matrix? | HS? | FA2-compatible? |
|--------|--------|-------------|-----|----------------|
| H2O | cumul. attn col-sum | Yes (eager) | No | No |
| ThinKV | key-perspective entropy | Yes (eager) | No | No |
| RaaS | LRU timestamp (attn) | Yes (eager) | No | No |
| HSVarianceEviction | l10−l21 rolling64 | No | Yes | Yes |
| DetrendendHSVariance | l10−l21 z-scored | No | Yes | Yes |
| BandAdaptiveHS | all Band A/B layers, weighted | No | Yes | Yes |
| AttentionHSProduct | cumul attn + HS z-score | Yes (eager) | Yes | No |
| HybridSegmentHS | ThinKV segs + HS ranking | Yes (eager) | Yes | No |
| KVValVariance | val-vector var rolling64 | No | No | Yes |
| KVKeyVariance | key-vector var rolling64 | No | No | Yes |
| LagKVKey | lag-relative key var | No | No | Yes |
| LagKV | lag-relative key+val | No | No | Yes |
