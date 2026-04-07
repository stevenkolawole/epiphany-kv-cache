# Paper Strategy — NeurIPS Submission

**Venue:** NeurIPS (9-page limit, ~4 figures typical, appendix unlimited)
**Status:** Pre-writing; strategy scratchpad. Update as Phase 1+ results arrive.

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

## Open Questions (revisit as results come in)

- Does the Phase 1 combined score (l10_rolling64 − l21_rolling64) hold up on actual
  eviction, or does Band B subtraction hurt in practice?
- Do AIME 2025/2026 results replicate Band A/B anatomy with sufficient n_eff?
- Can we quantify the latency cost of attention-based scoring vs ours on real hardware?
  (Would make Figure 1 Panel B quantitative rather than qualitative.)
- Is there an explanation for *why* Band A (l7–l13) is predictive and Band B (l18–l25)
  has opposite sign? A mechanistic account would strengthen the method section.

---

*Last updated: April 2026 — pre-Phase 1, post Phase 0B ablation*
