# Signal Reference: KV Cache Token Importance Proxies

A permanent reference for all signals collected in `collect_traces.py`, including motivation, computation, cost, and known failure modes. Last updated: April 2026.

---

## Background: What Problem Are We Solving?

During autoregressive generation, the KV cache grows linearly with sequence length. At 32k tokens across 32 layers, with 8 heads and head_dim=128, keys+values consume:

```
32 layers × 2 (K+V) × 8 heads × 128 head_dim × 32768 tokens × 2 bytes (fp16)
≈ 32 × 2 × 8 × 128 × 32768 × 2 = 54 GB
```

We cannot keep all tokens. The question is: **which tokens are safe to evict?**

The core hypothesis: **in long reasoning traces, cumulative attention score (H2O) is a poor proxy for importance**. The model may not attend to a token much — but removing it still changes the answer. We want signals that better predict whether masking a token changes the output.

---

## Why Attention-Score Methods Need 2–4× More VRAM

Methods like H2O, SnapKV, ThinKV, and StreamingLLM all require materialising the full `n × n` attention matrix. Here's why that's expensive:

**FlashAttention** (the default in modern transformers) computes attention in tiles — it never builds the full matrix. It reads Q, K, V in blocks and accumulates the output in O(n) memory. Throughput: 2–4× faster; peak memory: O(n) instead of O(n²).

**When you call `output_attentions=True`**, transformers forces eager attention — it disables FlashAttention and builds the full `(batch, heads, n, n)` float32 matrix. At 32k tokens:

```
1 × 32 heads × 32768 × 32768 × 4 bytes = 128 GB per layer (briefly held)
```

Even in fp16, this is prohibitive. In practice, PyTorch spills to CPU or OOMs.

**Our KV-vector signals** (`kv_key_var`, `kv_val_var`, etc.) read tensors **already in GPU memory** — the KV cache is there anyway. No extra computation. Zero VRAM overhead. Fully compatible with FlashAttention.

This is a practical engineering advantage: our method doesn't break when running at scale.

---

## Signal Taxonomy

| Group | Signals | FA2 compatible? | Peak memory | Computation timing |
|---|---|---|---|---|
| KV-vector | kv_key_var, kv_key_norm, kv_val_var, cross_head_var | ✓ Yes | ~free (reads existing KV cache) | Per decode step — fully online |
| Hidden-state | hs_l2_diff, hs_cos_dist, hs_norm | ✓ Yes | O(n) — one prefill pass | Post-hoc (after generation) — offline/batch |
| Attention | h2o_attn, attn_entropy | ✗ No | O(n²) — full n×n matrix per layer | Per decode step — online but costly |

**Why this table matters for the paper**: H2O, SnapKV, and ThinKV all sit in the bottom row — FA2-incompatible, O(n²) memory. Our signals are in the top two rows. Even if only HS signals (offline) beat attention baselines, that is still a meaningful result: better importance estimation with lower memory cost and no FlashAttention regression. If KV-vector signals also work, we additionally gain fully-online eviction with zero overhead. If neither beats attention baselines, we revisit the hypothesis.

Note: ThinKV is also not truly online — it accumulates attention entropy over a window before classifying a thought segment. "Online" in this context means computed incrementally per decode step, not requiring a second pass.

---

## Signal-by-Signal Breakdown

### 1. `kv_key_var` — KV Key Variance

**What it is**: For each token position `t`, take the key vector across all heads and layers. Compute the variance of that vector over the `head_dim` dimension, then average across heads and layers.

**Intuition**: The key vector determines *what this token matches against* when other tokens attend to it. If the key has high variance (spread out in embedding space), the token has a distinctive representational fingerprint — it's likely doing something specific. A key that is near-zero or uniform is an "empty" token that carries little semantic content.

**Why it might work**: Tokens that matter semantically tend to have more structured, high-variance key vectors. Filler tokens (e.g., "let me", "hmm") may produce flatter keys.

**Why it might not**: Variance in post-RoPE keys is inflated by the rotary position encoding at large positions. A token at position 10000 will have higher raw variance than the same token at position 100, purely from the positional rotation — not from content.

**Cost**: Essentially free — we read `past_key_values` which is already on GPU, compute `tensor.var(dim=-1)` per layer, and average. One pass per decode step. No extra memory.

**Note — post-RoPE and layer averaging**: Keys are post-RoPE and averaged across all 32 layers. Two compounding issues: (1) RoPE inflates key variance at large positions regardless of content (see Pre-RoPE section below); (2) averaging all layers dilutes any layer-specific semantic signal — if useful signal is concentrated in layers 12–20, mixing in layers 0–11 and 21–31 degrades the aggregate. Pre-RoPE ablation (Dim 2) and per-layer/upper-weighted KV signals (Dim 3) are both Phase 0B items.

---

### 2. `kv_key_norm` — KV Key L2 Norm

**What it is**: L2 norm of the key vector at each token position, averaged across heads and layers.

**Intuition**: Similar to key variance, but measures magnitude rather than spread. High-norm keys "push" their token into a prominent location in attention space — other tokens are likely to attend to it more.

**Why it might work**: The softmax in attention amplifies tokens with high key-query dot products. A large-norm key will consistently win the dot product competition regardless of the query, making it attended to by many positions.

**Why it might not**: Norm is dominated by a few outlier dimensions. Some models (especially those trained with weight decay) suppress norms uniformly. Not as directional as variance.

**Cost**: Same as kv_key_var — trivially free.

---

### 3. `kv_val_var` — KV Value Variance

**What it is**: Variance of the value vector at each position, averaged across heads and layers.

**Intuition**: The value vector is what actually gets written into the output when a token is attended to. High value variance = the token injects distinctive information into downstream representations. Low value variance = the token contributes a near-uniform, low-information update.

**Why it might work**: This is more directly connected to downstream effect than key variance. If a value vector is high-variance, attending to this token changes the output layer's representation significantly — removing it has a large effect.

**Why it might not**: Value vectors are linearly transformed by the output projection (`W_O`). The "informativeness" of a value depends on the projection, which we don't account for here. Two tokens may have the same value variance but very different output impact.

**Cost**: Free — same as key variance.

---

### 4. `cross_head_var` — Cross-Head Key Variance

**What it is**: For each token position, take the mean key vector across heads (one per layer), then compute the variance of those mean keys across the `head_dim` dimension. Average across layers.

**Intuition**: Each attention head specialises in different aspects of language (syntax, semantics, coreference, etc.). If a token's key vector looks very different across heads, it's a semantically rich token that multiple heads are attending to for different reasons — it's "multi-functionally important". If all heads produce the same key representation, the token is generic.

**Why it might work**: Polysemous or pivotal tokens (conjunctions, math operators, key entities) are likely to be attended to by many specialised heads simultaneously, creating high cross-head variance.

**Why it might not**: Cross-head variance conflates head specialisation with token importance. Some architectures have high intra-layer head diversity by design (e.g., due to low-rank structure in projection matrices).

**Cost**: Slightly more compute than per-head variance — requires stacking and averaging across heads before computing variance. Still trivially free.

---

### 5. `h2o_attn` — H2O Cumulative Attention

**What it is**: Cumulative attention score for each token — sum of attention weights received from all subsequent positions. At each decode step, for token `t`, add the current attention weight `attn[t]` to a running total.

**Intuition**: If many tokens attend to position `t` strongly and repeatedly, `t` is a "heavy hitter" — it's load-bearing in the model's computation. H2O (Heavy Hitter Oracle) is the paper formalising this: evict the tokens with lowest cumulative attention.

**Why it might work**: This is the current SOTA for KV eviction in non-reasoning models. Attention is explicitly what the model uses to retrieve information — if a token is never attended to, the model can generate without it.

**Why it might not**: In reasoning traces, the model often performs long exploratory chains before arriving at an insight. During exploration, the model may not attend to key tokens (because they're in the future of the reasoning, relative to the current position). Attention patterns shift dramatically at epiphanies — a token that was ignored for 5k tokens may suddenly become critical when the model starts citing it.

Additionally, at long contexts H2O suffers from the **attention sink** problem: transformers disproportionately concentrate attention on the first few tokens (positions 0–4) regardless of their semantic content — a softmax normalisation artifact over very long sequences, documented by Xiao et al. (StreamingLLM, ICLR 2024). At 32k tokens, most cumulative attention mass pools at sink positions, making H2O's ranking noisy and biased toward structurally early tokens rather than semantically important ones.

**Cost**: Expensive. Requires materialising the full `n × n` attention matrix at each decode step → forces eager attention → disables FlashAttention → 2–4× VRAM overhead.

---

### 6. `attn_entropy` — Attention Entropy (ThinKV-style)

**What it is**: Shannon entropy of the attention weight distribution at each decode step, for each token position as a destination. More precisely: for token at position `t` in layer `l`, compute `H = -Σ_i attn[i,t] * log(attn[i,t])` over all source positions `i`.

**Intuition**: Low entropy = the attention distribution is peaked (model is focused on specific tokens — "Thinking" mode). High entropy = attention is diffuse (model is spreading attention broadly — "Rambling" mode). ThinKV uses this to classify each thought segment as Reasoning (R), Exploration (E), or Thinking (T), then applies different retention budgets per segment type.

**Why it might work**: Entropy directly measures the model's "confidence" in its attention pattern. During focused computation (mathematical steps, logical deduction), entropy should be low. During rambling or filler generation, entropy should be high — and high-entropy-region tokens are safer to evict.

**Why it might not**: Entropy is a property of the attention distribution, not of individual tokens. Assigning a single entropy score to a token conflates the token's position with the model's current state. Early tokens in a generation look different from late tokens.

Attention entropy also inherits the attention-sink problem: at long contexts, the softmax distribution becomes increasingly spiky toward sink tokens, compressing the entropy range and reducing discriminability between thought-segment types. ThinKV (ICLR 2026) calibrates its R/E/T classifier on sequences up to ~8k tokens; whether its entropy-based segmentation remains valid at 32k is untested.

**Cost**: Same as h2o_attn — requires full attention matrix. Cannot use FlashAttention.

---

### 7. `hs_l2_diff` — Hidden-State L2 Difference

**What it is**: At each token position `t`, compute the L2 norm of the difference between the hidden state at `t` and the hidden state at `t-1`. This measures how much the residual stream "jumps" at each position.

**Intuition**: The residual stream accumulates information as it passes through layers. A large jump at position `t` means this token caused a significant shift in the model's internal representation — it's a "representational pivot". Tokens that don't change the residual stream much are probably redundant.

**Why it might work**: This is the most direct operationalisation of the "epiphany" hypothesis. When the model has an insight, the hidden state should shift sharply. We're looking for tokens that cause those shifts.

**Why it might not**: Large L2 differences may simply reflect long sequences (hidden states drift over time even without epiphanies). Also, the last-layer hidden state includes both semantic content *and* positional information. A long sequence may have monotonically increasing hs_l2_diff even without any pivots.

**Critical limitation — layer choice**: We currently use only the **final transformer layer**. Per transformer interpretability research (ROME/MEMIT — Meng et al. 2022; Geva et al. 2021; logit lens — nostalgebraist 2020), the final layer in a decoder-only model is specialised for next-token prediction: it maps the residual stream toward vocabulary space. Semantic content — factual knowledge, reasoning state, entity representations — peaks in **middle-to-upper layers** (~layers 16–24 for a 32-layer LLaMA-class model). Using the final layer is likely the worst choice for detecting meaningful representational change. Phase 0B Dimension 3 will test layers 16, 20, 24 individually. Expected to substantially improve HS results.

**Cost**: Requires a post-hoc forward pass over the full sequence (up to `--hs_max_len`, which defaults to `--max_new_tokens`). Sequences beyond this limit receive a `-1.0` sentinel and are excluded from correlation analysis. The forward pass uses FlashAttention internally (no n×n attention matrix stored — O(n) peak memory), and takes ~10–30s on an A100 for a 16k-token sequence. Collected once per trace, not per step.

**Timing**: computed after generation completes, so this signal is offline/batch rather than per-decode-step. This is a research design choice, not a memory or compatibility limitation — the forward pass is FA2-compatible and cheaper than any attention-based signal. Whether to pursue real-time HS-based eviction (e.g. via periodic re-evaluation or rolling windows) is a question for after the signal ablation confirms usefulness.

---

### 8. `hs_cos_dist` — Hidden-State Cosine Distance

**What it is**: Cosine distance (1 − cosine similarity) between consecutive hidden states. Measures directional change, independent of magnitude.

**Intuition**: Unlike L2 diff, cosine distance is scale-invariant. It measures whether the *direction* of the representation changed, not just its magnitude. A token that rotates the hidden state into a new subspace is semantically pivotal, even if the magnitude stays similar.

**Why it might work**: Cosine distance controls for the "drift" problem with hs_l2_diff. If L2 norms grow monotonically over a sequence, cosine distance won't — it only triggers on genuine directional shifts, which are more likely epiphany-driven.

**Why it might not**: Cosine distance ignores magnitude entirely. A tiny perturbation in direction may be statistically significant but semantically meaningless. Also, both cosine and L2 use only the **final layer** — which is specialised for next-token prediction, not semantic content. See `hs_l2_diff` note above; same limitation applies here.

**Cost**: Same as hs_l2_diff — one post-hoc forward pass, FA2-compatible, O(n) memory. Offline/batch timing.

---

### 9. `hs_norm` — Hidden-State L2 Norm

**What it is**: L2 norm of the hidden state at each token position. Measures the magnitude of the representation, not its change.

**Intuition**: Tokens with high-magnitude representations are "louder" in the residual stream — they contribute more to attention logits and downstream computations. This is analogous to kv_key_norm but measured at the residual stream rather than the attention key projection.

**Why it might work**: Some interpretability research shows that important tokens (entities, mathematical symbols, logical connectives) tend to have larger residual stream norms, because many layers deposit information onto them.

**Why it might not**: Norm can be dominated by a few outlier tokens (attention sinks, BOS tokens) regardless of semantic content. Also, RMS LayerNorm normalises norms at the start of each attention block — so effective norms are more uniform than raw norms suggest.

**Cost**: Same as hs_l2_diff — one post-hoc forward pass, FA2-compatible, O(n) memory. Offline/batch timing.

---

## Derived Temporal Variants (computed on-the-fly in signal_ablation.py)

For signals `kv_key_var`, `kv_val_var`, `hs_l2_diff`, `kv_key_var_preRoPE`, `kv_key_norm_preRoPE`, and all per-layer HS signals (`hs_l2_diff_l0` … `hs_l2_diff_l31`), we compute two temporal variants:

- **`{signal}_rolling64`**: Rolling mean over the past 64 tokens. Tests whether "sustained elevated signal" predicts importance better than the instantaneous value. Window=64 is ~2× the masking window size used in label_importance.
- **`{signal}_ema09`**: Exponential moving average (α=0.9). Strong recency bias — effectively a soft version of "current signal with short memory of recent past". Standard in signal processing; related to how ThinKV's entropy-based classifier accumulates statistics before making a segment decision.

**Why cumsum was removed**: cumsum of any non-negative signal (variance and L2-diff are always ≥ 0) is monotonically increasing — rank order is just 1, 2, 3, ..., n regardless of signal content. Spearman ρ against any temporally-structured label will be driven by position, not the signal itself. Confirmed empirically: all three cumsum variants produced identical ρ = -0.6043 on the first test trace. Rolling mean avoids this artifact while still testing temporal aggregation.

**Literature context**: There is no specific paper on temporal smoothing of KV-cache signals. EMA and rolling windows are standard in time-series analysis (Holt 1957 for EMA; related to the EWMA literature). For LLM inference specifically, ThinKV accumulates attention entropy over thought-segment windows before classifying — conceptually similar to rolling mean but with segment boundaries rather than a fixed window.

---

## How Attention-Score Methods Differ From Each Other

All of these require materialising attention weights, but they use those weights differently:

| Method | What they measure | Eviction criterion |
|---|---|---|
| **StreamingLLM** | Recency | Keep first K (attention sinks) + last W (recent window); evict everything else |
| **H2O** | Cumulative attention | Sum attn received over all steps; evict lowest-sum tokens |
| **SnapKV** | Prompt-phase attention | Observe which tokens the prompt attends to; pre-emptively cache only those |
| **ThinKV** | Segment-level attention entropy | Classify thought segment as R/E/T from attention entropy; apply per-segment retention budget |
| **RaaS** | Recency (decode) + preserved (prefill) | LRU eviction of decode tokens; unconditionally preserve all prefill tokens |

The key distinction:
- StreamingLLM: pure recency, no content signal
- H2O: global importance signal (cumulative attention), but blind to temporal dynamics
- SnapKV: prefix-phase signal, ignores decode-phase dynamics entirely
- ThinKV: segment-level signal (thought-type classification), not per-token
- RaaS: structural signal (prefill = always important), recency for decode

**Our approach**: per-token, compute-free signal from KV vectors already in cache. No attention matrix required.

---

## Phase 0B Signals

Two new signal groups added in Phase 0B collection (`--phase0b` flag):

### `kv_key_var_preRoPE` / `kv_key_norm_preRoPE` — Pre-RoPE Key Variance / Norm

**What it is**: Same as `kv_key_var` / `kv_key_norm`, but computed from keys **before** the rotary positional encoding is applied. Captured via a forward hook on `layer.self_attn.k_proj`; accumulated and averaged across all 32 layers.

**Why**: Post-RoPE keys have position-dependent variance inflation (token at position 10000 has higher raw variance than the same token at position 1, purely from the rotational angle). Pre-RoPE keys are "pure content" — positional information has not yet been mixed in. Hypothesis: pre-RoPE key variance is a cleaner importance signal.

**Cost**: Requires one full forward pass with hooks registered on all 32 `k_proj` modules. FA2-compatible (no attention matrix materialized). O(n) memory. Collected posthoc from saved token IDs via `extract_phase0b_signals.py`.

---

### `hs_l2_diff_l0` … `hs_l2_diff_l31` — Per-Layer Hidden-State L2 Diff (all 32 layers)

**What it is**: Same computation as `hs_l2_diff` (consecutive hidden-state L2 difference) but computed at every transformer layer (0–31) rather than only the final layer.

**Why**: The final transformer layer is specialised for next-token prediction (vocabulary projection). Semantic content — factual associations, reasoning state, entity representations — peaks in middle-to-upper layers (~16–24 for a 32-layer LLaMA-class model; see ROME/MEMIT, logit lens). Rather than guessing which specific layer to sample, we collect all 32 and let `signal_ablation.py` sweep over them empirically. Storage cost is negligible (each signal is a 1D float32 array of length `seq_len`; 32 layers × 16k tokens × 4 bytes ≈ 2MB/trace).

**Cost**: `output_hidden_states=True` materializes all 33 hidden states (embedding + 32 layer outputs) in memory during the single post-hoc forward pass regardless of how many we save. Choosing to store all 32 costs zero additional compute. FA2-compatible, O(n) memory.

---

## Pre-RoPE vs Post-RoPE (Phase 0B Ablation)

KV cache stores **post-RoPE** keys — the rotary positional encoding has already been applied. RoPE rotates each token's key vector by an angle proportional to its position. This means:

- At position 10000, the key vector is rotated 10000× more than at position 1.
- Raw variance/norm of post-RoPE keys will increase with position purely from the rotation — not from content.

**Pre-RoPE keys** require a forward hook to capture the key vectors before `apply_rotary_pos_emb()` is called. These are "pure content" keys, stripped of positional information. Our hypothesis: pre-RoPE key variance is a cleaner importance signal.

This ablation (Dimension 2) is deferred to Phase 0B. If the Phase 0 results show kv_key_var is promising, we run Phase 0B to check whether pre-RoPE improves it further.

---

## Interpretation Guide: What to Expect from Phase 0

When `signal_ablation.py` finishes, you'll see a Spearman ρ table. Here's how to read it:

- **ρ > 0**: Signal correlates with importance (higher signal → more likely to be important). Good.
- **ρ < 0**: Signal *anti-correlates* (higher signal → more likely unimportant). Also useful — just invert it.
- **|ρ| < 0.05**: Signal is basically noise.
- **|ρ| > 0.15**: Practically meaningful correlation.
- **|ρ| > 0.30**: Strong correlation — publishable signal.

**Key comparison**: Does any residual-stream signal beat `h2o_attn` in Spearman ρ?
- If **yes**: Core hypothesis validated. Proceed to Dimensions 2–5 to find the best configuration.
- If **no**: Hypothesis needs revision. Check which signals came closest. Consider: are our counterfactual labels reliable? Is 30 traces enough? Did the model actually answer anything correctly?

**attn_entropy vs h2o_attn**: These are not directly comparable — they measure different things. h2o_attn measures how much a token was *attended to* by future tokens (a retrospective reuse signal). attn_entropy measures how focused the model *was* when generating each token (a concurrent cognitive-state signal). attn_entropy's ρ will be negative when it works correctly: low entropy = focused/R-type token = more likely to be important.

More importantly: ThinKV's superiority over H2O on task accuracy comes from **segment-level R/E/T classification + different retention budgets per type**, not from attn_entropy having higher per-token Spearman ρ than h2o_attn. Token-level correlation analysis cannot determine whether ThinKV beats H2O. That comparison requires Phase 1 accuracy vs. cache-size curves.

---

## Phase 0B Ablation Results (April 2026)

> Full results, tables, confidence intervals, and Phase 1 recommendation in
> `experiments/phase0b_ablation_results.md`. This section is a condensed reference.
> Phase 0 preliminary results (March 2026, truncation-based labels) are archived in
> `data/phase_OA/` and superseded by the results below.

### Best signals per dataset (rolling64, sorted by |ρ|)

| Signal | math500 | math500_eager | aime2024 | aime2024_eager |
|---|---|---|---|---|
| kv_key_var | **+0.380** | +0.214 | −0.203 | −0.022 |
| hs_l2_diff best layer | −0.254 (l23) | −0.202 (l1) | −0.178 (l31) | **−0.227 (l21)** |
| attn_entropy (eager only) | — | +0.176 | — | −0.139 |
| cross_head_var | +0.207 | +0.101 | −0.094 | +0.006 |
| kv_val_var | −0.135 | −0.145 | +0.105 | −0.060 |
| h2o_attn (eager only) | — | **+0.050** | — | **−0.011** |

**h2o_attn is the weakest signal in every dataset.** attn_entropy outperforms it by
3–12×. Multiple HS and KV signals outperform h2o_attn across the board. Core hypothesis
confirmed: attention score is a poor token importance proxy.

### Definitive null results

| Ablation | Finding |
|---|---|
| preRoPE vs postRoPE | Max Δρ = 0.0005. RoPE contamination is negligible. Do not collect preRoPE signals. |
| hs_cos_dist vs hs_l2_diff | hs_cos_dist is consistently weaker at every layer. Cosine distance discards scale information that is predictive. |
| hs_norm | Max |ρ| = 0.087 (aime2024_eager). Weak and inconsistent. |
| Redundant signals | kv_key_norm ≈ kv_key_var (Δρ < 0.001). kv_key_{var,norm}_preRoPE ≈ postRoPE. hs_l2_diff = hs_l2_diff_l31. |

### Layer anatomy (hs_l2_diff_lN rolling64)

Two bands with consistent sign across all 4 datasets:

**Band A — layers 7–13: consistently POSITIVE ρ**
High hs_l2_diff at these layers → token is IMPORTANT (label=1).
Eviction rule: keep high-signal tokens, evict low-signal tokens.
Peak: l10 in math500_eager (ρ=+0.141), l8 in aime2024_eager (ρ=+0.155).
Interpretation: mid-network feature routing / factual retrieval layers (per ROME/MEMIT).
Important tokens cause larger hidden-state changes here.

**Band B — layers 18–25: consistently NEGATIVE ρ**
High hs_l2_diff at these layers → token is DISPENSABLE (label=0).
Eviction rule: keep low-signal tokens, evict high-signal tokens.
Peak: l23 in math500 (ρ=−0.254), l21 in aime2024_eager (ρ=−0.227).
Interpretation: upper-mid output-preparation layers. Large changes may mark verbose or
transitional tokens.

**Unstable layers:**
- l0: mixed (negative in math500 but positive in aime2024_eager)
- l14–l17: near-zero in math500, moderately negative in AIME — dataset-dependent
- l31 (last layer): strongly positive in math500 (+0.220), strongly negative in aime2024
  (−0.178). Follows same sign-flip pattern as kv_key_var. Do not use as universal signal.

### Sign flip between easy and hard problems

Most signals (kv_key_var, kv_val_var, cross_head_var, hs_l2_diff_l31, attn_entropy) flip
sign between math500 (~0.20 important_frac) and aime2024 (~0.52 important_frac). This is
driven by label density: when most tokens are important (AIME), the rare dispensable tokens
have different signal characteristics than when most tokens are dispensable (math500).

Only Band A (l7–l13) and Band B (l18–l25) maintain consistent sign across both datasets.

**Statistical caveat**: aime2024 has 14 labelled traces, aime2024_eager has 11. With these
effective sample sizes, 95% confidence intervals span approximately ±0.5–0.6 around any
point estimate. All AIME-side ρ values are consistent with zero. Extending to AIME 2025
and AIME 2026 (MathArena datasets) is required before AIME conclusions can be trusted.

### Temporal smoothing

Rolling64 outperforms ema09 outperforms raw by 30–57% universally.
Use rolling64 for all signals. Rolling window of 64 ≈ 2× masking window size (32).

### Phase 1 recommended signal

Combined: `hs_l2_diff_l10_rolling64 − hs_l2_diff_l21_rolling64` — exploits both bands.
Tokens high in Band A and low in Band B are doubly important. Tokens low in Band A and
high in Band B are doubly dispensable. Evict tokens with lowest combined score.

**Online computation (during generation — no extra forward pass):**
At each decode step, the current token's hidden states at all layers are already computed
via `output_hidden_states=True`. To get `hs_l2_diff_l10` for the new token, store the
previous token's HS at layers 10 and 21 (two vectors, ~20KB for a 5120-dim model) and
diff against the current step's HS. Apply causal rolling64 (mean of last 64 values).
Store the resulting importance score per token. Evict by ranking stored scores when cache
exceeds budget. No refresh cycle needed — hs_l2_diff is a fixed local property at
generation time, not context-dependent (unlike attention-based signals). Cost: negligible
additional memory vs KV signals (which read already-cached tensors with no extra storage).

Online fallback if HS unavailable: `kv_key_var_rolling64`. FlashAttention-compatible,
zero extra storage. Sign is task-difficulty-dependent; use with awareness of context.
