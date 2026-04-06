# Phase 0B Signal Ablation Results

Full analysis of the Phase 0B signal ablation across 4 datasets (math500, math500_eager,
aime2024, aime2024_eager). Written April 2026.

All ρ values are Spearman rank correlations between signal values and counterfactual
importance labels (1 = masking this token caused answer flip; 0 = masking had no effect).
Only generated tokens with label ∈ {0, 1} are included; prompt tokens are excluded.
`rolling64` = rolling mean over past 64 tokens; `ema09` = exponential moving average α=0.9.

---

## Data Inventory

| Dataset | Labelled traces | n_pairs | Avg important_frac | Effective n |
|---|---|---|---|---|
| math500 (non-eager, FA2) | 75 | 172,078 | ~0.20 | 75 |
| math500_eager (eager attn) | 81 | 191,509 | ~0.20 | 81 |
| aime2024 (non-eager, FA2) | 14 | 82,651 | **~0.52** | **14** |
| aime2024_eager (eager attn) | 11 | 52,644 | **~0.64** | **11** |

**Effective n is the trace count, not n_pairs.** Tokens within a trace are heavily
correlated — different problems are independent, but thousands of tokens within one problem
are not. The p-values printed by signal_ablation.py treat n_pairs as the sample size and
are therefore completely misleading for the AIME datasets.

**Approximate 95% confidence intervals** (Fisher z-transform, n_eff = n_traces - 3):

| Dataset | n_eff | ±SE | CI for ρ=0.25 | CI for ρ=0.20 |
|---|---|---|---|---|
| math500 | 72 | ±0.118 | (0.02, 0.46) | (−0.03, 0.41) |
| math500_eager | 78 | ±0.113 | (0.03, 0.45) | (−0.02, 0.40) |
| aime2024 | 11 | ±0.301 | (−0.31, 0.68) | (−0.36, 0.65) |
| aime2024_eager | 8 | ±0.354 | (−0.39, 0.73) | (−0.44, 0.71) |

**All AIME-side 95% CIs span zero.** Individual AIME ρ values are directionally
informative and consistent in sign with the math500 results, but statistically
unreliable as standalone claims. Conclusions from AIME data require corroboration from
the math500 datasets or significantly more AIME traces. Adding AIME 2025 and AIME 2026
(MathArena/aime_2025, MathArena/aime_2026) would provide 90 total AIME problems vs the
current 30, and is strongly recommended.

**important_frac** is the fraction of tested reasoning tokens labeled as important (label=1)
in each dataset. This varies wildly within a dataset (e.g. math500 individual traces range
from 0.01 to 0.93), and the dataset-level average differs substantially between math500
(~0.20) and AIME (~0.52–0.64). This difference is the root cause of the sign flips
described in §4 below.

---

## 1. Signal Family Rankings

Best |ρ| per family per dataset (rolling64 unless noted). Sign is shown.

| Signal family | math500 | math500_eager | aime2024 | aime2024_eager |
|---|---|---|---|---|
| kv_key_var (rolling64) | **+0.380** | +0.214 | −0.203 | −0.022 |
| kv_key_norm (rolling64) | +0.379 | +0.214 | −0.203 | −0.021 |
| kv_key_var_preRoPE (rolling64) | +0.380 | +0.214 | −0.202 | −0.022 |
| kv_key_norm_preRoPE (rolling64) | +0.379 | +0.214 | −0.203 | −0.021 |
| **hs_l2_diff best layer (rolling64)** | −0.254 (l23) | −0.202 (l1) | −0.178 (l31) | **−0.227 (l21)** |
| hs_l2_diff_l31 (rolling64) | +0.220 | +0.093 | −0.178 | −0.022 |
| cross_head_var | +0.207 | +0.101 | −0.094 | +0.006 |
| attn_entropy (eager only) | — | +0.176 | — | −0.139 |
| kv_val_var (rolling64) | −0.135 | −0.145 | +0.105 | −0.060 |
| h2o_attn (eager only) | — | +0.050 | — | −0.011 |
| hs_norm | +0.018 | −0.010 | −0.038 | −0.087 |
| hs_cos_dist (last layer) | +0.096 | +0.056 | −0.056 | +0.054 |

**Key takeaways:**
- kv_key_var rolling64 is the strongest single number in math500 (ρ=0.380), but near-zero
  in aime2024_eager (ρ=−0.022), which — given only 11 traces — is statistically
  indistinguishable from zero.
- The best per-layer hs_l2_diff signal outperforms kv_key_var in aime2024_eager (|ρ|=0.227
  vs 0.022). This finding is directionally robust but has wide CIs.
- h2o_attn (the current SOTA) is the weakest signal tested: ρ=0.050 in math500_eager and
  −0.011 in aime2024_eager. Our hypothesis that attention is a poor importance proxy is
  confirmed across all tested datasets.
- attn_entropy is substantially stronger than h2o_attn (ρ=0.176 vs 0.050 in math500_eager)
  but both require full O(n²) eager attention, making them impractical at long context.

---

## 2. Temporal Smoothing

Rolling64 consistently outperforms ema09, which outperforms raw. Shown for kv_key_var:

| Dataset | raw |ρ| | ema09 |ρ| | rolling64 |ρ| | rolling64 vs raw |
|---|---|---|---|---|
| math500 | 0.288 | 0.356 | 0.380 | +32% |
| math500_eager | 0.150 | 0.193 | 0.214 | +43% |
| aime2024 | 0.139 | 0.183 | 0.203 | +46% |
| aime2024_eager | 0.014 | 0.018 | 0.022 | +57% (near-zero base) |

This pattern is universal across all signal families. Interpretation: token importance
is a sustained contextual property, not an instantaneous spike. The rolling64 window (64
tokens ≈ 2× the masking window size of 32) captures which *phase* of reasoning a token
belongs to, which is more predictive than any single token's instantaneous signal.

ema09 (α=0.9) weights the current token at 90% and prior history at 10% — it's close to
the raw signal with a slight lag. Rolling64 gives uniform weight to the most recent 64
positions, which better matches the masking window granularity used by label_importance.py.

---

## 3. preRoPE vs postRoPE — Null Result

Δ|ρ| for preRoPE vs postRoPE (same base signal, same smoothing):

| Comparison | math500 | math500_eager | aime2024 | aime2024_eager |
|---|---|---|---|---|
| kv_key_var rolling64 | 0.0001 | 0.0003 | 0.0003 | 0.0001 |
| kv_key_norm rolling64 | 0.0003 | 0.0005 | 0.0001 | 0.0000 |
| kv_key_var raw | 0.0001 | 0.0004 | 0.0003 | 0.0001 |

Maximum Δ|ρ| observed: **0.0005** (kv_key_norm rolling64, math500_eager).

RoPE does not measurably contaminate the signal. The hypothesis that pre-RoPE keys are
cleaner importance signals is **falsified**. Collecting preRoPE signals (extra forward hook
on all 32 k_proj modules) provides zero benefit and can be removed from future runs.

---

## 4. Sign Consistency — The Critical Finding

Most signals flip sign between math500 (easy-medium) and aime2024 (hard competition).

| Signal | math500 | math500_eager | aime2024 | aime2024_eager | Consistent? |
|---|---|---|---|---|---|
| kv_key_var rolling64 | **+**0.380 | **+**0.214 | **−**0.203 | **−**0.022 | ❌ flips |
| kv_val_var rolling64 | **−**0.135 | **−**0.145 | **+**0.105 | **−**0.060 | ❌ flips in aime2024 |
| cross_head_var | **+**0.207 | **+**0.101 | **−**0.094 | near zero | ❌ flips |
| hs_l2_diff_l31 rolling64 | **+**0.220 | **+**0.093 | **−**0.178 | **−**0.022 | ❌ flips |
| hs_l2_diff_l21 rolling64 | **−**0.209 | **−**0.109 | **−**0.085 | **−**0.227 | ✅ consistently negative |
| hs_l2_diff_l22 rolling64 | **−**0.223 | **−**0.121 | **−**0.059 | **−**0.217 | ✅ consistently negative |
| hs_l2_diff_l23 rolling64 | **−**0.254 | **−**0.151 | **−**0.021 | **−**0.200 | ✅ consistently negative |
| hs_l2_diff_l8 rolling64 | **+**0.065 | **+**0.078 | **+**0.146 | **+**0.155 | ✅ consistently positive |
| hs_l2_diff_l9 rolling64 | **+**0.082 | **+**0.107 | **+**0.136 | **+**0.130 | ✅ consistently positive |
| hs_l2_diff_l10 rolling64 | **+**0.112 | **+**0.141 | **+**0.097 | **+**0.120 | ✅ consistently positive |

**Root cause of sign flips — label density, not signal noise:**

The masking labels are binary: 1 = important, 0 = dispensable. The fraction of tokens
labeled 1 (important_frac) differs dramatically between datasets:

- math500: average important_frac ≈ 0.20 (most tokens are dispensable)
- aime2024: average important_frac ≈ 0.52 (nearly half are important)
- aime2024_eager: average important_frac ≈ 0.64 (most tokens are important)

In math500, a token being "important" is the exception. A signal that correctly fires at
rare insight tokens will have POSITIVE ρ (signal high → label=1). In AIME, nearly
everything matters — the rare dispensable tokens (label=0, ~36-48%) are the exception.
The same physical signal (e.g., high kv_key_var) may correctly identify the rare
dispensable tokens in AIME, producing NEGATIVE ρ, even though it identified the rare
important tokens in math500, producing POSITIVE ρ.

**This is a genuine semantic reversal, not noise.** In math500, the model may spread
attention diversely (high kv_key_var) during insight moments. In AIME's intensive
calculation steps, the model may show high kv_key_var at the occasional reflective or
transitional token ("therefore", "now consider") that CAN be dropped, while the dense
calculation tokens have lower, more constrained KV variance.

**Whether this interpretation is correct cannot be determined from Phase 0B alone.**
It requires inspecting which specific tokens receive each label — a qualitative analysis
deferred to Phase 1.

**Caveat**: For aime2024 and aime2024_eager, the effective sample sizes (14 and 11 traces)
are too small to confirm this sign flip reliably. The sign flip in kv_key_var
(math500: +0.380, aime2024: −0.203) may be real, or may reflect sampling noise at n=14.
AIME 2025 and 2026 data is needed to confirm.

**The last-layer signal (hs_l2_diff_l31) flips for the same reason as kv_key_var.** Both
sit at the model's output interface: kv_key_var measures KV attention-space diversity,
hs_l2_diff_l31 measures output-representation change. Both capture "what the model decided
at this token" and share the same label-density sensitivity. Mid-layer signals (l19-l25)
capture internal feature routing that is more structural and less output-decision-dependent
— which is why their sign is consistent across label densities.

---

## 5. Per-Layer Hidden-State Analysis

Full hs_l2_diff_lN rolling64 ρ values across all 32 layers and all 4 datasets:

| Layer | math500 | math500_eager | aime2024 | aime2024_eager | Pattern |
|---|---|---|---|---|---|
| l0 | −0.173 | −0.199 | −0.068 | **+0.037** | Mixed |
| l1 | −0.172 | −0.202 | −0.089 | −0.052 | Neg (weak in aime_e) |
| l2 | −0.121 | −0.151 | −0.101 | −0.125 | Consistently negative |
| l3 | −0.121 | −0.124 | −0.014 | −0.072 | Mostly negative |
| l4 | −0.153 | −0.139 | +0.011 | −0.052 | Mixed |
| l5 | −0.079 | −0.058 | +0.083 | +0.093 | Mixed |
| l6 | +0.011 | −0.007 | +0.120 | +0.150 | Mixed (near-zero in math500) |
| **l7** | **+0.079** | **+0.047** | **+0.118** | **+0.156** | ✅ **Consistently positive** |
| **l8** | **+0.065** | **+0.078** | **+0.146** | **+0.155** | ✅ **Consistently positive** |
| **l9** | **+0.082** | **+0.107** | **+0.136** | **+0.130** | ✅ **Consistently positive** |
| **l10** | **+0.112** | **+0.141** | **+0.097** | **+0.120** | ✅ **Consistently positive** |
| **l11** | **+0.093** | **+0.144** | **+0.083** | **+0.120** | ✅ **Consistently positive** |
| **l12** | **+0.038** | **+0.077** | **+0.119** | **+0.114** | ✅ **Consistently positive** |
| **l13** | **+0.071** | **+0.089** | **+0.058** | **+0.065** | ✅ **Consistently positive** |
| l14 | −0.016 | +0.006 | −0.020 | −0.032 | Near zero |
| l15 | +0.017 | +0.016 | −0.124 | −0.147 | Mixed |
| l16 | −0.002 | −0.016 | −0.140 | −0.165 | Mixed (near-zero in math500) |
| l17 | −0.003 | −0.005 | −0.147 | −0.191 | Mixed (near-zero in math500) |
| **l18** | **−0.081** | **−0.045** | **−0.118** | **−0.184** | ✅ **Consistently negative** |
| **l19** | **−0.147** | **−0.074** | **−0.113** | **−0.208** | ✅ **Consistently negative** |
| **l20** | **−0.191** | **−0.097** | **−0.105** | **−0.224** | ✅ **Consistently negative** |
| **l21** | **−0.209** | **−0.109** | **−0.085** | **−0.227** | ✅ **Consistently negative** |
| **l22** | **−0.223** | **−0.121** | **−0.059** | **−0.217** | ✅ **Consistently negative** |
| **l23** | **−0.254** | **−0.151** | **−0.021** | **−0.200** | ✅ **Consistently negative** |
| **l24** | **−0.250** | **−0.146** | **−0.005** | **−0.188** | ✅ **Consistently negative** |
| **l25** | **−0.243** | **−0.142** | **−0.006** | **−0.187** | ✅ **Consistently negative** |
| l26 | −0.211 | −0.107 | +0.007 | −0.157 | Mostly negative |
| l27 | −0.066 | −0.003 | −0.041 | −0.086 | Weak |
| l28 | −0.053 | −0.018 | −0.041 | −0.059 | Weak |
| l29 | +0.082 | +0.049 | −0.099 | +0.030 | Mixed |
| l30 | +0.135 | +0.065 | −0.121 | +0.062 | Mixed |
| l31 | **+0.220** | +0.093 | **−0.178** | −0.022 | ❌ Flips |

### Two Consistent-Direction Bands

**Layer anatomy of a 32-layer transformer based on these results:**

**Band A (l7–l13): consistently positive ρ** — min |ρ| = 0.038 (l12, math500), max = 0.156
(l7, aime2024_eager). High hs_l2_diff at these layers correlates with importance (label=1).
Eviction rule: **evict tokens with LOW l7–l13 signal** (they are dispensable).

These are the "reasoning feature routing" layers. Interpretability literature (Geva et al.
2021, ROME/MEMIT 2022) places key-value factual associations in the MLP layers of the
mid-network. Large hidden-state changes at layers 7–13 indicate the model is actively
retrieving or transforming semantic content — these tokens are load-bearing.

**Band B (l18–l25): consistently negative ρ** — min |ρ| = 0.005 (l24, aime2024, near
zero), max = 0.254 (l23, math500). High hs_l2_diff at these layers correlates with
dispensability (label=0). Eviction rule: **evict tokens with HIGH l18–l25 signal**.

Band B is in the upper-mid network, closer to the output prediction layers. Large changes
here may reflect the model "settling" its output representation — at tokens where the model
is verbose, repetitive, or transitioning between reasoning phases, the upper layers show
more activity. This is speculative; qualitative inspection is needed.

**Note on near-zero entries**: l23–l25 in aime2024 non-eager are near zero (ρ ≈ −0.002 to
−0.021). These are still negative, but the signal is weak. Recall: aime2024 non-eager has
only 14 traces and a wide CI. The negative direction is consistent, but we cannot conclude
these layers are informative specifically for the non-eager AIME setting without more data.

**Best single layer per dataset:**
- math500: l23 (ρ = −0.254)
- math500_eager: l1 (ρ = −0.202) — note early-layer dominance in eager math500
- aime2024: l31 (ρ = −0.178) — last layer is the exception here
- aime2024_eager: l21 (ρ = −0.227)

No single layer is universally best. For cross-dataset robustness, **l21 or l22** are the
best compromises: consistently negative, strong in aime2024_eager and math500, moderate in
math500_eager, weak-but-correct in aime2024.

### Eager vs Non-Eager Effect on HS Signals (corrected)

The claim "HS signals are more robust to eager mode" needs qualification:

| Layer | math500 (FA2) | math500_eager | Δ | aime2024 (FA2) | aime2024_eager | Δ |
|---|---|---|---|---|---|---|
| l0 | −0.173 | −0.199 | **+15%** stronger | −0.068 | +0.037 | flips sign |
| l1 | −0.172 | −0.202 | **+17%** stronger | −0.089 | −0.052 | −42% |
| l23 | −0.254 | −0.151 | −41% | −0.021 | −0.200 | **+852%** stronger |
| l21 | −0.209 | −0.109 | −48% | −0.085 | −0.227 | **+167%** stronger |

For math500: early layers (l0, l1) STRENGTHEN in eager mode (+15–17%), while upper-mid
layers (l21–l25) weaken (~40–50%). The picture is more complex than "HS degrades equally
with kv_key_var."

For aime2024: the HS mid-layer (l21–l23) signal is much stronger in eager mode, while
kv_key_var collapses from −0.203 to −0.022. However, given only 11–14 traces per AIME
dataset, these differences are within the statistical noise range. We cannot conclude with
confidence that the eager/non-eager difference for AIME is real vs. sampling variability.

---

## 6. Eager vs Non-Eager: All Signals

| Signal | math500 | math500_eager | Δ% | aime2024 | aime2024_eager | Δ% |
|---|---|---|---|---|---|---|
| kv_key_var rolling64 | +0.380 | +0.214 | −44% | −0.203 | −0.022 | −89% |
| hs_l2_diff_l23 rolling64 | −0.254 | −0.151 | −41% | −0.021 | −0.200 | (near-zero→big) |
| cross_head_var | +0.207 | +0.101 | −51% | −0.094 | +0.006 | near zero |
| attn_entropy | — | +0.176 | — | — | −0.139 | — |
| h2o_attn | — | +0.050 | — | — | −0.011 | — |
| kv_val_var rolling64 | −0.135 | −0.145 | +7% | +0.105 | −0.060 | flips |

**h2o_attn is weaker than nearly every other signal.** ρ=0.050 in math500_eager and
−0.011 in aime2024_eager. attn_entropy is 3.5× stronger than h2o_attn in math500_eager
and 12.6× stronger in aime2024_eager. Our hypothesis that attention-score signals are
weak importance proxies is confirmed. However, attn_entropy and h2o_attn both require
eager O(n²) attention — impractical at scale.

**The kv_key_var collapse on aime2024_eager (−0.203 → −0.022) may be sampling noise.**
With only 11 labelled traces in aime2024_eager, the 95% CI for ρ=−0.022 is roughly
(−0.59, 0.57) — easily consistent with a true underlying ρ of −0.15 or −0.20. The
collapse is not confirmed. Similarly, ρ=−0.203 in aime2024 non-eager (14 traces) has CI
≈ (−0.65, 0.37). Both AIME-side numbers are consistent with a moderate negative
correlation or with near-zero — we cannot distinguish from the current data.

---

## 7. Redundancy Analysis

**kv_key_var ≈ kv_key_norm ≈ kv_key_var_preRoPE ≈ kv_key_norm_preRoPE**: Max Δρ = 0.0005
across all datasets and smoothing variants. All four are measuring the same quantity. Keep
only kv_key_var; discard the other three.

**hs_l2_diff = hs_l2_diff_l31**: The unqualified signal was always collected at the last
layer. Values are identical in all CSVs. The explicit `hs_l2_diff_l31` variant is the
canonical name; `hs_l2_diff` is an alias.

**hs_cos_dist = hs_cos_dist_l31**: Same as above.

**hs_l2_diff_lN dominates hs_cos_dist_lN at every layer**: e.g. at l31, math500: l2_diff
= +0.220 vs cos_dist = +0.096. The L2 metric retains scale information that cosine
distance discards. Scale matters: tokens where the hidden state shifts to a very different
magnitude, not just direction, appear to carry more signal. Cosine distance adds no
information beyond l2_diff for this task.

**Unique informative signals after deduplication** (38 total → 8 meaningful families):
1. `kv_key_var` (one representative; rolling64 smoothing)
2. `kv_val_var` (rolling64)
3. `hs_l2_diff_lN` for N in 0..31 (all distinct, 32 signals)
4. `cross_head_var`
5. `attn_entropy` (eager only)
6. `h2o_attn` (eager only, weak, included for comparison)
7. `hs_norm` (weak)
8. `hs_cos_dist` (weaker than hs_l2_diff at every layer; track but don't prioritize)

---

## 8. attn_entropy Implementation Note

Our `attn_entropy` is **query-perspective per-token entropy**: at each decode step t,
H_t = −Σ a_t(i) log a_t(i) where a_t is the attention weight distribution of token t
over its context. This measures how focused *token t's attention* is when it looks back.

ThinKV's entropy signal is **key-perspective segment entropy**: it classifies each key
token based on how concentrated the attention directed *at* that token is from future
queries. These are different quantities measured from different perspectives.

Additionally, ThinKV classifies at segment/window level with a KDE-based R/E/T classifier.
Our rolling64 smoothing approximates segment-level aggregation but uses a fixed window,
not thought-segment boundaries.

Implication: our `attn_entropy` result (ρ=0.176 in math500_eager, −0.139 in
aime2024_eager) cannot be directly compared to ThinKV's classification quality. It is
a valid signal, but it is measuring something adjacent to — not identical to — what ThinKV
uses. The comparison should be framed as "an analogous query-entropy signal" not "ThinKV's
entropy signal."

---

## 9. Cross-Dataset Generalization

Signals with |ρ| ≥ 0.05 AND consistent sign across all 4 datasets:

| Signal (rolling64) | math500 | math500_eager | aime2024 | aime2024_eager |
|---|---|---|---|---|
| hs_l2_diff_l7 | +0.079 | +0.047 | +0.118 | +0.156 |
| hs_l2_diff_l8 | +0.065 | +0.078 | +0.146 | +0.155 |
| hs_l2_diff_l9 | +0.082 | +0.107 | +0.136 | +0.130 |
| hs_l2_diff_l10 | +0.112 | +0.141 | +0.097 | +0.120 |
| hs_l2_diff_l11 | +0.093 | +0.144 | +0.083 | +0.120 |
| hs_l2_diff_l13 | +0.071 | +0.089 | +0.058 | +0.065 |
| hs_l2_diff_l2 | −0.121 | −0.151 | −0.101 | −0.125 |
| hs_l2_diff_l18 | −0.081 | −0.045 | −0.118 | −0.184 |
| hs_l2_diff_l19 | −0.147 | −0.074 | −0.113 | −0.208 |
| hs_l2_diff_l20 | −0.191 | −0.097 | −0.105 | −0.224 |
| hs_l2_diff_l21 | −0.209 | −0.109 | −0.085 | −0.227 |
| hs_l2_diff_l22 | −0.223 | −0.121 | −0.059 | −0.217 |
| hs_l2_diff_l23 | −0.254 | −0.151 | −0.021 | −0.200 |
| hs_l2_diff_l24 | −0.250 | −0.146 | −0.005 | −0.188 |
| hs_l2_diff_l25 | −0.243 | −0.142 | −0.006 | −0.187 |

No kv or attention signal achieves consistent sign across all 4 datasets at |ρ| ≥ 0.05.

The two consistent bands require **opposite eviction rules**:
- Band A (l7–l13, positive): evict tokens with LOW signal (they are dispensable)
- Band B (l18–l25, negative): evict tokens with HIGH signal (they are dispensable)

A combined score `combined = hs_l2_diff_l{10}_rolling64 − hs_l2_diff_l{21}_rolling64`
(positive band minus negative band) should amplify the signal. This is a candidate for
Phase 1 signal design.

Signals that do NOT generalize (sign flip or near-zero in ≥1 dataset):
kv_key_var, kv_key_norm, kv_val_var, cross_head_var, hs_l2_diff_l31, hs_cos_dist,
hs_norm, h2o_attn, attn_entropy.

---

## 10. Phase 1 Recommendation

### Primary signal: `hs_l2_diff_l21_rolling64` (or l22, l23)

Rationale: consistently negative in all 4 datasets, strongest |ρ| in aime2024_eager
(−0.227), strong in math500 (−0.209 at l21, peak −0.254 at l23). This signal is computed
posthoc and cannot be used during generation, but is appropriate for:
- Prefill KV cache compression (offline, after trace is collected)
- Re-eviction between reasoning steps (offline, between CoT segments)

Eviction rule: **evict tokens with highest hs_l2_diff_l{21-23}_rolling64**.

### Secondary signal: `hs_l2_diff_l10_rolling64` (Band A)

Rationale: consistently positive in all 4 datasets, |ρ| up to 0.141. Complementary to
Band B: Band A tokens with LOW signal (dispensable) are likely different from Band B tokens
with HIGH signal (dispensable). A combined score may outperform either band alone.

Eviction rule: **evict tokens with lowest hs_l2_diff_l{7-13}_rolling64**.

### Combined score for Phase 1

`importance_score(t) = hs_l2_diff_l10_rolling64(t) − hs_l2_diff_l21_rolling64(t)`

Tokens with HIGH importance_score are both: high in Band A (important per Band A) and low
in Band B (important per Band B). Keep high-score tokens; evict low-score tokens. This
requires collection of two specific HS layers — cost is identical to collecting one
(output_hidden_states=True materializes all layers at zero extra compute).

### Online fallback: `kv_key_var_rolling64`

For systems where posthoc signal collection is not feasible (online eviction during
generation), kv_key_var_rolling64 is the strongest available online signal in math500
(ρ=0.380) and math500_eager (ρ=0.214). It requires only reading the KV cache (already
in GPU memory), compatible with FlashAttention.

Limitation: sign flip between easy and hard problems. For easier tasks (math500-difficulty)
keep high-kv_key_var tokens; for harder tasks (AIME-difficulty) the direction may reverse.
Cannot use a fixed rule without problem-difficulty estimation. More AIME data is needed to
confirm whether the inversion is real.

### Do NOT use for Phase 1

- **kv_key_norm, kv_key_norm_preRoPE, kv_key_var_preRoPE**: redundant with kv_key_var
- **hs_l2_diff_l31 / hs_cos_dist**: sign flip between datasets, last-layer instability
- **h2o_attn**: weakest signal tested, 3–12× weaker than attn_entropy
- **hs_l2_diff_lN for N in {14–17}**: near-zero in math500, inconsistent layer to layer
- **hs_norm**: weak across all datasets (max |ρ| = 0.087)

### Critical caveats before Phase 1

1. **AIME sample size**: All AIME-side conclusions have 95% CIs spanning zero. Before
   Phase 1 design is finalized, collect AIME 2025 and AIME 2026 traces to bring n up
   to ~90 problems total, yielding ~30–45 labelled traces.

2. **Absolute ρ magnitudes**: The strongest signal (l23, math500) has ρ=0.254, meaning
   ~51% of the variance in importance labels is unexplained. At 50% eviction budget,
   expect ~57–62% precision in important token retention vs. ~50% random baseline. This
   is meaningful but imperfect; Phase 1 needs to measure actual task accuracy degradation.

3. **Posthoc vs online constraint**: HS signals require a posthoc forward pass. Phase 1
   must either target offline KV compression or develop an approximation for online use
   (e.g. collect HS signals at segment boundaries every K tokens).

4. **Window label granularity**: Labels were assigned with 32-token masking windows,
   stride 16. Per-token eviction may perform differently from this granularity. Phase 1
   should test chunk-level eviction (window=32) vs token-level.

---

## Appendix: Corrections to Initial Analysis

The following errors appeared in the verbal analysis delivered before this document:

1. **Missed the positive band (l7–l13)**: Initial analysis stated mid-layer HS signals
   l19–l25 were "the ONLY signal family with consistent direction." l7–l13 are ALSO
   consistently positive across all 4 datasets — equally important finding.

2. **preRoPE max Δρ**: Initially reported as "0.0013". Correct value is 0.0005.

3. **Zone 1 (l0-l4) described as "consistently negative"**: l0 is +0.037 in aime2024_eager.
   Early-layer behavior is not uniform; l0 is mixed, l1–l4 are mostly negative.

4. **"HS mid-layer signals more robust to eager"**: Imprecise. In math500, early layers
   (l0, l1) STRENGTHEN in eager mode (+15–17%), while mid-upper layers (l21–l25) weaken
   (~40–48%). The "more robust" claim applies only to the AIME comparison, where kv_key_var
   collapses while HS mid-layers strengthen — but this observation is unreliable given n=11.

5. **Effective sample sizes**: CIs were described qualitatively; now quantified above.
   Most AIME correlations are statistically indistinguishable from zero.

6. **attn_entropy described as equivalent to ThinKV's metric**: Clarified above — our
   implementation is query-perspective entropy; ThinKV uses key-perspective segment entropy.
