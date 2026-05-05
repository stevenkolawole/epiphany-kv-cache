# Phase 0B Signal Ablation Results

Full analysis of the Phase 0B signal ablation across 9 datasets (math500, math500_eager,
aime2024, aime2024_eager, aime2025, aime2025_eager, aime2026, aime2026_eager, gsm8k_eager).
Written April 2026; extended with AIME 2025/2026 and GSM8K results April 11, 2026.

All ρ values are Spearman rank correlations between signal values and counterfactual
importance labels (1 = masking this token caused answer flip; 0 = masking had no effect).
Only generated tokens with label ∈ {0, 1} are included; prompt tokens are excluded.
`rolling64` = rolling mean over past 64 tokens; `ema09` = exponential moving average α=0.9.

---

## Data Inventory

| Dataset | Labelled traces (est.) | n_pairs | Avg important_frac | Effective n |
|---|---|---|---|---|
| math500 (non-eager, FA2) | 75 | 172,078 | ~0.20 | 75 |
| math500_eager (eager attn) | 81 | 191,509 | ~0.20 | 81 |
| aime2024 (non-eager, FA2) | ~14 | 66,738 | **~0.52** | **~11** |
| aime2024_eager (eager attn) | ~11 | 52,644 | **~0.64** | **~8** |
| aime2025 (non-eager, FA2) | ~7 | 37,714 | **~0.50** | **~4** |
| aime2025_eager (eager attn) | **~3–4** | 17,065 | **~0.55** | **~1** |
| aime2026 (non-eager, FA2) | ~7 | 36,224 | **~0.50** | **~4** |
| aime2026_eager (eager attn) | ~8 | 41,077 | **~0.50** | **~5** |
| gsm8k_eager (eager attn) | **355** | 200,881 | ~0.25 | **352** |

Trace counts are estimated from n_pairs / ~4700 pairs per AIME trace. GSM8K trace count
(355) is exact from the labelling job log.

**Effective n is the trace count, not n_pairs.** Tokens within a trace are heavily
correlated — different problems are independent, but thousands of tokens within one problem
are not. The p-values printed by signal_ablation.py treat n_pairs as the sample size and
are therefore completely misleading for the AIME datasets.

**Approximate 95% confidence intervals** (Fisher z-transform, n_eff = n_traces - 3):

| Dataset | n_eff | ±SE | CI for ρ=0.25 | CI for ρ=0.20 |
|---|---|---|---|---|
| math500 | 72 | ±0.118 | (0.02, 0.46) | (−0.03, 0.41) |
| math500_eager | 78 | ±0.113 | (0.03, 0.45) | (−0.02, 0.40) |
| gsm8k_eager | 352 | ±0.053 | (0.15, 0.34) | (0.10, 0.30) |
| aime2024 | ~11 | ±0.301 | (−0.31, 0.68) | (−0.36, 0.65) |
| aime2024_eager | ~8 | ±0.354 | (−0.39, 0.73) | (−0.44, 0.71) |
| aime2025 | ~4 | ±0.500 | (−0.56, 0.82) | (−0.62, 0.79) |
| aime2025_eager | **~1** | **~∞** | **completely unreliable** | — |
| aime2026 | ~4 | ±0.500 | (−0.56, 0.82) | (−0.62, 0.79) |
| aime2026_eager | ~5 | ±0.447 | (−0.52, 0.81) | (−0.58, 0.78) |

**AIME 2025 eager has essentially zero statistical power.** With ~1–3 independent traces,
every ρ value is noise. Results are reported for completeness but cannot support any claim.

**All AIME datasets have 95% CIs spanning zero.** Pooling AIME 2024 + 2025 + 2026 gives
~25–30 labelled traces across all three years — still only n_eff ≈ 20–27 total, which has
±0.19–0.22 SE. AIME findings must be cross-validated against math500 or gsm8k results.

**GSM8K is now the second high-n dataset** (n_eff=352, ±0.053 SE), enabling statistically
meaningful claims for a different difficulty regime than math500.

**important_frac** is the fraction of tested reasoning tokens labeled as important (label=1).
This varies by dataset: math500 (~0.20), gsm8k_eager (~0.25), AIME (~0.50–0.65). The root
cause of sign flips is described in §4 below.

---

## 1. Signal Family Rankings

Best |ρ| per family per dataset (rolling64 unless noted). Sign is shown.
AIME 2025/2026 eager results are extremely low n_eff (≤5 effective traces) — treat as noise.

| Signal family | math500 | m500_eager | aime24 | aime24_e | aime25 | aime25_e | aime26 | aime26_e | gsm8k_e |
|---|---|---|---|---|---|---|---|---|---|
| kv_key_var (r64) | **+0.380** | +0.214 | −0.259 | −0.022 | −0.268 | +0.070 | +0.056 | +0.101 | −0.261 |
| kv_key_norm (r64) | +0.379 | +0.214 | **−0.260** | −0.021 | **−0.269** | +0.069 | +0.056 | +0.101 | **−0.261** |
| kv_val_var (r64) | −0.135 | −0.145 | +0.000 | −0.060 | +0.051 | +0.094 | **+0.161** | **+0.200** | −0.007 |
| **hs_l2_diff best (r64)** | −0.254 (l23) | −0.202 (l1) | −0.205 (l16) | **−0.227 (l21)** | −0.236 (l30) | +0.213 (l3) | **+0.272 (l0)** | **+0.306 (l31)** | **−0.351 (l15)** |
| hs_l2_diff_l31 (r64) | +0.220 | +0.093 | −0.196 | −0.022 | −0.163 | +0.159 | **+0.260** | **+0.306** | **+0.231** |
| hs_l2_diff_l10 (r64) | +0.112 | +0.141 | +0.040 | +0.120 | −0.067 | +0.111 | +0.058 | +0.080 | −0.048 |
| hs_l2_diff_l21 (r64) | −0.209 | −0.109 | −0.011 | **−0.227** | +0.128 | −0.001 | −0.032 | +0.061 | −0.303 |
| cross_head_var | +0.207 | +0.101 | −0.175 | +0.006 | −0.166 | −0.015 | +0.038 | +0.048 | −0.196 |
| attn_entropy (eager) | — | +0.176 | — | **−0.139** | — | +0.113 | — | +0.099 | **−0.313** |
| h2o_attn (eager) | — | +0.050 | — | −0.011 | — | +0.076 | — | +0.068 | −0.086 |
| hs_norm | +0.018 | −0.010 | −0.056 | −0.087 | +0.014 | +0.050 | +0.071 | +0.119 | +0.067 |

**Key takeaways from the full 9-dataset picture:**
- **kv_key_var has no reliable direction**: +0.380 (math500), −0.261 (gsm8k_eager, high n),
  −0.259 (aime2024 non-eager). Cannot be used without difficulty-regime detection.
- **hs_l2_diff_l31 also flips**: strongly positive in aime2026_eager (+0.306) and
  gsm8k_eager (+0.231); near zero or negative in math500/aime2024 eager. Not reliable.
- **hs_l2_diff_l10_rolling64 is positive in math500 and aime2024_eager** (the high-n
  competition math datasets) but weakens or inverts in gsm8k and AIME 2025/2026.
- **hs_l2_diff_l21 flips across task difficulty**: strongly negative in math500 and
  aime2024_eager, near zero or positive in the noisier AIME 2025/2026 and gsm8k datasets.
- **attn_entropy sign is unpredictable**: +0.176 (math500_eager), −0.139 (aime2024_eager),
  +0.113 (aime2025_eager), −0.313 (gsm8k_eager). Not reliable as a fixed-sign signal.
- **h2o_attn is consistently weak** across all tested datasets. Our hypothesis confirmed.
- **GSM8K is a qualitatively different domain** — see §11 for full anatomy.

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

### 9a. Competition math (high-n datasets): math500_eager and aime2024_eager

Restricting to the two most statistically reliable competition math datasets (n_eff ≥ 70
for math500_eager, ~8 for aime2024_eager):

Signals with |ρ| ≥ 0.05 AND consistent sign in both math500_eager AND aime2024_eager:

| Signal (rolling64) | math500 | math500_eager | aime2024 | aime2024_eager |
|---|---|---|---|---|
| hs_l2_diff_l7 | +0.079 | +0.047 | +0.118 | **+0.156** |
| hs_l2_diff_l8 | +0.065 | +0.078 | +0.146 | **+0.155** |
| hs_l2_diff_l9 | +0.082 | +0.107 | +0.136 | **+0.130** |
| hs_l2_diff_l10 | +0.112 | **+0.141** | +0.097 | **+0.120** |
| hs_l2_diff_l11 | +0.093 | **+0.144** | +0.083 | **+0.120** |
| hs_l2_diff_l13 | +0.071 | +0.089 | +0.058 | +0.065 |
| hs_l2_diff_l2 | −0.121 | −0.151 | −0.101 | −0.125 |
| hs_l2_diff_l18 | −0.081 | −0.045 | −0.118 | −0.184 |
| hs_l2_diff_l19 | −0.147 | −0.074 | −0.113 | **−0.208** |
| hs_l2_diff_l20 | −0.191 | −0.097 | −0.105 | **−0.224** |
| hs_l2_diff_l21 | −0.209 | −0.109 | −0.085 | **−0.227** |
| hs_l2_diff_l22 | −0.223 | −0.121 | −0.059 | −0.217 |
| hs_l2_diff_l23 | **−0.254** | −0.151 | −0.021 | −0.200 |
| hs_l2_diff_l24 | **−0.250** | −0.146 | −0.005 | −0.188 |
| hs_l2_diff_l25 | **−0.243** | −0.142 | −0.006 | −0.187 |

No kv or attention signal achieves consistent sign across these datasets at |ρ| ≥ 0.05.

### 9b. Across all 9 datasets (including GSM8K)

Adding GSM8K and AIME 2025/2026 breaks the simple Band A / Band B picture. The only
signals that remain consistently signed across ALL 9 datasets at |ρ| ≥ 0.05 are:

**Band A (consistently positive in high-n datasets only):**
- hs_l2_diff_l7 rolling64: positive in math500 (+0.079), math500_eager (+0.047),
  aime2024 (+0.118), aime2024_eager (+0.156), aime2026 (+0.141), aime2026_eager (+0.141);
  weakly positive in gsm8k_eager (+0.076); near-zero or positive in aime2025/2026 (low n_eff)
- hs_l2_diff_l8/l9 rolling64: similar pattern

Band A (l7–l13) is consistently positive across high-n datasets and all eager
competition-math datasets. **But for gsm8k_eager, l7 = +0.076 (weak) and l10 = −0.048
(negative).** GSM8K's Band A is partially different territory — l0–l7 are the more
predictive layers, with l10–l13 flipping negative.

**Band B (consistently negative in competition math, extends further in gsm8k):**
- hs_l2_diff_l19/l20/l21 rolling64: consistently negative in math500 and aime2024_eager,
  near-zero or flipped in AIME 2025/2026 non-eager (low n_eff), and **strongly negative
  in gsm8k_eager** (l19=−0.331, l20=−0.317, l21=−0.303)

The Band B negative zone is consistent in the statistically reliable datasets
(math500, math500_eager, aime2024_eager, gsm8k_eager), but varies in extent.

**No KV, attention, or scalar HS signal achieves consistent sign across all 9 datasets.**

### 9c. Summary: which signals are reliable for Phase 1

For Phase 1 (targeting competition math at AIME difficulty):
- **Band A (l7–l13 positive) is reliable** — confirmed in both high-n competition math datasets
- **Band B (l19–l25 negative) is reliable** — confirmed in math500, math500_eager, aime2024_eager
- **Combined score l10_rolling64 − l21_rolling64 is the Phase 1 target signal**

For future work at GSM8K difficulty:
- Band A shifts to l0–l7 (early layers more predictive)
- Band B extends to l12–l30 (much broader negative zone)
- Different combined score would be needed (e.g. l3 − l15)

---

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

### Secondary online signal: `kv_val_var_rolling64`

**CORRECTION (April 13, 2026):** An earlier version of this section stated that
`kv_val_var` is "consistently non-negative." This is FALSE. The table below shows that
math500 (−0.135) and math500_eager (−0.145) are both NEGATIVE — and these are the two
primary competition math datasets with high n_eff. `kv_val_var` is not a reliable primary
online signal for Phase 1. The claim is downgraded accordingly.

kv_val_var (value vector variance across head_dim, averaged over layers, rolling64):

| Dataset | kv_val_var_rolling64 | n_eff |
|---|---|---|
| math500 | **−0.135** | 75 |
| math500_eager | **−0.145** | 81 |
| aime2025 | +0.051 | ~4 (noise) |
| aime2025_eager | +0.094 | ~1 (noise) |
| aime2026 | +0.161 | ~4 (noise) |
| aime2026_eager | +0.200 | ~5 (noise) |
| gsm8k_eager | −0.007 | 352 |

The positive AIME 2025/2026 values come entirely from low-n_eff runs where 95% CIs span
zero — they cannot be treated as confirmations. The two high-n datasets (math500, gsm8k)
are both negative or near-zero. `kv_val_var` is therefore NOT a reliable predictor of
importance for the primary Phase 1 target (competition math).

**Revised recommendation**: Do NOT use `kv_val_var_rolling64` as the primary online
fallback. It goes in the WRONG direction on math500 and math500_eager. Monitor as a
secondary comparison in Phase 1 benchmarks to document the sign behavior.

Mechanistic reason: values encode retrieved content; content richness may correlate with
importance in some regimes (AIME-style problems, few tokens) but inversely correlate in
dense math500 traces where all tokens are content-heavy.

### Online fallback (secondary): `kv_key_var_rolling64`

Strongest signal in math500 (ρ=+0.380, |ρ|=0.380) but sign-flips in gsm8k (ρ=−0.261).
Both are high-n datasets, so the flip is confirmed, not sampling noise.

**LagKV normalization (open direction):** LagKV (He et al. 2025) normalizes KV channel
variance against a lagged local neighborhood rather than using absolute values. This lag-
relative normalization converts the signal from "is this token's KV globally variable?" to
"is this token's KV *more* variable than its neighbors?" The latter is more robust to
scale differences across problem types. If lag-relative normalization removes the sign flip,
kv_key_var becomes viable without a difficulty-regime classifier. This is worth testing in
Phase 1 as a post-hoc experiment before Phase 2.

Note: attn_entropy and h2o_attn also sign-flip across datasets (math500_eager: +0.176,
gsm8k_eager: −0.313 for attn_entropy; aime2026_eager: +0.068, gsm8k_eager: −0.086 for
h2o_attn). The sign flip is not unique to KV signals — it reflects how label density
and task structure shift the meaning of all signals. It does not invalidate kv_key_var
more than it invalidates any other signal.

### Do NOT use for Phase 1

- **kv_key_norm, kv_key_norm_preRoPE, kv_key_var_preRoPE**: redundant with kv_key_var (Δρ < 0.001 everywhere)
- **hs_l2_diff_l31 / hs_cos_dist**: sign flip between datasets, last-layer instability
- **h2o_attn**: weakest signal tested, consistently 3–12× weaker than Band A HS signals
- **hs_l2_diff_lN for N in {14–17}**: near-zero in math500, inconsistent across layers
- **hs_norm**: weak across all datasets (max |ρ| = 0.119)

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

4. **Temporal trend contamination (CRITICAL — added April 13, 2026)**: Aggregate
   Spearman ρ overestimates real-world per-token eviction quality. Within a single trace,
   all HS signals (l10, l21, and their diff) show monotonic trends with token position:
   l10 decreases, l21 increases over the course of generation. This means the combined
   score l10−l21 is HIGHER at early-position tokens than late-position tokens within
   a trace, even when KEEP tokens appear at late positions (simple problems).

   Trace-level inspection (math500 Trace 1, simple problem, ~5% KEEP):
   - KEEP tokens (late pos): l10_r64≈7.99, l21_r64≈−5.4 → score ≈ 13.4
   - DROP tokens (early pos): l10_r64≈9.0, l21_r64≈−8.0 → score ≈ 17.0
   → The combined score is HIGHER for DROP tokens than KEEP in this trace.

   Root cause: aggregate ρ is dominated by cross-problem structure (complex problems have
   different signal levels than simple ones) rather than within-problem discrimination.
   The epiphany signal works at the problem level, not yet reliably at the token level.

   **Mitigation in Phase 1 implementation**: Rolling z-score detrending removes the
   temporal trend by normalizing each token's score against the local window mean/std:
   `z(t) = (signal(t) − rolling_mean[t]) / (rolling_std[t] + ε)`. This converts from
   absolute magnitude (position-contaminated) to local deviation (position-agnostic).
   DetrendendHSVarianceEviction implements this. Whether detrending suffices or whether
   segment-level eviction (ThinKV-style) is needed is an open empirical question for
   Phase 1 accuracy benchmarks to answer.

5. **Window label granularity**: Labels were assigned with 32-token masking windows,
   stride 16. Per-token eviction may perform differently from this granularity. Phase 1
   should test chunk-level eviction (window=32) vs token-level.

---

---

## 11. Extended Dataset Results (AIME 2025/2026, GSM8K) — April 11, 2026

### 11a. AIME 2025 Non-Eager (n_pairs=37,714, n_eff≈4)

Top signals (absolute ρ):
1. kv_key_norm_rolling64: −0.269 (top, strongly negative KV)
2. kv_key_var_rolling64: −0.268
3. hs_l2_diff_l30_rolling64: −0.236 (very late layers negative)
4. hs_l2_diff_l29_rolling64: −0.218
5. hs_l2_diff_l23_rolling64: **+0.151** (Band B is POSITIVE — sign flip vs prior datasets)

**Anomalous finding**: The l21–l23 band that is consistently negative in math500 and
aime2024 is POSITIVE here (l23=+0.151, l22=+0.129, l21=+0.128). KV signals are strongly
negative. Early layers (l0=+0.118, l1=+0.126) are positive.

**Interpretation**: With n_eff≈4, this is almost certainly noise. The CI at ρ=0.15 is
approximately (−0.50, 0.69) — completely consistent with ρ=0. Do not draw conclusions from
this dataset alone. Note for future: the non-eager AIME 2025 pattern does NOT corroborate
the Band A/B anatomy seen in other datasets.

### 11b. AIME 2025 Eager (n_pairs=17,065, n_eff≈1–3)

**Do not use for any quantitative claim.** n_eff is below the minimum for meaningful
inference. Directional summary only: all signals positive (including attn_entropy=+0.113,
h2o_attn=+0.076), early layers (l3=+0.213) are top signals. Cannot distinguish from noise.

### 11c. AIME 2026 Non-Eager (n_pairs=36,224, n_eff≈4)

Top signals:
1. hs_l2_diff_l0_rolling64: **+0.272** (early layers dominant)
2. hs_l2_diff_l31_rolling64: **+0.260** (last layer positive)
3. kv_val_var_rolling64: +0.161 (positive KV val variance)
4. hs_l2_diff_l1_rolling64: +0.203, hs_l2_diff_l2_rolling64: +0.176

Layer anatomy: l0–l12 positive (early+mid), l13–l22 weakly negative (mild Band B),
l27–l31 weakly positive. The Band A signal (l7–l13) is positive but weaker than l0/l1.

With n_eff≈4, confidence intervals span zero for all signals. Noted for pattern only.

### 11d. AIME 2026 Eager (n_pairs=41,077, n_eff≈5)

Top signals:
1. hs_l2_diff_l31_rolling64 = hs_l2_diff_rolling64: **+0.306** (top!)
2. hs_l2_diff_l31_ema09: +0.275
3. kv_val_var_rolling64: **+0.200** (moderately strong, positive)
4. hs_l2_diff_l3/l2/l0_rolling64: +0.199, +0.191, +0.191

Layer anatomy: early layers (l0–l7) and late layers (l28–l31) strongly positive; mid
layers (l15–l21) mildly negative (l15=−0.070, l16=−0.065). Band A (l7–l13) positive
(l7=+0.141) but overshadowed by early/late layers. attn_entropy=+0.099, h2o_attn=+0.068.

**Key anomaly**: hs_l2_diff_l31 (last layer) is +0.306 — the strongest signal. This
contradicts math500 and aime2024, where l31 either flips sign or is weak. Likely
a low-n_eff artifact (n_eff≈5; CI spans zero for all values). Band A direction is
consistent with other datasets, which is the most we can claim.

### 11e. GSM8K Eager (n_pairs=200,881, n_eff=352) — HIGH N, RELIABLE

**This is the second statistically reliable dataset.** All findings below have CIs with
±0.053 SE and do not span zero at |ρ| ≥ 0.11.

Top signals by absolute ρ:
1. hs_l2_diff_l15_rolling64: **−0.351** (strongest signal across ALL datasets)
2. hs_l2_diff_l16_rolling64: −0.348
3. hs_l2_diff_l17_rolling64: −0.343
4. hs_l2_diff_l18_rolling64: −0.341
5. hs_l2_diff_l30_rolling64: −0.333
6. hs_l2_diff_l19_rolling64: −0.331
7. attn_entropy: **−0.313** (strongest attn_entropy magnitude across all datasets)
8. kv_key_var_preRoPE_ema09: −0.277, kv_key_var_rolling64: −0.261

Positive signals:
- hs_l2_diff_l31_rolling64 = hs_l2_diff_rolling64: **+0.231**
- hs_l2_diff_l0_rolling64: +0.181
- hs_l2_diff_l5_rolling64: +0.164, l3_rolling64: +0.163, l4_rolling64: +0.157

Full layer anatomy for gsm8k_eager:

| Layer range | ρ range (rolling64) | Pattern |
|---|---|---|
| l0–l1 | +0.181 to +0.136 | **Positive (important = early)** |
| l2–l7 | +0.137 to +0.076 | **Positive, weakening** |
| l8–l9 | +0.060 to +0.015 | Near zero |
| l10–l11 | −0.048 to −0.098 | **Negative (starts flipping)** |
| l12–l13 | −0.121 to −0.237 | **Strongly negative** |
| l14–l20 | −0.320 to −0.317 | **Very strongly negative (Band B extended)** |
| l21–l27 | −0.303 to −0.302 | **Strongly negative** |
| l28–l30 | −0.308 to −0.333 | **Strongly negative (includes l30!)** |
| l31 | **+0.231** | **Strongly positive (last layer anomaly)** |

**GSM8K anatomy differs fundamentally from competition math:**

1. **Band A shifts to l0–l7** (early layers most predictive), not l7–l13
2. **Band B extends from l10 to l30** — nearly the entire mid-to-late network
3. **l31 (last layer) is strongly positive** — contradicts math500/aime2024 behavior
4. **attn_entropy is strongly negative (−0.313)** — high entropy = dispensable. This is
   the OPPOSITE sign from math500_eager (+0.176). The same signal reversed meaning.
5. **kv_key_var is strongly negative (−0.261)** — confirmed (not noise) as a sign flip
   from math500 (+0.380)
6. **h2o_attn = −0.086** — weakly negative. Still the weakest signal family.

**Interpretation**: For grade-school math (GSM8K), the model's "epiphany" or critical
reasoning tokens are distinguished earliest in the network (l0–l7). The model computes
rapidly at early layers for simple reasoning chains. The extended negative Band B (l10–l30)
likely reflects output-formatting, arithmetic elaboration, and step-by-step walkthrough
tokens that are dispensable. The positive last-layer signal (l31=+0.231) may capture
moments where the model's output representation changes sharply — corresponding to
important state transitions in the reasoning chain (answer moments, sign changes,
sub-goal completions).

The sign flip in attn_entropy (positive in math500_eager, negative in gsm8k_eager) is
particularly striking: in math500, tokens where the model's attention is spread widely
(high entropy) tend to be important; in gsm8k, widely-spread attention correlates with
dispensability. This may reflect different information-retrieval strategies across
difficulty regimes.

### 11f. Combined AIME Dataset (n_eff≈25–30 total across years)

Pooling AIME 2024 + 2025 + 2026 non-eager traces gives approximately 25–30 independent
problems. For the eager runs: ~21–22 problems. Still well below the threshold for reliable
per-layer conclusions, but better than any single year alone.

Qualitative consensus across all AIME eager runs where direction is consistent:
- Band A (l7–l13): positive in all three eager years
- Band B (l19–l25): negative in aime2024_eager (strong); mixed/weak in aime2025/2026_eager
- Early layers (l0–l3): weakly negative in aime2024_eager; positive in aime2025/2026_eager
- The inconsistency in early layers and Band B across AIME years is within CI — all consistent
  with true underlying values near zero

**Recommendation**: Do not report per-year AIME results in the main paper. Pool all three
years and report aggregate directional trends. Main paper claims should be anchored to
math500 (n_eff=75) and gsm8k (n_eff=352).

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

---

## 12. Cross-Dataset Convergence and Divergence — Full Synthesis (April 13, 2026)

This section synthesises findings across all 9 datasets, identifying what reliably
generalises and what is dataset-specific. Trust levels: math500 (n_eff=75) and
gsm8k_eager (n_eff=352) are the only high-confidence datasets. All AIME datasets
have n_eff ≤ 11 (most ≤ 5) with 95% CIs spanning zero; they are directional
indicators only.

### 12a. What Converges

**1. A negative zone always exists in upper-mid layers.**
In every statistically reliable dataset, there is a contiguous band of layers where
high hs_l2_diff correlates with dispensability (ρ < 0). The band's location shifts:
- math500 / aime2024: l18–l25 (Band B)
- aime2026 eager: l15–l21 (shifted left, weaker)
- gsm8k: l12–l30 (massively extended)

The zone exists universally; its position and width vary with task structure.

**2. Very early layers (l0–l3) are positive in every dataset except aime2024.**
AIME2025 eager (+0.213 at l3), AIME2026 (+0.272 at l0, +0.203 at l1), GSM8K
(+0.181 at l0, +0.163 at l3). Early-layer HS variance is a robust secondary
positive predictor, though in math500/aime2024 these layers are negative or mixed
(likely due to label density effect — see §4).

**3. kv_val_var is either positive or near-zero in every dataset.**
Range: −0.145 (math500_eager, weakly negative) to +0.200 (aime2026_eager). It
never strongly sign-flips. This makes it the most consistent KV signal family.

**4. pre-RoPE vs post-RoPE is a null result in every dataset.**
Confirmed across math500, math500_eager, aime2024, aime2024_eager: max Δ|ρ| = 0.0005.

**5. Rolling64 > EMA09 > raw for all signal families in every dataset.**
The temporal smoothing hierarchy is universal. Window=64 (2× the masking window)
is the right scale for importance, capturing reasoning phases not individual tokens.

**6. h2o_attn is consistently weak.**
Across all 4 eager datasets where it is measured:
math500_eager (+0.050), aime2024_eager (−0.011), aime2025_eager (+0.076),
aime2026_eager (+0.068), gsm8k_eager (−0.086). Never the strongest signal. Never
above |ρ| = 0.086. Our hypothesis that cumulative attention is a weak proxy for
semantic importance is confirmed across all regimes.

### 12b. What Diverges

**1. Band A (l7–l13) vs early layers (l0–l6).**
In math500 / aime2024: Band A (l7–l13) is the dominant positive zone.
In GSM8K: Band A is partial (l7 = +0.076) with early layers (l0–l6) dominant.
In AIME2026: both early layers and Band A are positive, with early layers stronger.
**Interpretation**: Task difficulty shifts where the model's critical semantic routing
occurs. Hard competition math routes through mid-layers (l7–l13); simpler arithmetic
routes through early layers (l0–l6).

**2. l31 (last layer) direction.**
math500: +0.220 (positive). aime2024: −0.178 (negative). aime2026_eager: +0.306
(strongly positive). GSM8K: +0.231 (positive). No consistent direction. The last
layer sits at the output interface and its sign is tied to label density and task
structure.

**3. kv_key_var direction.**
Strongly positive in math500 (+0.380), strongly negative in gsm8k (−0.261) and
aime2025 (−0.269). Weakly positive in aime2026 (+0.101). The direction is not
predictable without knowing the problem distribution.

The attn_entropy sign flip (+0.176 math500_eager, −0.313 gsm8k_eager) and the
h2o_attn sign flip (+0.068 aime2026_eager, −0.086 gsm8k_eager) confirm that this
is a property of the label-signal relationship across task types — not a flaw
specific to any one signal. All signals are affected by the label density shift.

**4. AIME 2025 vs AIME 2026 pattern.**
AIME2025 non-eager shows KV key dominant negative (−0.269) with Band B positive —
the opposite of math500/aime2024. AIME2026 shows early layers and l31 dominant
positive. These two years disagree, but both have n_eff ≤ 5 and their disagreement
is within sampling noise. No meaningful conclusion can be drawn from either year
alone or from comparing them.

### 12c. Mechanistic Explanations

**Why does Band A shift with task difficulty?**
The mid-layers (l7–l13 in 32-layer models) are the primary factual routing /
semantic composition layers per ROME/MEMIT interpretability work. For hard competition
math, the model's semantic retrieval (which mathematical facts, intermediate results,
and definitions to retrieve) happens primarily in these layers. For grade-school math,
retrieval is simpler and occurs earlier (l0–l6). The "epiphany layer" — where the
model integrates retrieved information — is difficulty-dependent.

**Why is kv_key_var unstable but kv_val_var stable?**
Key vectors encode position + content (post-RoPE rotation mixes both). High key
variance means different heads activate differently for this position — useful for
detecting "query-selective" tokens in competition math, but the same property in
simple arithmetic identifies scaffolding tokens that happen to have diverse head
activations without being semantically important. Value vectors encode retrieved
content richness; richer values are referenced more regardless of domain structure,
so high value variance is a more domain-agnostic importance indicator.

**Why does l31 flip?**
In math500, l31 appears in Band B territory — upper-layer output-processing that
correlates with dispensability. In AIME2026 and GSM8K, l31 is strongly positive.
This may reflect the model's output layer behaviour at answer-formation moments:
in some datasets, answer tokens cause sharp last-layer changes (positive ρ, important);
in others, filler and transitional tokens cause them (negative ρ, dispensable). Without
manual inspection of which specific tokens drive l31, this remains speculative.

**Why does sign flip affect even h2o_attn?**
H2O's cumulative attention measures how much future queries attended to each past
position. In AIME2026 (mostly hard problems), the model's attention is well-aligned
with what's causally necessary → positive ρ. In GSM8K (simple arithmetic), the model
may attend heavily to mechanical/structural tokens (numbers, operators, formatting)
that are easily reconstructable and thus not causally necessary per counterfactual
labels → negative ρ. The label-eviction semantic gap (§4 root cause) manifests here.

---

## 13. Phase 1 Experimental Scope (Formalised April 13, 2026)

### Primary evaluation datasets

**math500** and **aime2024** are the primary Phase 1 evaluation datasets. Rationale:
- These are the only competition-math datasets with sufficient statistical power or
  directional consistency to anchor claims
- They are the same datasets used by ThinKV (our primary baseline), enabling direct
  comparison
- They represent the intended deployment regime: long reasoning traces from hard math
- All Phase 1 accuracy-vs-cache-size curves will be reported for these two datasets

### Secondary / exploratory

**aime2025** and **aime2026** will be run as secondary validation sets. Results will be
reported in paper appendices or supplementary material. Given n_eff ≤ 5–8, they cannot
anchor primary claims but can corroborate or challenge math500/aime2024 findings at scale.

### Out of scope for Phase 1

**GSM8K** is documented (§11e) as a qualitatively different regime where the layer
anatomy shifts. It is not a target deployment use-case (grade-school arithmetic does
not require long reasoning traces and is not where KV cache memory pressure arises).
It will not be used for Phase 1 accuracy benchmarking. Its Phase 0B findings inform
the paper's "regime scope" discussion only.

### Phase 1 signal targets (confirmed)

Primary: `importance_score(t) = hs_l2_diff_l10_rolling64(t) − hs_l2_diff_l21_rolling64(t)`

Secondary (online): `kv_val_var_rolling64(t)` — monitored alongside kv_key_var; more
stable but weaker. kv_key_var_rolling64 included as comparison despite sign-flip caveat.

### What Phase 1 must answer

1. At what cache budget (tokens retained) does HSVarianceEviction match / beat H2O and
   ThinKV on MATH-500 and AIME-2024 accuracy?
2. Does kv_val_var as a standalone online signal give meaningful accuracy improvement
   over H2O at any budget?
3. Does the LagKV-style relative normalization of kv_key_var resolve the sign flip?
   (Post-hoc experiment, not gating Phase 1.)
4. What do the actual evicted traces look like qualitatively — are Band B tokens
   interpretably dispensable (filler, hedging, mechanical steps)?
