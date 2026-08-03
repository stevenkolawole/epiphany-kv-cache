# Recovered signal-ablation CSVs — analysis for the rebuttal (2026-08-03)

Provenance: 15 CSVs recovered from dangling git blobs in this repo's object store
(`git cat-file blob <sha>`; SHAs recorded in the recovery script). Schema:
`signal,spearman_rho,p_value,n_pairs,note`, 149 signal rows per dataset.

## 1. Exact reproduction of paper Table 2 (provenance check)

| dataset | layer | paper | recovered CSV | |
|---|---|---|---|---|
| math500 | l10 | +0.112 | +0.1121 | OK |
| math500 | l21 | −0.209 | −0.2091 | OK |
| math500 | l12 | +0.038 | +0.0376 | OK |
| math500_eager | l10 | +0.141 | +0.1411 | OK |
| aime2024_eager | l21 | −0.227 | −0.2275 | OK |
| math500 | l23 | −0.254 | −0.2536 | OK |

→ The recovered CSVs are the paper's source data. Any derived table can cite them.

## 2. NEW: cumulative-attention (h2o_attn) ρ — backs the paper's claim S5

Paper claims (results.tex:30-33) "cumulative attention is the weakest signal measured,
|ρ|≤0.09 on every eager dataset" WITHOUT a table. The recovered values:

| dataset | h2o_attn | attn_entropy | kv_key_var_r64 | kv_val_var_r64 |
|---|---|---|---|---|
| math500_eager | +0.050 | +0.176 | +0.214 | −0.145 |
| aime2024_eager | −0.011 | −0.139 | −0.022 | −0.060 |
| gsm8k_eager | −0.086 | −0.313 | −0.261 | −0.007 |
| aime2025_eager | +0.076 | +0.113 | +0.070 | +0.094 |
| aime2026_eager | +0.068 | +0.099 | +0.101 | +0.200 |

**Claim S5 VERIFIED**: max |ρ(h2o_attn)| = 0.086 ≤ 0.09 across all five eager datasets.
Add this as a table (or App-A extension) — it converts an unsupported claim into a
supported one. (FA2 datasets have no h2o_attn — attention not materialized there.)

## 3. CAUTION: band means across datasets (rolling64 hs_l2_diff)

| dataset | Band A (7–13) | Band B (18–25) | n_eff (App G) |
|---|---|---|---|
| math500 | +0.077 | −0.200 | 72 |
| math500_eager | +0.098 | −0.111 | 78 |
| aime2024 | +0.041 | −0.037 | 11 |
| aime2024_eager | +0.123 | −0.204 | 8 |
| gsm8k_eager | −0.051 | −0.301 | 352 |
| aime2025 | **−0.061** | **+0.094** | ≤5 |
| aime2025_eager | +0.116 | −0.031 | ≤5 |
| aime2026 | +0.074 | −0.021 | ≤5 |
| aime2026_eager | +0.093 | **+0.067** | ≤5 |

- The +A/−B sign pattern holds on the high-power competition-math configs and
  INVERTS or vanishes on aime2025/aime2026 — **but every 2025/2026 value has
  SE ≥ 0.45 (App G), so those signs are uninformative noise.**
- REBUTTAL RULE: do NOT claim the bands replicate on AIME 2025/2026. DO say the
  per-layer data exists, is power-limited exactly as App G computes, and that the
  pooled-ACCURACY run (E12) is the meaningful robustness check at these n.
- GSM8K Band A negative (−0.051) confirms App F's band-shift finding from the
  recovered data directly.

## 4. Files

- `results_recovered/{math500,math500_eager,aime2024,aime2024_eager,gsm8k_eager,
  aime2025,aime2025_eager,aime2026,aime2026_eager}_signal_ablation.csv`
- `results_recovered/phase_OA/*, phase_OB/*` — earlier-phase variants (superseded).
- Regeneration analysis script: inline in the rebuttal work log.
