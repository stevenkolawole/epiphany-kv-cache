# Model Variants & Testing Strategy

## Why Model Choice Matters

This project targets long reasoning traces — sequences of 5,000–60,000 tokens generated during internal chain-of-thought reasoning. Vanilla instruction-tuned models produce 100–500 token outputs for math problems; the KV cache barely fills. The signal we are testing (hidden-state variance as importance proxy) only becomes meaningful under genuine cache pressure from long traces.

**Primary targets**: DeepSeek-R1-Distill variants. These produce the long CoT traces that stress the KV cache and make eviction decisions matter.

---

## Model Variants

### Primary (Reasoning Models — Long Traces)

- **DeepSeek-R1-Distill-LLaMA-8B** (`deepseek-ai/deepseek-r1-distill-llama-8b`)
  - DeepSeek-R1 distilled into LLaMA-3.1-8B backbone
  - Typical trace length: 2,000–20,000 tokens on MATH-500; up to 60,000 on AIME
  - **Primary experimental target** — most accessible at 8B, longest traces
  - Used by both ThinKV and ChunkKV — direct head-to-head comparisons possible

- **DeepSeek-R1-Distill-Qwen-7B** (`deepseek-ai/deepseek-r1-distill-qwen-7b`)
  - Same distillation approach, Qwen-7B backbone
  - Cross-architecture robustness check; run after LLaMA-8B variant is validated
  - Not used by existing baselines — results here are additional generalization evidence only

- **Qwen2.5-Math-7B-Instruct** (`Qwen/Qwen2.5-Math-7B-Instruct`)
  - Math-specialized reasoning model; used by RaaS as their primary test model
  - Required for direct comparability with RaaS results
  - Generates longer, denser math reasoning traces than general instruction models

### Secondary (Reasoning Models — Larger)

- **DeepSeek-R1-Distill-Qwen-14B** (`deepseek-ai/deepseek-r1-distill-qwen-14b`)
  - Used by ThinKV; longer and more complex traces than 8B variants
  - Memory-intensive — run only after method is validated at 8B scale

### Control / Comparison (Vanilla Instruction Models)

- **LLaMA-3.1-8B-Instruct** (`meta-llama/Llama-3.1-8B-Instruct`)
  - Short traces (~100–400 tokens), minimal cache pressure
  - Used by ChunkKV — results directly comparable
  - Use to confirm our method does *not* regress on short-trace models

- **Qwen2-7B-Instruct** (`Qwen/Qwen2-7B-Instruct`)
  - Same role as LLaMA-3.1-8B-Instruct; used by ChunkKV

**Note**: Mistral-7B-Instruct-v0.3 is *not* a reasoning model and does not appear in any of our primary baselines (ThinKV, RaaS). Removed.

---

## Testing Priority

### Phase 0B (Signal Ablation — COMPLETE)
- **Model**: DeepSeek-R1-Distill-LLaMA-8B only
- **Tasks**: MATH-500 (100 samples), AIME 2024 (30 samples), AIME 2025/2026 (30 each, in progress), GSM8K (500, in progress — difficulty-robustness check for Band A/B layer anatomy)
- **Outcome**: Band A (l7–l13, positive ρ) and Band B (l18–l25, negative ρ) identified. Combined score `l10_rolling64 − l21_rolling64` is Phase 1 candidate. h2o_attn is weakest signal tested. See `experiments/phase0b_ablation_results.md`.

### Phase 1 (Eviction Implementation + Baseline Comparison — NEXT)
- **Models**: DeepSeek-R1-Distill-LLaMA-8B (primary), Qwen2.5-Math-7B-Instruct (RaaS comparability)
- **Tasks**: MATH-500, AIME 2024, GSM8K (low-pressure control)
- **Purpose**: Implement `HSVarianceEviction` using Phase 0B signal; establish accuracy vs. cache-size curves against H2O, ThinKV, RaaS

### Phase 2 (Robustness / Generalization)
- **Models**: Add DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-14B; add LLaMA-3.1-8B-Instruct and Qwen2-7B-Instruct as negative controls
- **Tasks**: Add LiveCodeBench (required for ThinKV head-to-head), HotpotQA (Gap F non-monotonic recall — not a ThinKV/RaaS comparison point)
- **Purpose**: Confirm results generalize across model size and architecture

---

## Expected Trace Characteristics

| Model | Typical trace length | Cache pressure at 512 | Baseline paper |
|---|---|---|---|
| DeepSeek-R1-Distill-LLaMA-8B | ~2,468 tokens (MATH-500); ~9,020 (AIME) | High | ThinKV, ChunkKV |
| DeepSeek-R1-Distill-Qwen-7B | Similar to LLaMA-8B variant | High | — (generalization only) |
| Qwen2.5-Math-7B-Instruct | Long math-specialized traces | High | RaaS |
| DeepSeek-R1-Distill-Qwen-14B | Longer than 8B; ~14,166 tokens (LiveCodeBench) | Very high | ThinKV |
| LLaMA-3.1-8B-Instruct | 100–400 tokens | None (never fills) | ChunkKV |
| Qwen2-7B-Instruct | 100–400 tokens | None | ChunkKV |

*Trace length figures for DeepSeek-R1-Distill-LLaMA-8B and Qwen-14B are from ThinKV's reported averages. All other figures are estimates; actual measurements will replace these after Phase 0 data collection.*

---

## Key Empirical Questions by Model

**DeepSeek-R1-Distill-LLaMA-8B** (primary):
- ✅ Does hidden-state variance correlate with counterfactual importance labels? Yes — Band A (l7–l13) consistently positive ρ across datasets.
- ✅ Which signal variant wins? `hs_l2_diff_l10_rolling64 − hs_l2_diff_l21_rolling64` combined score. Rolling64 universally best smoother.
- **Open**: Does our method outperform ThinKV/RaaS at the same cache budget on MATH-500, AIME 2024? (Phase 1)

**Qwen2.5-Math-7B-Instruct** (RaaS comparability):
- Do our results on GSM8K, MATH-500, AIME reproduce the RaaS baseline numbers?
- Does our method outperform RaaS's LRU-timestamp eviction on these benchmarks?

**DeepSeek-R1-Distill-Qwen-7B** (generalization):
- Do optimal signal variant settings from the LLaMA-8B ablation transfer to Qwen architecture?
- Does the pyramidal layer-variance pattern (Dimension 3) hold in Qwen as well?

**LLaMA-3.1-8B-Instruct / Qwen2-7B-Instruct** (negative control):
- Does our method cause accuracy degradation even when eviction is minimal/never triggered?
- Are variance scores well-calibrated on short non-reasoning traces?

---

## How to Run

```bash
# Primary target — reasoning model, MATH-500
python scripts/poc_harness.py \
  --model deepseek-ai/deepseek-r1-distill-llama-8b \
  --dataset MATH-500 \
  --cache_size 128 \
  --eviction_method semantic

# Cross-architecture check
python scripts/poc_harness.py \
  --model deepseek-ai/deepseek-r1-distill-qwen-7b \
  --dataset MATH-500 \
  --cache_size 128 \
  --eviction_method semantic

# Negative control — vanilla model
python scripts/poc_harness.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset MATH-500 \
  --cache_size 512 \
  --eviction_method semantic
```
