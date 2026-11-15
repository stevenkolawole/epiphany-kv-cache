# Predictions and pass criteria, written before any run (2026-11-14)

Setting: sk-p4d card 7 (A100-SXM4-40GB) alone; Llama-8B = DeepSeek-R1-Distill-Llama-8B (32 layers, 32 query heads,
8 KV heads, head dim 128, 128 KiB of bf16 KV per token, 16.1 GB of weights). Methods: EpiKV, EpiKV^-Q, R-KV official
(a329f29), H2O, RaaS, kvpress 0.5.5 DecodingPress(SnapKV); DMS (nvidia/Qwen3-8B-DMS-8x) against base Qwen3-8B and
EpiKV on Qwen3-8B (bands 11/24). Compaction interval 128 for EpiKV, R-KV and the kvpress decoding press; H2O and RaaS
evict every step. L = cache length at a compaction, K = budget, B = batch.

Anchor (measured, results/rebuttal_cost_split/llama.json, batch 1): EpiKV compaction 6.0 / 7.4 / 9.6 / 14.1 ms at
K = 512 / 1024 / 2048 / 4096, decode forward 33.1 ms/token.

## Predictions (analytical unless marked)
1. R-KV compaction memory. `cal_similarity` builds, per layer and KV head, an L x L cosine matrix (bf16, 2 B), a
   boolean mask (1 B), an int64 index matrix from `torch.where` and the int64 `zeros_like` it reads (8 + 8 B):
   about 19 B per element, so the transient is about 19 * B * 8 * L^2 bytes (layers run one after another).
   L = K + 128: K=1024 -> 0.2 GB per sequence; K=4096 -> 2.7 GB per sequence, so B=16 at K=4096 OOMs on 40 GB.
   Prompt compression at prefill (L = prompt length): 8k -> 10 GB, 16k -> 41 GB, so R-KV OOMs on prompts >= 16k.
2. R-KV compaction time. Similarity matmul 2 * 8 * L^2 * 128 FLOP per layer (1.2 TFLOP at K=4096 over 32 layers)
   plus several passes over the L^2 arrays (~90 GB of traffic at K=4096): 60-120 ms per compaction at K=4096, B=1,
   about 5x EpiKV's 14 ms; at K=1024 about 13x less, within 2x of EpiKV's 7.4 ms. Grows with B linearly.
3. EpiKV compaction scales about linearly in L: reads every layer's keys once (L * 64 KiB) for the key variance,
   one layer's keys for the 8-query relevance, and copies the kept cache (K * 128 KiB). Transient = one copy of the
   kept cache (K * 128 KiB * B). Its per-step score update costs ~0.2 ms per sequence (two host syncs).
4. H2O and RaaS. At decode the attention weights are small (B * 32 * L * 2 B per layer), so eager-with-weights costs
   little over FlashAttention at batch 1 (<10% at K <= 4096). Their dominant extra is a full cache copy EVERY step
   (index_select of all layers): 2 * K * 128 KiB * B bytes of traffic per token, about 1x-2x the attention's own
   cache read. Prediction: at B=16, K=4096, H2O/RaaS ms/token >= 20% above EpiKV; at B=1 within 10% of EpiKV.
5. Batch-1 decode ms/token at K <= 2048: every method within 10% of no eviction (weights-bound), as the paper found.
6. Capacity (sequences per card at budget K). Resident = 16.1 GB + B * (K + 128) * 128 KiB, plus per method:
   EpiKV one kept-cache copy at compaction and a hidden-state bank of 16 KiB per cached decode token (12.5% of the
   KV bytes); H2O/RaaS one cache copy every step; R-KV 19 * 8 * L^2 B per sequence; kvpress a hidden-state buffer of
   up to 256 states x 32 layers x 8 KiB = 64 MiB per sequence. Predicted max batch at K=1024: EpiKV ~ H2O ~ RaaS
   within 15% of each other (~70-80), R-KV below them (~45), kvpress below EpiKV; at K=4096: R-KV < 8, others ~20.
   No eviction at context 8k / 16k / 32k: ~22 / ~11 / ~5.
7. Prefill (batch 1, logits for the last position only). Eager with attention weights (H2O, RaaS) keeps 32 layers of
   L^2 * 32 heads * 2 B: 8.6 GB at 2k, 34 GB at 4k -> OOM at 4k (as in the paper's tab:prefill-latency). EpiKV's
   harness asks for all hidden states (33 * L * 8 KiB): +8.9 GB at 32k, which fits at 16k and may OOM at 32k;
   the method itself needs only the last token's state at two layers. R-KV OOMs at >= 16k (prediction 1).
   kvpress SnapKV prefill computes 64 window queries x L per layer: < 5% over FlashAttention, no OOM to 32k.
8. DMS per-step time at equal live cache length: 10-30% above base Qwen3-8B (its eviction gate and paged-cache
   bookkeeping); at equal total context it is faster than the base once the context passes ~4k.

## Pass criteria
- Real code path: every compaction record shows the cache shrinking to the budget (R-KV, EpiKV with fill, H2O,
  RaaS, kvpress: exactly K); a cell without that is a failure, not a number. DMS live entries < total tokens.
- PerSeq (EpiKV at batch > 1) is checked against the unmodified batch-1 path: same kept cache, bit for bit.
- Exclusive card: nvidia-smi shows card 7 empty before each launch; nothing else runs on it during a run.
- Noise: three measured repeats (steady: three 128-step windows after a warmup window); a difference between methods
  counts only if it exceeds 3% and the repeat spread.
- A prediction holds if the measurement is within 2x of it for time and within 25% for memory; OOM predictions hold
  if the OOM happens at the predicted point (one grid step either way is a partial hold).
