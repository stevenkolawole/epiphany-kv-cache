# What the KV-eviction baselines cost on one GPU (rebuttal, submission 50503)

Measured time and GPU memory of EpiKV against the baselines' own costs: per compaction, per decode token at
batch 1, 4 and 16, the largest batch one card decodes, and prefill. All numbers come from the JSON records in
`records/`; `TABLES.md` is generated from them (`python3 scripts/baseline_cost_tables.py <folder>`).
Predictions were written before any run in `PREDICTION.md`; the outcome against them is at the end of this file.
`analytical.json` holds FLOPs, bytes and transient memory per compaction from `scripts/baseline_cost_model.py`
(analytical, not measured).

## Headline (measured; Llama-8B, A100-SXM4-40GB, GiB)

At budget K, steady decode (three 128-step windows after a warmup window; compaction every 128 steps for EpiKV,
R-KV and kvpress, every step for H2O and RaaS). "Compaction extra" = the compaction step minus a normal step.

| method | ms/token K=1024 B=1 | ms/token K=4096 B=1 | ms/token K=4096 B=16 | compaction extra, K=4096 B=1 | peak GiB K=4096 B=16 | max batch K=1024 / 4096 |
|---|---|---|---|---|---|---|
| no eviction | 33.0 | 33.0 | 57.7 (cache 4096-4608) | - | 24.1 | 47 at 4k context, 23 at 8k, 11 at 16k, 5 at 32k |
| EpiKV | 34.4 | 34.5 | 63.6 | 13.2 ms | 32.3 | 82 / 22 |
| EpiKV, batched path (same decisions) | 34.0 | 34.1 | 57.9 | 10.7 ms | 23.4 | 128 / 45 |
| EpiKV^-Q | 33.8 | 33.7 | 62.7 | 8.2 ms | 26.0 | 112 / 34 |
| H2O | 36.7 | 39.7 | 200.4 | 6.7 ms every step | 31.1 | 94 / 23 |
| RaaS | 36.9 | 39.9 | 200.4 | 6.8 ms every step | 31.1 | 94 / 23 |
| R-KV (official) | 34.5 | 35.7 | OOM | 165.5 ms | OOM | 64 / 7 |
| kvpress SnapKV DecodingPress | 37.3 | 37.3 | 65.0 | 36.4 ms | 25.8 | 106 / 34 |

Compaction from a long cache (L tokens down to K=1024, batch 1; step time includes the decode forward; no
eviction's decode step at L is 33.6 / 46.2 / 46.5-54 / 47-56 ms):

| L | EpiKV | H2O | R-KV | kvpress | R-KV transient |
|---|---|---|---|---|---|
| 4096 | 51.7 ms | 40.0 | 219.1 | 75.6 | 2.40 GiB |
| 8192 | 73.2 | 44.8 | 788.4 | 96.8 | 9.59 GiB |
| 16384 | 96.1 | 54.4 | OOM | 127.1 | - |
| 32768 | 158.8 | 86.2 | OOM | 209.0 | - |

Prefill, batch 1 (ms; peak GiB):

| L | FlashAttention-2 | EpiKV harness (all hidden states) | H2O / RaaS (eager + weights) | R-KV (compress to 1024) | kvpress SnapKV | Qwen3-8B base | DMS |
|---|---|---|---|---|---|---|---|
| 2048 | 160 (15.4) | 161 (15.9) | 284 (24.3) | 210 (16.2) | 190 (15.3) | 181 | 1973 |
| 4096 | 313 (15.9) | 314 (16.9) | OOM | 476 (18.6) | 342 (15.5) | 348 | 4997 |
| 8192 | 650 (16.9) | 650 (18.8) | OOM | 1271 (26.9) | 700 (16.0) | 721 | 11436 |
| 16384 | 1441 (18.8) | 1443 (22.5) | OOM | OOM | 1512 (16.9) | 1595 | 29567 |
| 32768 | 3516 (22.6) | 3519 (30.1) | OOM | OOM | 3651 (18.7) | 3898 | 89598 |

DMS (nvidia/Qwen3-8B-DMS-8x, its own decode, teacher-forced): 55.2-57.1 ms/token at batch 1 (windows after the first) with 668-1888 live
entries per head (context 1152-8320), against 45.9-46.0 ms for base Qwen3-8B at cache 1024-8192 (+20-24% at equal
or larger live length); at batch 16, 60.9-61.1 ms against base 48.8 (cache 1024), 65.8 (4096), 107.9 (8192).
EpiKV on Qwen3-8B (bands 11/24): 48.2 ms/token at K=1024 batch 1 (1.05x base), 71.4 ms at K=4096 batch 16;
max batch 131 / 73 / 38 / 19 at K = 512 / 1024 / 2048 / 4096, against base Qwen3-8B 41 / 20 / 10 / 5 at context
4k / 8k / 16k / 32k.

## Where each method's cost comes from (measured, with the analytical count that explains it)
- R-KV builds an L x L key-similarity matrix per layer and KV head at every compaction, with an int64 index matrix
  and its int64 zeros beside it (`third_party/rkv/utils.py`, `cal_similarity`). Transient measured 2.55 GiB at
  K=4096 batch 1 and 9.6 GiB from an 8k cache (analytical 19 B x 8 heads x L^2: 2.5 and 9.5 GiB). Hence the OOM at
  batch 16 (K=4096), at batch 4 from 8k, and on any prompt of 16k or more (it compresses the prompt at prefill).
  Its decode step itself runs FlashAttention, so at batch 1 its amortised cost is small (1.05-1.08x).
- H2O and RaaS need the attention weights every step, so they run eager attention, and they evict one token every
  step with a copy of the whole cache. At batch 16, K=4096 that costs 143 ms per token over no eviction (3.5x).
  Eager prefill with weights keeps 32 layers of L^2 x 32 heads: OOM at 4k.
- kvpress DecodingPress(SnapKV) recomputes 64 window queries per layer at each compaction (36 ms at K=4096 batch 1,
  242 ms at batch 16) and keeps a 256-state hidden-state buffer per layer (64 MiB per sequence), which is why its
  max batch at K=512 (128) is below EpiKV's (152). It compresses one layer at a time, so its transient is small and
  its max batch at K >= 1024 is above EpiKV's (106 vs 82 at 1024; 34 vs 22 at 4096).
- EpiKV: 6.9 / 13.2 ms per compaction at K=1024 / 4096 batch 1 (0.05-0.10 ms per token amortised) plus about
  0.3 ms per sequence per step for the hidden-state score (two host syncs). Its batch > 1 path calls the batch-1
  policy once per sequence (see Protocol). In `steady_epikv` at K=4096 batch 16 the policy's own time is 9.1 ms per
  token and a normal step takes 62.3 ms against 56.2 ms without eviction (EpiKV^-Q, same per-sequence update:
  8.0 ms per token, 62.7 ms per token); the compaction adds 87-161 ms every 128 steps. The arithmetic is small (analytical: 0.5-1 GFLOP and 0.6-2.4 GB per
  compaction at batch 1). The compaction gathers a full copy of the kept cache before freeing the old one
  (transient = K x 128 KiB x batch, 8.0 GiB at K=4096 batch 16), and the policy banks a 16 KiB hidden state per
  cached decode token: these two are why H2O/RaaS fit 4-24% more sequences and kvpress more at K >= 1024 on the per-sequence path; the
  batched path removes both (see "Batched EpiKV policy").
- DMS has no compaction step; its per-step cost (+20-24% over base at equal live length) is its paged cache and
  eviction bookkeeping in the reference HuggingFace code. Its prefill (the same code, 510-token chunks) is 6-23x
  slower than the base model's FlashAttention prefill.

## Throughput per card at each method's largest batch (measured, Llama-8B)
Steady decode with every window restarted from the cache at K (`--restart_windows`), at the largest batch that ran
512 steps without OOM (`tput_*`; the capacity probes run 12 steps, and per-step torch.cat in transformers'
DynamicCache fragments memory over longer runs, so a few probe-passing batches fail here; each failing size was
retried in a fresh process and failed again, records `tput_*_fresh_*`). Tokens/s per card = batch x 1000 / ms per
step.

| method | K=2048: batch, tokens/s | K=4096: batch, tokens/s |
|---|---|---|
| EpiKV, batched path | 80 (lower bound: 81-87 not tried in a fresh process), 478 | 45, 278 |
| EpiKV, per-sequence path | 42, 407 | 22, 255 |
| EpiKV^-Q (keeps 832 / 1384 after a compaction) | 58, 430 | 34, 265 |
| kvpress SnapKV DecodingPress | 63, 492 | 34, 267 |
| R-KV (official) | 21, 393 | 6, 133 |
| H2O | 47, 165 | 23, 82 |
| RaaS | 47, 165 | 23, 82 |
| no eviction at context 8k / 16k / 32k | 23, 169 / 11, 90 / 5, 47 | |

At these batches decode is bound by reading the cache (and by DynamicCache copying it on every step), so doubling
the batch nearly doubles the step time: the batched path doubles EpiKV's max batch at K=4096 (22 -> 45) but adds
9% tokens/s. kvpress leads at K=2048 by 3%, EpiKV leads at K=4096 by 4%; both are near the measured noise.

## Batched EpiKV policy (src/epikv_batched.py, method `epikv_batched`)
- Cause (profile, `records/profile_epikv_K4096_B16.json`, torch.profiler over 8 normal steps and one compaction at
  K=4096, batch 16): the per-sequence path makes 2 host syncs per sequence per step (`aten::item` 256 in 8 steps)
  and slices every layer's cache and every hidden state once per sequence (about 1,750 `aten::slice` per step);
  the policy took 14.1 ms of a 70 ms profiled step. The compaction is per-sequence decision code (1,008 syncs) plus
  a gather of the full kept cache before the old one is freed.
- Change: one KVSegHSEviction per sequence still makes every decision with the unmodified code. Per step, the
  band differences are computed with the original (1, d) shapes and read back with one transfer; at a compaction
  the keep masks are captured and the cache is gathered one layer at a time in place; only the last 8 banked hidden
  states stay on the GPU (the policy reads only those and the bank's length). The per-sequence path is unchanged.
- Bit identity (`records/check_epikv_batched.json`, `scripts/check_epikv_batched.py`): real DeepSeek-R1-Distill-
  Llama-8B reasoning traces (data_slim60.jsonl, a different trace stream per row), teacher-forced, 5 compactions per
  cell, K in {1024, 4096} x batch in {1, 4, 16}: kept index tensors torch.equal at all 30 compactions, every step's
  score identical (0 mismatches over 3,840 steps), bank length equal to the cached decode tokens, gathered K/V equal
  to an index_select with the reference indices. All 6 cells PASS. CPU unit test `tests/test_epikv_batched.py`
  (tiny random Llama, batch 1-3, with and without the query read): 4 passed.
- Speed (steady, median of 3 windows, spread <= 0.13 ms): batch 16: 45.2 -> 38.8 ms/token at K=1024 (-14%),
  63.6 -> 57.9 at K=4096 (-9%); batch 4: 38.7 -> 36.9 (-4.5%) and 38.8 -> 37.2 (-4.2%); batch 1: 34.4 -> 34.0 and
  34.5 -> 34.1 (-1%, under the 3% bar). Profiled policy time per step 14.1 -> 3.4 ms; 0 `aten::item` syncs.
  Compaction step unchanged within 3% (223 -> 219 ms at K=4096 batch 16): its decisions are the same per-sequence code.
- Memory: compaction transient at K=4096 batch 16 7.98 -> 0.20 GiB; resident 24.0 -> 23.0 GiB (bank). One
  compaction from L=4096 to K=1024 at batch 16: 250 -> 188 ms, transient 1.26 -> 0.13 GiB.
- Capacity (fresh process per probe, `capacity_fresh_epikv_batched_K*.json`): 296 / 128 / 88 / 45 at
  K = 512 / 1024 / 2048 / 4096, against 152 / 82 / 43 / 22 for the per-sequence path, 188 / 94 / 47 / 23 for
  H2O/RaaS, 128 / 106 / 63 / 34 for kvpress, 156 / 64 / 24 / 7 for R-KV. At K=1024 the limit is fragmentation:
  batch 136 fails while 34.0 GiB are allocated and 4.8 GiB are reserved but free (expandable segments were not
  tried, for any method). The in-process search for this path drifted after its first OOM (batch-1 peak 19.4 GiB
  at K=4096 against 15.6 fresh) and is kept as `records/superseded/capacity_epikv_batched.in_process_after_oom.json`.
- Kept: every batch > 1 change exceeds the 3% bar; batch 1 does not move.

## DMS largest batch (lowest priority; `records/dms_capacity_dms.json`)
Through NVIDIA's reference code, prefilling 8,192 tokens per sequence and then decoding 8 steps: 8 sequences fit, 9
do not (peak 34.9 GiB at 8). Unlike the other capacity rows, this includes DMS's own prefill, which dominates its
memory (peak grows about 2.4 GiB per sequence while its live cache after prefill is about one eighth of the context).
Base Qwen3-8B at 8k context without eviction: 20 (prefill excluded); EpiKV on Qwen3-8B at K=1024: 73.

## Corrections (2026-11-15)
- `steady_epikv_noq` (now `records/superseded/steady_epikv_noq.cache_below_K.json`) did not measure decode at
  budget K, at any batch size (B=1 and 4 as well as 16). EpiKV^-Q has no fill: per 128-token segment it keeps at
  most 64/32/8 tokens by key-variance tertile, so a compaction from K+128 leaves 416 / 552 / 832 / 1384 tokens at
  K = 512 / 1024 / 2048 / 4096 (prompt 128 + recent 128 + tier caps). The steady protocol ran four windows on one
  cache, so after the warm-up window's compaction the cache climbed back for hundreds of steps and the three
  recorded windows held 553-1768 tokens with no compaction. It is the method's behaviour, not a harness fault, but
  those cells are not comparable with the other rows. `steady_epikv_noq_restart` (`--restart_windows`) starts every
  window from the cache at K, so each window holds 127 steps at K..K+126 and one compaction from K+128 (checked:
  n_compactions = 1 in every window). Audit of all other steady records: every window reaches cache length >= K and
  has 1 compaction (EpiKV, R-KV, kvpress) or 128 (H2O, RaaS); none other is affected.
- EpiKV^-Q's capacity record is valid as it stands: each probe starts at K+120 and asserts the compaction on step
  8 from K+128, the point of peak memory; its larger max batch reflects a smaller kept copy and no hidden-state
  bank (weight 0).
- The batch-16 slowdown statement in "Where each method's cost comes from" now cites `steady_epikv`'s own policy
  time instead of the invalid EpiKV^-Q cells.

## Protocol
- Card: sk-p4d card 7 (A100-SXM4-40GB, driver 595.91.07), alone (run.sh refuses to start if the card holds more
  than 500 MiB); cards 0-6 were busy with another user's jobs throughout (shared host CPU and PCIe).
- Models: DeepSeek-R1-Distill-Llama-8B (rev 6a6f4aa4), Qwen/Qwen3-8B (rev b968826d), nvidia/Qwen3-8B-DMS-8x
  (rev 231c1ed2, loaded through the vendored `third_party/dms`). bf16.
- Stacks (one per method family; compare a method only with the no-eviction row from its own stack):
  `/opt/dlami/nvme/kvcache_rkv/venv` torch 2.7.0+cu126, transformers 4.55.4, flash-attn 2.8.3 (Llama: none,
  EpiKV, EpiKV^-Q, H2O, RaaS, R-KV); `/opt/dlami/nvme/kvcache/venv` same torch, transformers 4.57.1 (Qwen3-8B,
  DMS); `/opt/dlami/nvme/kvcache_cost/venv_kvpress` same torch and flash-attn, transformers 4.57.1, kvpress 0.5.5
  (kvpress needs transformers >= 4.56; its own no-eviction rows are `*_none_kvpressvenv`, within 1% of the 4.55.4
  rows). The shared venvs were not modified.
- Text: MATH-500 problems and solutions concatenated, tokenized; caches built by chunked prefill of one sequence
  and replicated over the batch; decode steps teacher-force the next tokens at logical positions. The first 128
  tokens are the prompt, the rest count as decoded; EpiKV's per-token state is replayed from the prefill's hidden
  states through the policy's own update.
- Method settings: EpiKV = kv_seg_hs_fill, refresh_tau 128 steps, query_sim replace, query_mode qk, window 8,
  bands 10/21 (Qwen3 11/24); EpiKV^-Q = kv_seg_hs, tau 128 steps, no fill; keep_recent_k 128 for all harness
  methods. EpiKV^-Q's steady numbers come from `steady_epikv_noq_restart` (see Corrections).
  R-KV = official code a329f29 unmodified (window 8, kernel 7, mix_lambda 0.1, retain_ratio 0.2, direction last),
  compression forced every 128 steps (its step_length rule with a fixed phase). kvpress = DecodingPress(
  SnapKVPress(window 64, kernel 5), compression_interval 128, target_size K, buffer 256). H2O/RaaS = src/eviction.py
  as in the paper, eager attention with output_attentions.
- Batch > 1: EpiKV's policy is batch-1 code; `PerSeq` runs one policy object per sequence, captures each keep mask
  and gathers the batch once. Checked against the unmodified batch-1 path (`records/smoke2_check_epikv*.json`): same
  keep mask and same kept K/V, K in {512, 1024, 4096}. H2O and RaaS run as written, which averages attention over
  the batch and shares one keep mask (same work as per-sequence masks, so the cost is theirs; decisions are not).
  R-KV and kvpress are batched natively.
- Timing: torch.cuda.synchronize() around every step and policy call; memory from torch.cuda.max_memory_allocated,
  reset before each step (resident = allocated before the step, transient = peak - resident).
- Capacity: doubling then bisection on batch; a probe passes if the batch decodes through one compaction (EpiKV,
  R-KV: compaction on step 8 of 12; H2O/RaaS: 8 evicting steps; kvpress: 132 steps) without CUDA OOM; no eviction
  decodes 8 steps at the given context. Prefill of new requests is not part of a probe. Precision: the search stops
  when the gap is under max(1, max_batch/32) (`first_oom` in the records). In-process searches drifted
  after their first OOM in two configurations, EpiKV on Qwen3-8B (transformers 4.57.1; batch-1 peak 17.0 GiB after
  an OOM against 15.6 GiB fresh) and batched EpiKV on Llama-8B (4.55.4; 19.4 against 15.6 GiB); the cause was not
  traced. Both use `scripts/capacity_search.py` (a fresh process per probe). Every other method's batch-1 capacity
  peak equals its steady-run peak, so their in-process searches are not affected.

## Commands
On sk-p4d, from `/opt/dlami/nvme/kvcache_cost/kvcache` (scripts/benchmark.py and src/ at epiphany-kv-cache
f3b46c1; md5 benchmark.py 96a619f3, eviction.py 10a11beb). `run.sh <venv> <tag> <args>` sets HF_HOME, card 7,
offline mode and writes `~/rebuttal_baseline_costs/<tag>.json` and `logs/<tag>.log`. The box-side scripts are
copied in `run_scripts/` (kvpress venv: `setup_kvpress_venv.sh`). The grid, in order (`grid.sh`, then
`followup.sh`, then `cap_q3.sh`; smokes in `smoke.sh`):

    for m in none epikv epikv_noq h2o raas rkv; do run.sh rkv steady_$m --mode steady --method $m; done
    run.sh kvpress steady_snapkv_decoding --mode steady --method snapkv_decoding
    run.sh kvpress steady_none_kvpressvenv --mode steady --method none
    run.sh main steady_none_qwen3 --mode steady --method none --model Qwen/Qwen3-8B --Ks 512 1024 2048 4096 8192
    run.sh main steady_epikv_qwen3 --mode steady --method epikv --model Qwen/Qwen3-8B --band_a_layer 11 --band_b_layer 24
    run.sh main dms_decode_dms --mode dms_decode --method dms --model nvidia/Qwen3-8B-DMS-8x --tokenizer Qwen/Qwen3-8B --batches 1 4 16 --steps 8192
    for m in epikv epikv_noq h2o raas rkv none; do run.sh rkv compaction_$m --mode compaction --method $m; done
    run.sh kvpress compaction_snapkv_decoding --mode compaction --method snapkv_decoding
    for m in epikv epikv_noq h2o raas rkv; do run.sh rkv capacity_$m --mode capacity --method $m; done
    run.sh rkv capacity_none --mode capacity --method none --Ks 4096 8192 16384 32768
    run.sh kvpress capacity_snapkv_decoding --mode capacity --method snapkv_decoding
    run.sh main capacity_none_qwen3 --mode capacity --method none --model Qwen/Qwen3-8B --Ks 4096 8192 16384 32768
    P="--Ls 1024 2048 4096 8192 16384 32768 --Ks 1024"
    for m in none epikv h2o raas rkv; do run.sh rkv prefill_$m --mode prefill --method $m $P; done
    run.sh kvpress prefill_snapkv_prefill --mode prefill --method snapkv_prefill $P
    run.sh kvpress prefill_none_kvpressvenv --mode prefill --method none $P
    run.sh main prefill_none_qwen3 --mode prefill --method none --model Qwen/Qwen3-8B $P
    run.sh main prefill_dms --mode prefill --method dms --model nvidia/Qwen3-8B-DMS-8x --tokenizer Qwen/Qwen3-8B $P
    for K in 512 1024 2048 4096; do capacity_search.py --python <main venv python> --K $K --max_batch 256 \
        --output capacity_fresh_epikv_qwen3_K$K.json -- --method epikv --model Qwen/Qwen3-8B --band_a_layer 11 --band_b_layer 24; done

Defaults: `--Ks 512 1024 2048 4096 --Ls 4096 8192 16384 32768 --batches 1 4 16 --windows 4 --repeats 3`.
`steady_rkv` was rerun after a fix: the first run built its K >= 1024 caches with the previous cell's R-KV budget
still set, so the caches were 512 long (the invalid file is kept on p4d as `steady_rkv.invalid_budget_carryover.json`).
The superseded in-process Qwen3 EpiKV capacity search is `records/capacity_epikv_qwen3.json` (biased low; use
`capacity_fresh_epikv_qwen3_K*.json`). `status.txt` is the run log.

## What was not verified
- Batch > 1 for EpiKV and the baselines uses identical replicated sequences; per-sequence keep counts are equal by
  construction (PerSeq asserts it). Real batches with different lengths would need padding or ragged caches.
- Timing on a host whose other seven cards were at 100% utilisation; repeat spreads are in TABLES.md (mostly under
  1%; no eviction at batch 16, K=4096 spreads 3.9 ms as its cache grows).
- DMS was run through its reference HuggingFace code; a fused-kernel deployment may differ. DMS capacity was not
  measured.
- No accuracy is measured here; costs only.

## Outcome against PREDICTION.md
1. R-KV transient: held (2.55 vs 2.5 GiB; 9.6 vs 9.5 GiB; OOM at batch 16 K=4096 and on prompts >= 16k).
2. R-KV time: K=4096 held within 2x (165 ms vs 60-120 predicted), but R-KV costs 12.5x EpiKV there, not 5x;
   K=1024 failed (24 ms, 3.5x EpiKV, not within 2x): its per-layer fixed costs are larger than the count predicted.
3. EpiKV about linear in L: held (policy about 18 ms at 4k, 100 ms at 32k); transient = one kept copy: held (8.0 GiB).
4. H2O/RaaS: at batch 16 K=4096 3.2x EpiKV's ms/token: held (>= 20%). At batch 1 within 10% of EpiKV: held at
   K=1024 (7%), failed at K=4096 (15%).
5. Batch 1, K <= 2048 within 10% of no eviction: held for EpiKV (4%) and R-KV (5%); failed for kvpress (12%)
   and H2O/RaaS (9-14%).
6. Capacity: EpiKV within 15% of H2O/RaaS at K=1024 (82 vs 94): held; R-KV below them: held (64; 7 at K=4096);
   no eviction 23 / 11 / 5: held; kvpress below EpiKV: failed at K >= 1024 (106 vs 82), held at K=512 (128 vs 152).
7. Prefill: eager OOM at 4k held; EpiKV harness +7.5 GiB at 32k held (predicted 8.3 GiB) and it fit; R-KV OOM at
   16k held; kvpress SnapKV +3.3% at 32k over its own stack's FlashAttention prefill held.
8. DMS: +20-24% at equal live length held; faster than base at equal context past 4k held at batch 16 (61 vs 66 ms
   at 4k, 108 at 8k), failed at batch 1 (57 vs 46 ms at 8k).

## Code versions during the grid
`scripts/baseline_costs.py` changed three times while the grid ran, each change outside the code paths of the
runs already finished: the `oom_phase` field (recorded from the compaction runs on), the R-KV budget reset before a
cache build (from `compaction_rkv` and the `steady_rkv` rerun on), and `--probe_only` (for `capacity_search.py`).
The committed file is the final one, identical to the copy on sk-p4d. Measurements ran src/eviction.py and scripts/benchmark.py
from f3b46c1 on the box (md5 10a11beb, 96a619f3); later commits by another lane add WindowRuleEviction and a kvpress
method to those files without touching any measured code path. On 2026-11-15 baseline_costs.py gained `--restart_windows`,
`--first_fit`, the `epikv_batched` method, the `dms_capacity` mode and a device-agnostic dummy in `prime`; none
changes the code path of a method measured before.
