# EpiKV new-model grid, Oct 28 - Nov 3 2026 (3,416 cells, all H200)

Pulled Nov 4 from uk-p5 (8x H200). Every line of `scripts/p5_grid/queue_oct28.txt` produced one file here.

- `p5/results_baselines/` (2,580): H2O, R-KV, LongFlow, ThinKV (`thinkv_faithful`) and true EpiKV^-Q (`epikvnoq`: `kv_seg_hs`,
  tau 128 steps, no query term, no fill) for Qwen3-14B, Qwen3-8B and Nemotron-Nano-8B, MATH-500 (20-problem shards) and
  GPQA-Diamond (11-problem shards), K = 512, 1024, 2048, 4096.
- `p5/results_nemo_h200/` (659): the whole Nemotron grid (none, EpiKV, seg, RaaS; LagKV on MATH) rerun on H200. It
  replaces the Oct A100 Nemotron cells in `../aws_grid_oct2026/p4d_epikv/`.
- `p5/results_gpqa_qwen3_h200/` (177): the 177 gpqa_qwen3 cells that ran on A100 in October, rerun on H200.

The Qwen3 none/EpiKV/seg/LagKV/RaaS cells are in `../aws_grid_oct2026/p5/` (`results_newmodels`, `results_qwen3`,
`results_gpqa_new`). With these files every new-model grid is single-hardware (H200).

Tables: `TABLES.md`, written by `python3 scripts/grid_tables.py` (pools shards by dataset index, pairs every method on
the same problems, exact McNemar). n = 495 on MATH-500 and 198 on GPQA-Diamond for every cell.

## Checks (Nov 4)

- **Copy.** Per-directory file counts and md5 of every file match uk-p5.
- **Coverage and completeness.** 3,416 queue lines, 3,416 files, all complete: no partial block, no problem with an error
  or 0 generated tokens. With the October Qwen3 cells, 3,877 complete files read. No problem appears twice in a cell,
  and the problem text at each dataset index is the same in every cell.
- **Code and libraries.** On uk-p5: `eviction.py d29dec0d`, `benchmark.py 621e3eab` (the pins), same md5 as the repo for
  every `src/*.py`; torch 2.6.0+cu124, transformers 4.57.1, flash-attn 2.7.4.post1.
- **Configs.** Each grid's queue lines carry one flag set. EpiKV lines carry the query flags and fill; `epikvnoq` lines
  carry tau and neither. The attention-reading baselines (RaaS, H2O, R-KV, LongFlow, ThinKV) ran on the eager kernel
  (file `meta.attn_impl`); EpiKV, EpiKV^-Q, seg, LagKV and none on FlashAttention-2.
- **Qwen3-8B MATH `seg` (Sep 28) is the true EpiKV^-Q.** Its generations are token-identical to the new `epikvnoq` cells
  (495/495 at every K). The "flags unverified" caveat of the October README is closed for that grid.
- **A100 vs H200 on the rerun cells** (problem records paired across all methods and K, so not independent):
  gpqa_qwen3 37.2 vs 36.9%, discordant 135/130; Nemotron GPQA 39.9 vs 41.1%, 246/279; Nemotron MATH 79.4 vs 78.7%,
  320/264 (McNemar p=0.02 on the pooled records). Token-identical: 40%, 14%, 9%. Only H200 cells are in the tables.

## Caveats

- **`seg` is EpiKV without fill** in every grid except Qwen3-8B MATH (above). It is not EpiKV^-Q; the `epikvnoq`
  rows are.
- **Nemotron** uses Llama-8B's bands (10/21), untuned, and the system prompt "detailed thinking on".
- **No LagKV on GPQA** for the new models; it was never in that grid.
- **`wall_time_s` and peak memory are not comparable** across cells: 4-6 jobs shared each card under MPS, and the
  attention-reading baselines ran eager.
- **p5's live `results_newmodels/` is not the reference copy.** It still holds the A100 pieces of three Qwen3-14B seg
  cells (K4096_s100, K4096_s300, K512_s400) and 37 stale math_nemo files; the H200-swapped version is
  `../aws_grid_oct2026/p5/results_newmodels/`, which `grid_tables.py` reads. It is not copied here.

## What the numbers say (measured, `TABLES.md`)

- **MATH-500.** EpiKV beats RaaS at K=512 on both Qwen3 models (Qwen3-14B 58/19, Qwen3-8B 65/14, p<.001) and ties it at
  every K on Nemotron. It beats H2O, R-KV, LongFlow, LagKV and ThinKV at K=512 and 1024 on all three models (p<.05). At
  K=4096 it is within paired noise of no eviction on all three; there R-KV beats it on Nemotron (7/18, p=0.04).
- **The query term.** EpiKV beats EpiKV^-Q at every K on both Qwen3 models on MATH (p<.01) and at K=512 and 1024 on
  Nemotron.
- **GPQA-Diamond.** EpiKV vs RaaS is not significant at any K on any new model. EpiKV beats H2O, LongFlow and ThinKV at
  K<=1024 on all three (p<.001).
- **Loops.** On MATH with both Qwen3 models, most errors each policy adds over no eviction are cap hits, at every
  K<=2048 and for every policy. They are not for ThinKV on Nemotron MATH (26 of 226 at K=1024), ThinKV on Qwen3-14B
  GPQA, R-KV at K=512 on Qwen3-8B GPQA (14 of 77), or 12 of the 27 policy-budget cells at K<=2048 on Nemotron GPQA,
  where the added errors are few (13-81). The paper's "under every policy and on every model we test" does not hold on this grid as written.
