# EpiKV AWS grid, Oct 2026 (872 cells)

Pulled Oct 8 from p5 (8x H200) and p4d (8x A100-40GB). Same code on both boxes
(eviction.py d29dec0d, benchmark.py 621e3eab), torch 2.6.0, transformers 4.57.1,
flash-attn 2.7.4.post1, same Qwen3-14B snapshot (40c06982...).

A cell is complete when every block has partial=false and no problem has an error or
0 generated tokens. All 872 manifest cells (p5/epikv_master.txt) have exactly one such
file under */results_*/. results_consistency/ and a100_replaced/ are not grid cells.

## Hardware per grid (the box that produced each cell file)

| grid | H200 (p5) | A100 (p4d) |
|---|---|---|
| gpqa_q14b | 234 | 0 |
| math_q14b | 85 | 0 |
| gpqa_qwen3 | 57 | 177 (p4d_kvcache/) |
| gpqa_nemo | 0 | 234 |
| math_nemo | 0 | 85 |

gpqa_qwen3 is the only grid split across hardware, by cell.

## Split math_q14b cells

32 long cells were cut on Oct 8 into a finished prefix plus 10-problem pieces
(p5/splits/plan.json, pieces in p5/results_newmodels/splits/), then merged with
merge_splits.py. A test merge of a finished cell reproduced its block exactly. Pre-merge
files: p5/splits/premerge/. Per-problem provenance: p5/splits/provenance/.

## A100 vs H200 check, and the swap

All 32 jobs p4d ran for the Qwen3-14B grids (16 math_q14b pieces, 16 gpqa_q14b cells;
339 problems) were rerun on p5 (p5/results_consistency/). Script: compare_a100_h200.py.

- token-identical generations: 25.4%; identical answers: 85.5%
- MATH: 77.9% vs 77.9% (discordant 7/7, McNemar p=1.00)
- GPQA: A100 48.9% vs H200 44.9% (discordant 12/5, p=0.14)

Greedy outputs diverge across hardware (same-GPU runs are bit-exact), with no
significant accuracy shift at this n. The H200 reruns then replaced the A100 results,
so the Qwen3-14B grids are single-hardware. The A100 versions are in a100_replaced/.

## Reporting caveats

- **`seg` is not EpiKV^-Q.** The Oct 3 manifest builder gave every `kv_seg_hs` line the
  query flags (`--query_sim_mode replace --query_mode qk --query_sim_window 8`), so the
  `seg` cells of every new grid are EpiKV without fill. The paper's EpiKV^-Q row has no
  query term and no fill. Do not report this column under that name. For the Qwen3-8B
  MATH-500 grid (`p5/results_qwen3/`, Sep 28) the flags are unverified; the launch lines
  are on p5.
- **Nemotron** runs use the code's default bands (10/21, located on Llama-8B; Nemotron-Nano-8B
  is Llama-3.1-8B based) and the system prompt "detailed thinking on".
- **Stale files.** `p5/results_newmodels/` holds 37 incomplete `math_nemo_*` files from
  before those cells moved to p4d. The complete math_nemo cells are in
  `p4d_epikv/results_newmodels/`. Select files by completeness (no partial block, no errored
  problem), never by name.
- **Shards.** The n500 ablations in `p5/results_n500/` cover MATH-500 shards 100-400
  (n=395); shard 0 of those configurations is not in this directory.
- Tables built from these files: `~/workspace/emnlp-26/reports/EPIKV_RESULTS_SINCE_ICLR.md`.
