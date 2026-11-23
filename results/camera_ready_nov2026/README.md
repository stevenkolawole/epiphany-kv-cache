# Camera-ready runs, Nov 20-23 2026

Every file checked: no per-problem error, no zero-token problem, every block has n_samples problems.

| Folder | Host | What | Files |
|---|---|---|---|
| p5_epikv_noq | uk-p5 H200 | EpiKV^-Q (kv_seg_hs, tau flags + bands, no query flags) rerun: MATH-500 Qwen3-14B and Nemotron, GPQA-Diamond Qwen3-8B, Qwen3-14B, Nemotron; K 512-4096 | 376 |
| p5_aime_newmodels | uk-p5 H200 | AIME 2024-26, Qwen3-14B and Nemotron, every method, one problem per file | 3420 |
| p5_rkv_aime_newmodels | uk-p5 H200 | released R-KV and EpiKV on R-KV's stack (rkv_venv), AIME, three added models | 216 |
| p5_gpqa_K512 | uk-p5 H200 | GPQA-Diamond K=512 files from results_baselines | 645 |
| babel_aime_qwen3_8b_fill | Babel L40S | AIME Qwen3-8B, the remaining methods (same host as that column's EpiKV/RaaS/no-eviction cells) | 1260 |
| babel_longflow_qwen3_8b | Babel L40S | LongFlow native, Qwen3-8B, GPQA K 512-4096 and AIME K 4096/8192 | 108 |

The EpiKV^-Q files replace the added-model "seg" cells in the five-model tables, which ran with the query read on.
Babel and p5 run different library versions; compare within a host.
Line generator: scripts/p5_grid/make_aime_fill_lines.py.
