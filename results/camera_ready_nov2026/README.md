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

The added-model "seg" cells in the five-model tables ran with the query read on; p5_epikv_noq holds the same cells without it.
Babel and p5 run different library versions; compare within a host.
Line generator: scripts/p5_grid/make_aime_fill_lines.py.

Copies of finished runs that were only on the remote boxes:

| Folder | Host | What | Files |
|---|---|---|---|
| p5_rkv_aime_paper_models | uk-p5 H200 | released R-KV, AIME, Llama-8B and Qwen-7B, K 4096/8192, cap 16384, all 90 problems | 36 |
| p5_gpqa_paper_models_baselines | uk-p5 H200 | GPQA-Diamond H2O and ThinKV, Llama-8B and Qwen-7B, K 1024-4096 | 216 |
| babel_longflow_qwen7b_aime | Babel L40S | LongFlow native, AIME, Qwen-7B, K 8192 | 18 |
| babel_longflow_qwen3_math_gpqa | Babel L40S | LongFlow native, MATH-500 Qwen3-8B and Qwen3-14B K 512-4096, GPQA K 1024-4096 | 268 |

Queued on p5 behind the other project's jobs (scripts/p5_grid/p5_camera_gate.sh appends scripts/p5_grid/p5_camera_fill.txt
once the GPUs are idle), output `results_camera_fill/`: GPQA-Diamond K=512 EpiKV^-Q, H2O and ThinKV for Llama-8B and
Qwen-7B (the existing K=512 "seg" lines for these two carry the query flags), RaaS K=512 piece s121 for both (5 OOM problems
each), and AIME 2024 Qwen-7B EpiKV K=8192 (10 of 30 problems were OOM on p4d A100-40GB).
