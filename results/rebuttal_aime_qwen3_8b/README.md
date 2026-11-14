# Qwen3-8B on AIME 2024-2026 (pooled n=90), Babel L40S, Nov 14 2026

Greedy, cap 32,768 new tokens. Methods: no eviction (`none`), DMS (`nvidia/Qwen3-8B-DMS-8x`, learned eviction, 8x),
EpiKV (`kv_seg_hs_fill`, bands 11/24, refresh 128) and RaaS at K=4096 and 8192. One problem per file,
`<method>_<dataset>_k<K>_s<idx>.json`; `none` is tagged k0 and DMS k8 (its compression ratio).

Stack: Babel L40S only, torch 2.7.0+cu126, transformers 4.57.1, flash-attn 2.8.3 (the Table 2 versions;
`setup_env_and_smoke.sbatch`); harness at epiphany-kv-cache 737678b (benchmark.py 96a619f3, eviction.py 10a11beb).
Queue: `lines.txt` run by `aime_worker.sbatch` (claim-based workers, 8 general + 24 preempt). 540/540 files, no errors.

Read-out: `python3 ../../../rebuttals/scripts/aime_qwen3_8b_pool.py .` (accuracy, Wilson 95%, cap-hit, paired exact
McNemar against EpiKV at each K). DMS live KV comes from `kv_live` in each DMS record.
