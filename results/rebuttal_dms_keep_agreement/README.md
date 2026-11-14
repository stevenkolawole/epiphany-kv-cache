# Does a training-free policy keep what the DMS checkpoint learned to keep? (Qwen3-8B, Babel L40S)

DMS (`nvidia/Qwen3-8B-DMS-8x`) generates greedily (cap 16,384) on GPQA-Diamond problems 0-29 and MATH-500 problems 0-29;
`scripts/dms_keep_agreement.py record` saves its per-token, per-(layer, KV head) eviction decisions (`traces/<set>_<crc>.npz`).
`forced` replays each DMS trace through the base Qwen3-8B under EpiKV (`kv_seg_hs_fill`, bands 11/24), RaaS and random
eviction at K=1024 and saves which generated tokens each policy still holds at the end (`traces/<set>_<crc>__<method>_k1024.npy`).
Stack and harness as `../rebuttal_aime_qwen3_8b` (transformers 4.57.1, flash-attn 2.8.3; this script at 133fa93).
`lines.txt` is the run as executed (60 records, 180 replays, all rc=0), through `overlap_worker.sbatch`.

Read-out: `python3 ../../../rebuttals/scripts/dms_keep_agreement_readout.py traces`. Scored tokens are generated tokens older
than DMS's 512-token decision window. DMS keep score = share of the 288 (layer, head) pairs keeping the token; AUC = how well
that score separates the tokens a policy kept from those it evicted (0.5 = none), bootstrap over problems.
