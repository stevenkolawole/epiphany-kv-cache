# Full grid, card 7, one process per (mode, method). Results: ~/rebuttal_baseline_costs/<tag>.json
R=/opt/dlami/nvme/kvcache_cost/run.sh
Q="--model Qwen/Qwen3-8B"
QB="--band_a_layer 11 --band_b_layer 24"
DMS="--model nvidia/Qwen3-8B-DMS-8x --tokenizer Qwen/Qwen3-8B"
CTX="--Ks 4096 8192 16384 32768"
for m in none epikv epikv_noq h2o raas rkv; do $R rkv steady_$m --mode steady --method $m; done
$R kvpress steady_snapkv_decoding --mode steady --method snapkv_decoding
$R kvpress steady_none_kvpressvenv --mode steady --method none
$R main steady_none_qwen3 --mode steady --method none $Q --Ks 512 1024 2048 4096 8192
$R main steady_epikv_qwen3 --mode steady --method epikv $Q $QB
$R main dms_decode_dms --mode dms_decode --method dms $DMS --batches 1 4 16 --steps 8192
for m in epikv epikv_noq h2o raas rkv none; do $R rkv compaction_$m --mode compaction --method $m; done
$R kvpress compaction_snapkv_decoding --mode compaction --method snapkv_decoding
for m in epikv epikv_noq h2o raas rkv; do $R rkv capacity_$m --mode capacity --method $m; done
$R rkv capacity_none --mode capacity --method none $CTX
$R kvpress capacity_snapkv_decoding --mode capacity --method snapkv_decoding
$R main capacity_none_qwen3 --mode capacity --method none $Q $CTX
$R main capacity_epikv_qwen3 --mode capacity --method epikv $Q $QB
P="--Ls 1024 2048 4096 8192 16384 32768 --Ks 1024"
for m in none epikv h2o raas rkv; do $R rkv prefill_$m --mode prefill --method $m $P; done
$R kvpress prefill_snapkv_prefill --mode prefill --method snapkv_prefill $P
$R kvpress prefill_none_kvpressvenv --mode prefill --method none $P
$R main prefill_none_qwen3 --mode prefill --method none $Q $P
$R main prefill_dms --mode prefill --method dms $DMS $P
echo GRID_DONE >> ~/rebuttal_baseline_costs/status.txt
