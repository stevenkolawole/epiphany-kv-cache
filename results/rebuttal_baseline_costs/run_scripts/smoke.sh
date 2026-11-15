R=/opt/dlami/nvme/kvcache_cost/run.sh
Q="--model Qwen/Qwen3-8B"
$R rkv smoke_check_epikv --mode check_perseq --method epikv --Ks 512 1024
$R rkv smoke_steady_epikv --mode steady --method epikv --Ks 512 --batches 1 2 --windows 2
$R rkv smoke_steady_rkv --mode steady --method rkv --Ks 512 --batches 1 2 --windows 2
$R rkv smoke_steady_h2o --mode steady --method h2o --Ks 512 --batches 1 2 --windows 2
$R rkv smoke_steady_raas --mode steady --method raas --Ks 512 --batches 1 2 --windows 2
$R kvpress smoke_steady_snapkv --mode steady --method snapkv_decoding --Ks 512 --batches 1 2 --windows 2
$R rkv smoke_comp_rkv --mode compaction --method rkv --Ls 4096 --Ks 1024 --batches 1 --repeats 1
$R rkv smoke_comp_epikv --mode compaction --method epikv --Ls 4096 --Ks 1024 --batches 1 2 --repeats 1
$R kvpress smoke_comp_snapkv --mode compaction --method snapkv_decoding --Ls 4096 --Ks 1024 --batches 1 --repeats 1
$R rkv smoke_prefill_rkv --mode prefill --method rkv --Ls 1024 2048 --Ks 1024 --repeats 1
$R kvpress smoke_prefill_snapkv --mode prefill --method snapkv_prefill --Ls 2048 --Ks 1024 --repeats 1
$R rkv smoke_cap_epikv --mode capacity --method epikv --Ks 512 --max_batch 4
$R main smoke_dms --mode dms_decode --method dms --model nvidia/Qwen3-8B-DMS-8x --tokenizer Qwen/Qwen3-8B --batches 1 --steps 256
$R main smoke_steady_epikv_q3 --mode steady --method epikv $Q --band_a_layer 11 --band_b_layer 24 --Ks 512 --batches 1 2 --windows 2
echo SMOKE_ALLDONE >> ~/rebuttal_baseline_costs/status.txt
