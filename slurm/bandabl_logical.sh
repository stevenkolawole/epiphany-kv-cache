#!/bin/bash
#SBATCH --job-name=bandabl_log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/home/skolawol/workspace/kvcache/slurm_logs/bandabl_log_%A_%a.out
# Band ablation (BgY3 Q5 / i2R71 Q4) rerun under LOGICAL post-eviction
# positions; the rewound run it replaces was cancelled 09-10. Same manifest.
# Results in $HOME (visible from the login node). Babel = transformers 5.2.0:
# compare only within this directory.
set -u
cd /home/skolawol/workspace/kvcache
export HF_HOME=/data/hf_cache/skolawol TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1
PY=/home/skolawol/miniconda3/envs/kvcache/bin/python
OUT=/home/skolawol/workspace/kvcache/results_bandabl_logical
mkdir -p "$OUT" logs/bandabl_logical
MANIFEST=/home/skolawol/workspace/kvcache/slurm/bandabl_manifest.tsv
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$LINE" ]; then echo "no manifest line for task $SLURM_ARRAY_TASK_ID"; exit 1; fi
TAG=$(echo "$LINE" | cut -f1)
METHOD=$(echo "$LINE" | cut -f2)
EXTRA=$(echo "$LINE" | cut -f3)
echo "[bandabl] task=$SLURM_ARRAY_TASK_ID $TAG method=$METHOD extra=[$EXTRA] $(date)"
$PY - <<'PREFLIGHT' || { echo "[bandabl] ABORT: no CUDA device on $(hostname)"; exit 1; }
import sys, torch
ok = torch.cuda.is_available()
print(f"[preflight] torch={torch.__version__} cuda={ok} name={torch.cuda.get_device_name(0) if ok else 'none'}")
sys.exit(0 if ok else 1)
PREFLIGHT
if [ "$METHOD" = "none" ]; then
    # shellcheck disable=SC2086
    $PY scripts/benchmark.py \
        --dataset math500 --methods none --cache_sizes 512 1024 2048 4096 \
        --n_samples 25 $EXTRA --attn_impl flash_attention_2 --resume --logical_positions \
        --output "$OUT/${TAG}.json" > "logs/bandabl_logical/${TAG}.log" 2>&1
else
    # shellcheck disable=SC2086
    $PY scripts/benchmark.py \
        --dataset math500 --methods "$METHOD" --cache_sizes 1024 2048 \
        --n_samples 100 --start_idx 0 $EXTRA --attn_impl flash_attention_2 --resume --logical_positions \
        --output "$OUT/${TAG}.json" > "logs/bandabl_logical/${TAG}.log" 2>&1
fi
echo "[bandabl] rc=$? $TAG $(date)"
