#!/bin/bash
#SBATCH --job-name=tau_log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/home/skolawol/workspace/kvcache/slurm_logs/tau_log_%A_%a.out
# Eviction-frequency sweep (tau in {32,128}) under LOGICAL positions. The
# tau=1 pairing partner is the kv_seg_hs cell of results_bandabl_logical
# (the "diff" cell), same problems, same cluster.
set -u
cd /home/skolawol/workspace/kvcache
export HF_HOME=/data/hf_cache/skolawol TOKENIZERS_PARALLELISM=false TQDM_DISABLE=1
PY=/home/skolawol/miniconda3/envs/kvcache/bin/python
OUT=/home/skolawol/workspace/kvcache/results_tau_logical
mkdir -p "$OUT" logs/tau_logical
MANIFEST=/home/skolawol/workspace/kvcache/slurm/tau_manifest.tsv
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$LINE" ]; then echo "no manifest line for task $SLURM_ARRAY_TASK_ID"; exit 1; fi
METHOD=$(echo "$LINE" | cut -f1)
DATASET=$(echo "$LINE" | cut -f2)
SHARD=$(echo "$LINE" | cut -f3)
NSAMP=$(echo "$LINE" | cut -f4)
BUDGETS=$(echo "$LINE" | cut -f5)
CAP=$(echo "$LINE" | cut -f6)
TAG=$(echo "$LINE" | cut -f7)
EXTRA=$(echo "$LINE" | cut -f8)
NAME="${TAG}_${DATASET}_s${SHARD}"
echo "[tau] task=$SLURM_ARRAY_TASK_ID $NAME flags=[$EXTRA] cap=$CAP $(date)"
$PY - <<'PREFLIGHT' || { echo "[tau] ABORT: no CUDA device on $(hostname)"; exit 1; }
import sys, torch
ok = torch.cuda.is_available()
print(f"[preflight] torch={torch.__version__} cuda={ok} name={torch.cuda.get_device_name(0) if ok else 'none'}")
sys.exit(0 if ok else 1)
PREFLIGHT
# shellcheck disable=SC2086
$PY scripts/benchmark.py \
    --dataset "$DATASET" --methods "$METHOD" \
    --cache_sizes $BUDGETS --max_new_tokens "$CAP" \
    --n_samples "$NSAMP" --start_idx "$SHARD" \
    --attn_impl flash_attention_2 --resume --logical_positions $EXTRA \
    --output "$OUT/${NAME}.json" > "logs/tau_logical/${NAME}.log" 2>&1
rc=$?
errs=$(grep -c ERROR "logs/tau_logical/${NAME}.log" 2>/dev/null); errs=${errs:-0}
echo "[tau] rc=$rc errors=$errs $NAME $(date)"
if [ "$errs" -gt 0 ] 2>/dev/null; then
    echo "[tau] QUARANTINE $NAME"
    mv "$OUT/${NAME}.json" "$OUT/${NAME}.json.ERRORED" 2>/dev/null
fi
